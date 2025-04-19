############################################################################################
# Reference: https://huggingface.co/blog/annotated-diffusion
############################################################################################


import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


############################################
# Helper Functions
############################################

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t) 
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


############################################
# Beta Schedule
############################################

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


############################################
# Diffusion trainer
############################################

class DiffusionTrainer:
    def __init__(
        self,
        model,
        dataloader,
        text_encoder=None,
        timesteps=1000,
        beta_schedule='cosine',
        image_size=256,
        channels=1,
        batch_size=32,
        lr=2e-4,
        device=None,
        results_folder="./results",
        loss_type="huber",
        scheduler_type="step",
        scheduler_params = None
    ):
        self.model = model
        self.dataloader = dataloader
        self.text_encoder = text_encoder 
        self.timesteps = timesteps
        self.image_size = image_size
        self.channels = channels
        self.batch_size = batch_size
        self.loss_type = loss_type

        # Setup device
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        if self.text_encoder is not None:
            self.text_encoder.to(self.device)

        # Create folder to save the results
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        (self.results_folder / 'visualization').mkdir(exist_ok=True)
        (self.results_folder / 'plots').mkdir(exist_ok=True)
        (self.results_folder / 'checkpoints').mkdir(exist_ok=True)

        # Set up optimizer
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        # Set up learning rate scheduler
        if scheduler_type == "step" and scheduler_params != None:
            self.scheduler = StepLR(
                self.optimizer, 
                step_size=scheduler_params.get("step_size", 5), 
                gamma=scheduler_params.get("gamma", 0.9)
            )
        elif scheduler_type == "cosine" and scheduler_params != None:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_params.get("T_max", 100),
                eta_min=scheduler_params.get("eta_min", 1e-6)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        # Setup beta schedule
        self._setup_beta_schedule(beta_schedule)

        # History tracker
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'val_psnr': [],
            'val_ssim': [],
            'val_psnr_steps': {},
            'val_ssim_steps': {}
        }

    def _setup_beta_schedule(self, schedule_type):
        """Set up beta schedule and related parameters"""
        if schedule_type == 'linear':
            self.betas = linear_beta_schedule(self.timesteps)
        elif schedule_type == 'cosine':
            self.betas = cosine_beta_schedule(self.timesteps)
        elif schedule_type == 'quadratic':
            self.betas = quadratic_beta_schedule(self.timesteps)
        elif schedule_type == 'sigmoid':
            self.betas = sigmoid_beta_schedule(self.timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule_type}")
    
        # Move to device
        self.betas = self.betas.to(self.device)
        
        # Calculate diffusion process parameters
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Coefficients for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Posterior variance
        self.posterior_variance = (self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)).clamp_min(1e-20)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start, device=self.device)
            
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_noisy, noise
        
    def encode_text(self, captions):
        if self.text_encoder is None:
            return None
        
        with torch.no_grad():
            _, text_embeddings = self.text_encoder.encode_batch(captions)
            text_embeddings = text_embeddings.to(self.device)
        
        return text_embeddings
    
    def p_losses(self, x_start, t, noise=None, context=None):
        """Define loss function"""
        if noise is None:
            noise = torch.randn_like(x_start, device=self.device)

        # Add noise
        x_noisy, noise_target = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict noise
        if getattr(self.model, 'self_condition', False) is True:
            with torch.no_grad(): 
                if getattr(self.model, 'use_cross_attention', False) and (context is not None):
                    predicted_noise_1 = self.model(x_noisy, t, context=context)
                else:
                    predicted_noise_1 = self.model(x_noisy, t)
            
            if getattr(self.model, 'use_cross_attention', False) and (context is not None):
                predicted_noise = self.model(
                    x_noisy,
                    t,
                    context=context,
                    x_self_cond=predicted_noise_1
                )
            else:
                predicted_noise = self.model(x_noisy, t, x_self_cond=predicted_noise_1)

        else:
            if getattr(self.model, 'use_cross_attention', False) and (context is not None):
                predicted_noise = self.model(x_noisy, t, context=context)
            else:
                predicted_noise = self.model(x_noisy, t)

        # Calculate loss
        if self.loss_type == 'l1':
            loss = F.l1_loss(noise_target, predicted_noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise_target, predicted_noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(noise_target, predicted_noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        return loss
    
    def forward(self, x_start, t=None, captions=None, context=None):
        """Forward pass for evaluation"""
        batch_size = x_start.shape[0]
        if t is None:
            t = torch.ones(batch_size, device=self.device, dtype=torch.long) * (self.timesteps // 2)
        
        if captions is not None and context is None and self.text_encoder is not None:
            context = self.encode_text(captions)

        # Add noise
        x_noisy, noise = self.q_sample(x_start, t)
        
        # Predict noise
        with torch.no_grad():
            if hasattr(self.model, 'use_cross_attention') and self.model.use_cross_attention and context is not None:
                predicted_noise = self.model(x_noisy, t, context=context)
            else:
                predicted_noise = self.model(x_noisy, t)
        
        # Calculate loss
        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        
        return x_noisy, predicted_noise, noise, loss.item()

    @torch.no_grad()
    def _predict_eps(self, x, t, context=None):
        if getattr(self.model, "use_cross_attention", False) and context is not None:
            return self.model(x, t, context=context)
        return self.model(x, t)
        
    @torch.no_grad()
    def p_sample(self, x, t, t_index, context=None):
        """Single step of DDPM sampling"""
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        eps = self._predict_eps(x, t, context)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * eps / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        noise = torch.randn_like(x)
        variance = extract(self.posterior_variance, t, x.shape)

        #return model_mean + torch.sqrt(variance) * noise
        result = model_mean + torch.sqrt(variance) * noise
        return torch.clamp(result, min=-1.0, max=1.0)    # -> [-1, 1] 

    @torch.no_grad()
    def p_sample_loop(self, shape, context=None, n_steps=None, show_progress=True):
        """Complete DDPM sampling loop. Returns all images."""
        device = self.device
        b = shape[0]

        if n_steps is not None and n_steps < self.timesteps:
            step_size = self.timesteps // n_steps
            timesteps = torch.arange(0, self.timesteps, step_size, device=device)[:n_steps]
        else:
            timesteps = torch.arange(0, self.timesteps, 1, device=device)
            
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []

        if show_progress:
            iterator = tqdm(range(len(timesteps)), desc='DDPM sampling', total=len(timesteps))
        else:
            iterator = range(len(timesteps))
    
        for i in iterator:
            t_index = len(timesteps) - i - 1
            t = torch.full((b,), timesteps[t_index], device=device, dtype=torch.long)
            img = self.p_sample(img, t, t_index, context)
            imgs.append(img.cpu().detach())

        return imgs

    @torch.no_grad()
    def sample(self, batch_size=32, captions=None, context=None, n_steps=None, show_progress=True):
        """Method for generating samples"""
        image_shape = (batch_size, self.channels, self.image_size, self.image_size)

        if captions is not None and context is None and self.text_encoder is not None:
            context = self.encode_text(captions)

        samples = self.p_sample_loop(
            shape=image_shape, 
            context=context, 
            n_steps=n_steps,
            show_progress=show_progress
        )

        return samples[-1]
    
    @torch.no_grad()
    def sample_from_text(self, text, batch_size=1, n_steps=None, show_progress=True):
        if isinstance(text, str):
            text = [text] * batch_size
        elif len(text) == 1 and batch_size > 1:
            text = text * batch_size
        
        if len(text) != batch_size:
            raise ValueError(f"Number of captions ({len(text)}) must match batch_size ({batch_size})")
    
        context = self.encode_text(text)
        
        return self.sample(
            batch_size=batch_size, 
            context=context, 
            n_steps=n_steps,
            show_progress=show_progress
        )
        
    def compute_metrics(self, original, generated):       
        # to [0, 1]
        generated = (generated + 1) / 2
        original  = (original  + 1) / 2 

        # Initialize metrics
        psnr_metric = PeakSignalNoiseRatio().to(self.device)
        ssim_metric = StructuralSimilarityIndexMeasure().to(self.device)
        
        # Move images to compute device
        original = original.to(self.device)
        generated = generated.to(self.device)
        
        # Calculate PSNR
        psnr_val = psnr_metric(generated, original).item()
        
        # Calculate SSIM
        ssim_val = ssim_metric(generated, original).item()

        return {"psnr": psnr_val, "ssim": ssim_val}

    def validate(self, val_loader, n_steps=200):
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        batch_count = 0

        for batch in tqdm(val_loader, desc="Validation"):
            x = batch["pixel_values"].to(self.device)      
            captions = batch["caption"]
            batch_context = self.encode_text(captions)
        
            batch_size = x.shape[0]

            # Evaluate with forward method
            _, _, _, loss = self.forward(x, context=batch_context)
                
            # Generate samples and compute metrics
            with torch.no_grad():
                generated = self.sample(
                    batch_size=batch_size, 
                    context=batch_context, 
                    n_steps=n_steps, 
                    show_progress=False
                )
                metrics = self.compute_metrics(x, generated)

            total_loss += loss
            total_psnr += metrics["psnr"]
            total_ssim += metrics["ssim"]
            batch_count += 1

        # Calculate average metrics
        avg_loss = total_loss / batch_count
        avg_psnr = total_psnr / batch_count
        avg_ssim = total_ssim / batch_count
        
        # Update history
        self.history["val_losses"].append(avg_loss)
        self.history["val_psnr"].append(avg_psnr)
        self.history["val_ssim"].append(avg_ssim)
        
        return {"loss": avg_loss, "psnr": avg_psnr, "ssim": avg_ssim}
    
    def train(self, epochs, start_epoch=0, val_loader=None, n_steps=200, save_model_every_epoch=True):
        for epoch in range(start_epoch, epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for step, batch in enumerate(tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
                if isinstance(batch, dict):
                    x = batch["pixel_values"].to(self.device)
                    
                    # Get caption
                    captions = batch.get("caption", None)
                    context = self.encode_text(captions) if captions is not None and self.text_encoder is not None else None
                else:
                    x = batch.to(self.device)
                    context = None
                
                batch_size = x.shape[0]
                
                # Sample random timesteps
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                
                # Compute loss
                self.optimizer.zero_grad()
                loss = self.p_losses(x, t, context=context)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1

            # Update learning rate
            self.scheduler.step()
            
            # Print average loss
            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Update training history
            self.history["train_losses"].append(avg_loss)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader, n_steps)
                print(f"Validation - Loss: {val_metrics['loss']:.6f}, PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}")
            
            # Visualization
            if val_loader is not None:
                val_batch = next(iter(val_loader))
                sample_paths = self.visualize_samples(
                    epoch=epoch+1,
                    real_images=val_batch["pixel_values"],
                    captions=val_batch["caption"],
                    batch_size=8,
                    n_steps=n_steps
                )
                print(f"Saved {len(sample_paths)} visualization")

            # Save model at the end of each epoch
            if save_model_every_epoch:
                self.save_model(f"model-epoch-{epoch+1}.pt")

    def visualize_samples(self, epoch, real_images, captions, batch_size=8, n_steps=1000):
        vis_dir = self.results_folder / 'visualization' / f'epoch-{epoch}'
        vis_dir.mkdir(exist_ok=True, parents=True)
        
        sample_size = min(batch_size, len(real_images))
        real = real_images[:sample_size].to(self.device)
        caps = captions[:sample_size]
        
        ctx = self.encode_text(caps)
        gen = self.sample(batch_size=sample_size, context=ctx, n_steps=n_steps)

        print(f"Real: min={real.min().item():.4f}, max={real.max().item():.4f}, mean={real.mean().item():.4f}")
        print(f"Generated: min={gen.min().item():.4f}, max={gen.max().item():.4f}, mean={gen.mean().item():.4f}")
        
        saved = []
        for i in range(sample_size):
            fig, axes = plt.subplots(1, 2, figsize=(10,5))
                
            # Real
            img_r = real[i].cpu().numpy()
            if img_r.shape[0] == 1:
                axes[0].imshow(img_r[0], cmap='gray')
            else:
                axes[0].imshow(np.transpose(img_r, (1,2,0)))
            axes[0].set_title("Ground Truth")
            axes[0].axis('off')
            
            # Generated
            img_g = gen[i].cpu().numpy()
            if img_g.shape[0] == 1:
                axes[1].imshow(img_g[0], cmap='gray')
            else:
                axes[1].imshow(np.transpose(img_g, (1,2,0)))
            axes[1].set_title("Generated")
            axes[1].axis('off')
                
            # Caption
            cap = caps[i]
            cap = cap if len(cap) <= 100 else cap[:100] + "..."
            plt.suptitle(f"Caption: {cap}", fontsize=10)
            plt.tight_layout()
            
            path = vis_dir / f"sample-{i+1}.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved.append(str(path))
            
        return saved
            
    def save_model(self, filename, epoch=None):
        save_path = self.results_folder / 'checkpoints'

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        
        if epoch is not None:
            save_dict['epoch'] = epoch

        torch.save(save_dict, save_path / filename)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(str(checkpoint_path))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        next_epoch = 0
        if 'epoch' in checkpoint:
            next_epoch = checkpoint['epoch'] + 1
        elif len(self.history['train_losses']) > 0:
            next_epoch = len(self.history['train_losses'])
        
        return next_epoch

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_losses'], label='Training Loss')
        if self.history['val_losses']:
            plt.plot(self.history['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(str(self.results_folder / 'plots' /'loss_plot.png'))
        plt.close()

    def plot_metrics(self):
        if not self.history['val_psnr'] or not self.history['val_ssim']:
            return
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['val_psnr'])
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.title('Validation PSNR')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['val_ssim'])
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.title('Validation SSIM')
        
        plt.tight_layout()
        plt.savefig(str(self.results_folder / 'plots' /'metrics_plot.png'))
        plt.close()
