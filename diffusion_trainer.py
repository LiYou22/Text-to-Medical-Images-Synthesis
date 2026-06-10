############################################################################################
# Reference: https://huggingface.co/blog/annotated-diffusion
############################################################################################


import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
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
        scheduler_params = None,
        cond_drop_prob=0.1,
        ema_decay=0.9995,
        use_amp=False,
        model_config=None
    ):
        self.model = model
        self.dataloader = dataloader
        self.text_encoder = text_encoder
        self.timesteps = timesteps
        self.image_size = image_size
        self.channels = channels
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.cond_drop_prob = cond_drop_prob
        self.ema_decay = ema_decay
        self.model_config = model_config

        # Setup device
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Mixed precision: bf16 when supported (no scaler needed), else fp16 + GradScaler
        self.use_amp = use_amp and str(self.device).startswith("cuda")
        self.amp_dtype = torch.bfloat16 if (self.use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)

        if self.text_encoder is not None:
            self.text_encoder.to(self.device)

        # EMA copy of the UNet, used for sampling
        self.ema_model = deepcopy(self.model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        # Create folder to save the results
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        (self.results_folder / 'visualization').mkdir(exist_ok=True)
        (self.results_folder / 'plots').mkdir(exist_ok=True)
        (self.results_folder / 'checkpoints').mkdir(exist_ok=True)

        # Set up optimizer (include the text projection so it actually gets trained)
        params = list(model.parameters())
        if self.text_encoder is not None and getattr(self.text_encoder, "use_projection", False):
            params += list(self.text_encoder.projection.parameters())
        self.trainable_params = params
        self.optimizer = AdamW(params, lr=lr, weight_decay=0.01)

        # TensorBoard writer is created lazily in train() so inference-only
        # usage doesn't create log directories
        self.writer = None
        self.global_step = 0

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

        # Gradients flow through the trainable projection; the CLIP backbone
        # is frozen inside the encoder itself.
        _, text_embeddings = self.text_encoder.encode_batch(captions)
        return text_embeddings.to(self.device)

    @torch.no_grad()
    def _update_ema(self):
        for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.lerp_(p, 1.0 - self.ema_decay)
        for ema_b, b in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_b.copy_(b)
    
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
    def _predict_eps(self, x, t, context=None, model=None, guidance_scale=1.0, null_context=None):
        model = self.model if model is None else model
        if getattr(model, "use_cross_attention", False) and context is not None:
            if guidance_scale != 1.0 and null_context is not None:
                # Classifier-free guidance
                eps_cond = model(x, t, context=context)
                eps_uncond = model(x, t, context=null_context)
                return eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            return model(x, t, context=context)
        return model(x, t)

    @torch.no_grad()
    def p_sample(self, x, t, t_index, context=None, model=None, guidance_scale=1.0, null_context=None):
        """Single step of DDPM sampling"""
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        eps = self._predict_eps(x, t, context, model=model, guidance_scale=guidance_scale, null_context=null_context)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * eps / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        noise = torch.randn_like(x)
        variance = extract(self.posterior_variance, t, x.shape)

        return model_mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, context=None, model=None, guidance_scale=1.0, null_context=None, show_progress=True):
        """Full DDPM sampling loop over all timesteps. Returns all images."""
        device = self.device
        b = shape[0]

        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []

        iterator = range(self.timesteps - 1, -1, -1)
        if show_progress:
            iterator = tqdm(iterator, desc='DDPM sampling', total=self.timesteps)

        for t_index in iterator:
            t = torch.full((b,), t_index, device=device, dtype=torch.long)
            img = self.p_sample(img, t, t_index, context, model=model,
                                guidance_scale=guidance_scale, null_context=null_context)
            imgs.append(img.cpu().detach())

        return imgs

    @torch.no_grad()
    def ddim_sample_loop(self, shape, context=None, n_steps=50, eta=0.0, model=None,
                         guidance_scale=1.0, null_context=None, show_progress=True):
        """DDIM sampling over a strided subset of timesteps. Returns all images."""
        device = self.device
        b = shape[0]

        times = torch.linspace(-1, self.timesteps - 1, steps=n_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)
        imgs = []

        iterator = tqdm(time_pairs, desc='DDIM sampling') if show_progress else time_pairs

        for time, time_next in iterator:
            t = torch.full((b,), time, device=device, dtype=torch.long)
            eps = self._predict_eps(img, t, context, model=model,
                                    guidance_scale=guidance_scale, null_context=null_context)

            alpha = self.alphas_cumprod[time]
            x0 = (img - (1 - alpha).sqrt() * eps) / alpha.sqrt()
            x0.clamp_(-1.0, 1.0)

            if time_next < 0:
                img = x0
            else:
                alpha_next = self.alphas_cumprod[time_next]
                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                noise = torch.randn_like(img)
                img = x0 * alpha_next.sqrt() + c * eps + sigma * noise

            imgs.append(img.cpu().detach())

        return imgs

    @torch.no_grad()
    def sample(self, batch_size=32, captions=None, context=None, n_steps=None,
               guidance_scale=1.0, use_ema=True, show_progress=True):
        """Method for generating samples"""
        image_shape = (batch_size, self.channels, self.image_size, self.image_size)

        if captions is not None and context is None and self.text_encoder is not None:
            context = self.encode_text(captions)

        model = self.ema_model if use_ema else self.model
        model.eval()

        null_context = None
        if guidance_scale != 1.0 and context is not None and self.text_encoder is not None:
            null_context = self.encode_text([""] * batch_size)

        if n_steps is not None and n_steps < self.timesteps:
            samples = self.ddim_sample_loop(
                shape=image_shape,
                context=context,
                n_steps=n_steps,
                model=model,
                guidance_scale=guidance_scale,
                null_context=null_context,
                show_progress=show_progress
            )
        else:
            samples = self.p_sample_loop(
                shape=image_shape,
                context=context,
                model=model,
                guidance_scale=guidance_scale,
                null_context=null_context,
                show_progress=show_progress
            )

        return samples[-1]

    @torch.no_grad()
    def sample_from_text(self, text, batch_size=1, n_steps=None, guidance_scale=1.0,
                         use_ema=True, show_progress=True):
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
            guidance_scale=guidance_scale,
            use_ema=use_ema,
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
        total_loss = 0.0
        batch_count = 0

        # Denoising loss over the full val set with random t, same as training,
        # so train/val loss curves are directly comparable
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                x = batch["pixel_values"].to(self.device)
                captions = batch["caption"]
                batch_context = self.encode_text(captions)

                batch_size = x.shape[0]
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                loss = self.p_losses(x, t, context=batch_context)

                total_loss += loss.item()
                batch_count += 1

        avg_loss = total_loss / batch_count

        # Generative metrics only on the first val batch (fixed since
        # the loader does not shuffle); sampling the whole set is too slow
        gen_batch = next(iter(val_loader))
        real = gen_batch["pixel_values"].to(self.device)
        captions = gen_batch["caption"]
        with torch.no_grad():
            gen_context = self.encode_text(captions)
            generated = self.sample(
                batch_size=real.shape[0],
                context=gen_context,
                n_steps=n_steps,
                show_progress=False
            )
            metrics = self.compute_metrics(real, generated)

        # Update history
        self.history["val_losses"].append(avg_loss)
        self.history["val_psnr"].append(metrics["psnr"])
        self.history["val_ssim"].append(metrics["ssim"])

        return {
            "loss": avg_loss,
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "real": real,
            "captions": captions,
            "generated": generated
        }
    
    def train(self, epochs, start_epoch=0, val_loader=None, n_steps=200, save_model_every_epoch=True):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=str(self.results_folder / 'tensorboard'))

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

                # Caption dropout for classifier-free guidance
                if context is not None and self.cond_drop_prob > 0:
                    drop_mask = torch.rand(batch_size, device=self.device) < self.cond_drop_prob
                    if drop_mask.any():
                        null_context = self.encode_text([""] * batch_size)
                        context = torch.where(drop_mask[:, None, None], null_context, context)

                # Sample random timesteps
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

                # Compute loss (mixed precision when enabled)
                self.optimizer.zero_grad()
                with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    loss = self.p_losses(x, t, context=context)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.trainable_params, 0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self._update_ema()

                epoch_loss += loss.item()
                batch_count += 1
                self.global_step += 1
                if self.writer is not None and self.global_step % 10 == 0:
                    self.writer.add_scalar('train/step_loss', loss.item(), self.global_step)

            # Update learning rate
            self.scheduler.step()
            
            # Print average loss
            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")

            # Update training history
            self.history["train_losses"].append(avg_loss)
            self.writer.add_scalar('train/loss', avg_loss, epoch + 1)
            self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], epoch + 1)

            # Validation + visualization (reuse the samples generated during validation)
            if val_loader is not None:
                val_metrics = self.validate(val_loader, n_steps)
                print(f"Validation - Loss: {val_metrics['loss']:.6f}, PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}")

                self.writer.add_scalar('val/loss', val_metrics['loss'], epoch + 1)
                self.writer.add_scalar('val/psnr', val_metrics['psnr'], epoch + 1)
                self.writer.add_scalar('val/ssim', val_metrics['ssim'], epoch + 1)
                self.writer.add_images(
                    'val/generated',
                    ((val_metrics['generated'] + 1) / 2).clamp(0, 1),
                    epoch + 1
                )

                sample_paths = self.visualize_samples(
                    epoch=epoch+1,
                    real_images=val_metrics["real"],
                    captions=val_metrics["captions"],
                    generated=val_metrics["generated"],
                    batch_size=8,
                    n_steps=n_steps
                )
                print(f"Saved {len(sample_paths)} visualization")

            # Save model at the end of each epoch
            if save_model_every_epoch:
                self.save_model(f"model-epoch-{epoch+1}.pt")

        self.writer.flush()

    def visualize_samples(self, epoch, real_images, captions, generated=None, batch_size=8, n_steps=1000):
        vis_dir = self.results_folder / 'visualization' / f'epoch-{epoch}'
        vis_dir.mkdir(exist_ok=True, parents=True)

        sample_size = min(batch_size, len(real_images))
        real = real_images[:sample_size].to(self.device)
        caps = captions[:sample_size]

        if generated is not None:
            gen = generated[:sample_size]
        else:
            with torch.no_grad():
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
            'ema_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'history': self.history
        }

        if self.model_config is not None:
            save_dict['model_config'] = self.model_config

        if self.text_encoder is not None and getattr(self.text_encoder, "use_projection", False):
            save_dict['text_projection_state_dict'] = self.text_encoder.projection.state_dict()

        if epoch is not None:
            save_dict['epoch'] = epoch

        torch.save(save_dict, save_path / filename)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(str(checkpoint_path))

        ckpt_cfg = checkpoint.get('model_config')
        if ckpt_cfg is not None and self.model_config is not None:
            mismatch = {k: (v, self.model_config.get(k)) for k, v in ckpt_cfg.items()
                        if self.model_config.get(k) != v}
            if mismatch:
                raise ValueError(f"Checkpoint architecture does not match current config (checkpoint, current): {mismatch}")

        self.model.load_state_dict(checkpoint['model_state_dict'])
        # Older checkpoints have no EMA weights; fall back to the raw model
        self.ema_model.load_state_dict(checkpoint.get('ema_state_dict', checkpoint['model_state_dict']))
        if ('text_projection_state_dict' in checkpoint and self.text_encoder is not None
                and getattr(self.text_encoder, "use_projection", False)):
            self.text_encoder.projection.load_state_dict(checkpoint['text_projection_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

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
