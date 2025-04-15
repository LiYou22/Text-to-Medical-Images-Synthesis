import random
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from diffusion_trainer import DiffusionTrainer
from models.clip_encoder import CLIPEncoder
from models.conditional_unet import Unet
from dataset import IUXrayDataset, custom_collate


class Config:
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "./data/IU-XRay"
    
    # Encoder
    model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    max_length = 77
    use_projection = True
    diffusion_dim = 512
    
    # Unet
    channels = 1
    self_condition=True
    use_linear_attention=True
    use_cross_attention=True
    
    # Diffusion Trainer
    image_size = 256
    batch_size = 8
    epochs = 50
    lr = 5e-5
    timesteps = 1000
    beta_schedule = 'cosine'
    results_folder = "./results"
    loss_type = "huber"
    scheduler_type = "cosine"
    scheduler_params = {"T_max": 100, "eta_min": 1e-6}
    
    split_ratio = 0.9
    max_samples = None

    grad_clip_value=1.0
    save_model_every_epoch=True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if Config.device == "cuda":
        torch.cuda.manual_seed_all(seed)


def find_latest_checkpoint(results_folder):
    results_path = Path(results_folder) / "checkpoints"
    if not results_path.exists():
        return None
    
    model_files = []
    for pattern in ["model-*.pt", "model-epoch-*.pt"]:
        model_files.extend(list(results_path.glob(pattern)))
    
    if not model_files:
        return None
    
    latest_checkpoint = max(model_files, key=lambda x: x.stat().st_mtime)
    return latest_checkpoint


def main():
    set_seed(Config.seed)
    
    results_folder = Path(Config.results_folder)
    results_folder.mkdir(exist_ok=True)
    
    print(f"Loading datasets from {Config.data_dir}...")
    train_dataset = IUXrayDataset(
        data_dir=Config.data_dir, 
        image_size=Config.image_size, 
        split_ratio=Config.split_ratio, 
        is_train=True,
        max_samples=Config.max_samples
    )
    val_dataset = IUXrayDataset(
        data_dir=Config.data_dir, 
        image_size=Config.image_size, 
        split_ratio=Config.split_ratio, 
        is_train=False,
        max_samples=Config.max_samples
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        collate_fn=custom_collate,
        num_workers=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False, 
        collate_fn=custom_collate,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Initializing text encoder with {Config.model_name}...")
    text_encoder = CLIPEncoder(
        model_name=Config.model_name, 
        max_length=Config.max_length, 
        embedding_dim=None,
        use_projection=Config.use_projection
    )
    
    context_dim = Config.diffusion_dim if Config.use_projection else text_encoder.embedding_dim
    print(f"Using context dimension: {context_dim}")
    
    print("Initializing conditional UNet model...")
    unet = Unet(
        dim=64,
        init_dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=Config.channels,
        context_dim=context_dim,
        self_condition=Config.self_condition,
        use_linear_attn=Config.use_linear_attention,
        use_cross_attention=Config.use_cross_attention
    )
    
    model_parameters = sum(p.numel() for p in unet.parameters())
    print(f"UNet model parameters: {model_parameters:,}")
    
    print("Initializing diffusion trainer...")
    trainer = DiffusionTrainer(
        model=unet,
        dataset=train_dataloader,
        text_encoder=text_encoder,
        timesteps=Config.timesteps,
        beta_schedule=Config.beta_schedule,
        image_size=Config.image_size,
        channels=Config.channels,
        batch_size=Config.batch_size,
        lr=Config.lr,
        device=Config.device,
        results_folder=Config.results_folder,
        loss_type=Config.loss_type,
        scheduler_type=Config.scheduler_type,
        scheduler_params=Config.scheduler_params
    )

    start_epoch = 0
    checkpoint_path = find_latest_checkpoint(Config.results_folder)
    if checkpoint_path:
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        start_epoch = trainer.load_model(checkpoint_path)
        print(f"Resumed from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting training from scratch.")
    
    print(f"Starting training for {Config.epochs} epochs...")
    trainer.train(
        epochs=Config.epochs,
        start_epoch=start_epoch,
        val_loader=val_dataloader,
        grad_clip_value=Config.grad_clip_value,
        save_model_every_epoch=Config.save_model_every_epoch
    )
    
    trainer.save_model("model-final.pt")
    
    trainer.plot_losses()
    trainer.plot_metrics()
    
    print("Training completed!")


if __name__ == "__main__":
    main()