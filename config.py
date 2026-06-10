import torch


class Config:
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "./data/IU-XRay"

    # Text encoder
    model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    max_length = 128
    use_projection = True
    diffusion_dim = 512

    # UNet
    unet_dim = 64
    dim_mults = (1, 2, 4, 8)
    channels = 1
    self_condition = False
    use_linear_attention = True
    use_cross_attention = True

    # Diffusion
    image_size = 256
    timesteps = 1000
    beta_schedule = "cosine"
    n_steps = 250
    guidance_scale = 3.0

    # Training
    batch_size = 8
    epochs = 30
    lr = 2e-5
    loss_type = "l2"
    scheduler_type = "cosine"
    scheduler_params = {"T_max": 30, "eta_min": 2e-6}
    cond_drop_prob = 0.1
    ema_decay = 0.9995
    use_amp = True
    results_folder = "./results"
    split_ratio = 0.99
    max_samples = None
    save_model_every_epoch = True


def build_model_config(context_dim):
    """Architecture hyperparameters stored in checkpoints so a checkpoint
    can be validated against the model it is loaded into."""
    return {
        "unet_dim": Config.unet_dim,
        "dim_mults": tuple(Config.dim_mults),
        "channels": Config.channels,
        "self_condition": Config.self_condition,
        "use_linear_attention": Config.use_linear_attention,
        "use_cross_attention": Config.use_cross_attention,
        "context_dim": context_dim,
        "image_size": Config.image_size,
        "timesteps": Config.timesteps,
        "beta_schedule": Config.beta_schedule,
        "model_name": Config.model_name,
        "max_length": Config.max_length,
        "use_projection": Config.use_projection,
        "diffusion_dim": Config.diffusion_dim,
    }
