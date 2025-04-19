import random
import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from models.clip_encoder import CLIPEncoder
from models.conditional_unet import Unet
from diffusion_trainer import DiffusionTrainer


class Config:
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Text encoder
    model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    max_length = 128
    use_projection = True
    diffusion_dim = 512

    # UNet
    channels = 1
    self_condition=False
    use_linear_attention=True
    use_cross_attention=True
    
    # Diffusion trainer
    results_folder = "./results"
    batch_size = 1
    timesteps = 1000
    beta_schedule = "cosine"
    image_size = 256
    n_steps = 1000
    scheduler_type = "cosine"
    scheduler_params = {"T_max": 50, "eta_min": 5e-6} 


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if Config.device == "cuda":
        torch.cuda.manual_seed_all(seed)


def save_with_caption(image_tensor, caption, output_path):
    # [-1, 1] -> [0, 255]
    gen_image = ((image_tensor + 1) * 0.5 * 255).clamp(0, 255)
    gen_image = gen_image.squeeze().cpu().numpy().astype(np.uint8)
    
    plt.figure(figsize=(8, 10))
    
    # Generared
    if len(gen_image.shape) == 3 and gen_image.shape[0] in [1, 3]:
        if gen_image.shape[0] == 1:
            plt.imshow(gen_image[0], cmap='gray')
        else:
            plt.imshow(np.transpose(gen_image, (1, 2, 0)))
    else:
        plt.imshow(gen_image, cmap='gray')
    
    plt.axis('off')
    
    # Caption
    caption = caption if len(caption) <= 100 else caption[:100] + "..."
    plt.title(f"Caption: {caption}", fontsize=12, wrap=True)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    
    print(f"Generated image saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Text-to-Medical-Image Generation using Diffusion Models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--caption", type=str, required=True, help="Medical description for generating the image.")
    parser.add_argument("--output", type=str, default="generated_image.png", help="Path to save the output image.")
    parser.add_argument("--n_steps", type=int, default=Config.n_steps, help="Number of sampling steps.")
    parser.add_argument("--seed", type=int, default=Config.seed, help="Random seed for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generating multiple images from same prompt.")
    
    args = parser.parse_args()
    
    Config.seed = args.seed
    Config.n_steps = args.n_steps
    Config.batch_size = args.batch_size
    
    set_seed(Config.seed)
    
    results_folder = Path(Config.results_folder)
    results_folder.mkdir(exist_ok=True)
    
    output_path = Path(args.output)
    if output_path.suffix.lower() not in ['.png']:
        output_path = output_path.with_suffix('.png')
    
    print(f"Initializing text encoder with {Config.model_name}...")
    text_encoder = CLIPEncoder(
        model_name=Config.model_name, 
        max_length=Config.max_length, 
        diffusion_dim=Config.diffusion_dim,
        use_projection=Config.use_projection
    )
    
    context_dim = Config.diffusion_dim if Config.use_projection else text_encoder.embedding_dim
    
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
    
    trainer = DiffusionTrainer(
        model=unet,
        dataloader=None,
        text_encoder=text_encoder,
        timesteps=Config.timesteps,
        beta_schedule=Config.beta_schedule,
        image_size=Config.image_size,
        channels=Config.channels,
        batch_size=Config.batch_size,
        lr=0,
        device=Config.device,
        results_folder=Config.results_folder,
        scheduler_type=Config.scheduler_type,
        scheduler_params=Config.scheduler_params
    )
    
    print(f"Loading model from checkpoint: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=Config.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    trainer.model.eval()
    
    print(f"Generating image(s) with caption: \"{args.caption}\"")
    print(f"Using {args.n_steps} sampling steps")
    
    with torch.no_grad():
        generated_images = trainer.sample_from_text(
            text=args.caption,
            batch_size=Config.batch_size,
            n_steps=args.n_steps
        )
    
    if Config.batch_size == 1:
        save_with_caption(generated_images[0], args.caption, output_path)
    else:
        for i in range(Config.batch_size):
            img_path = output_path.with_stem(f"{output_path.stem}_{i+1}")
            save_with_caption(generated_images[i], args.caption, img_path)
        
        print(f"Individual images saved with prefix {output_path.stem}")


if __name__ == "__main__":
    main()