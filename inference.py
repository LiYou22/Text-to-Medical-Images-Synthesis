import random
import torch
import numpy as np
import argparse
from pathlib import Path
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from models.clip_encoder import CLIPEncoder
from models.conditional_unet import Unet
from diffusion_trainer import DiffusionTrainer


class Config:
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Text encoder
    model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    max_length = 77
    use_projection = True
    diffusion_dim = 512

    # UNet
    channels = 1
    use_cross_attention=True
    use_self_attention=True
    use_linear_attention=True
    
    # Diffusion trainer
    results_folder = "./inference_results"
    batch_size = 1
    timesteps = 300
    beta_schedule = 'cosine'
    image_size = 256
    n_steps = 1000


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if Config.device == "cuda":
        torch.cuda.manual_seed_all(seed)


def save_with_caption(image_tensor, caption, output_path):
    image = (image_tensor + 1) * 0.5
    image = image.squeeze().cpu().numpy()
    
    plt.figure(figsize=(10, 12))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(caption, fontsize=12, wrap=True)
    plt.tight_layout()
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Text-to-Medical-Image Generation using Diffusion Models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--caption", type=str, required=True, help="Medical description for generating the image.")
    parser.add_argument("--output", type=str, default="generated_image.png", help="Path to save the output image.")
    parser.add_argument("--n_steps", type=int, default=Config.n_steps, help="Number of sampling steps.")
    parser.add_argument("--seed", type=int, default=Config.seed, help="Random seed for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generating multiple images from same prompt.")
    parser.add_argument("--create-animation", action="store_true", help="Create an animation of the diffusion process")
    
    args = parser.parse_args()
    
    Config.seed = args.seed
    Config.n_steps = args.n_steps
    Config.batch_size = args.batch_size
    
    set_seed(Config.seed)
    
    results_folder = Path(Config.results_folder)
    results_folder.mkdir(exist_ok=True)
    
    output_path = Path(args.output)
    if output_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
        output_path = output_path.with_suffix('.png')
    
    print(f"Initializing text encoder with {Config.model_name}...")
    text_encoder = CLIPEncoder(
        model_name=Config.model_name, 
        max_length=Config.max_length, 
        embedding_dim=None,
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
        self_condition=Config.use_self_attention,
        use_linear_attn=Config.use_linear_attention,
        cross_condition=Config.use_cross_attention
    )
    
    trainer = DiffusionTrainer(
        model=unet,
        dataset=None,
        text_encoder=text_encoder,
        timesteps=Config.timesteps,
        beta_schedule=Config.beta_schedule,
        image_size=Config.image_size,
        channels=Config.channels,
        batch_size=Config.batch_size,
        lr=0,
        device=Config.device,
        results_folder=Config.results_folder
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
        if output_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            save_image((generated_images + 1) / 2, output_path)
            print(f"Generated image saved to {output_path}")

            caption_path = output_path.with_stem(f"{output_path.stem}_with_caption")
            save_with_caption(generated_images[0], args.caption, caption_path)
            print(f"Generated image with caption saved to {caption_path}")
    else:
        grid_path = output_path.with_stem(f"{output_path.stem}_grid")
        save_image((generated_images + 1) / 2, grid_path, nrow=int(Config.batch_size**0.5))
        print(f"Grid of generated images saved to {grid_path}")
        
        for i in range(Config.batch_size):
            img_path = output_path.with_stem(f"{output_path.stem}_{i+1}")
            save_image((generated_images[i] + 1) / 2, img_path)
            
            caption_path = img_path.with_stem(f"{img_path.stem}_with_caption")
            save_with_caption(generated_images[i], args.caption, caption_path)
        
        print(f"Individual images saved with prefix {output_path.stem}")

    if args.create_animation:
        print("Creating animation of the diffusion process...")
        with torch.no_grad():
            context = trainer.encode_text([args.caption])
            samples = trainer.p_sample_loop(
                shape=(1, Config.channels, Config.image_size, Config.image_size),
                context=context,
                n_steps=Config.n_steps
            )
            
            animation_path = output_path.with_stem(f"{output_path.stem}_animation")
            trainer.create_animation(
                samples=samples,
                save_path=str(animation_path.with_suffix('.gif'))
            )
            print(f"Animation saved to {animation_path.with_suffix('.gif')}")

if __name__ == "__main__":
    main()