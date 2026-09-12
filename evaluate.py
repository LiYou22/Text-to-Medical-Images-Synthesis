import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm

from config import Config, build_model_config
from dataset import IUXrayDataset, custom_collate
from diffusion_trainer import DiffusionTrainer
from models.clip_encoder import CLIPEncoder
from models.conditional_unet import Unet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if Config.device == "cuda":
        torch.cuda.manual_seed_all(seed)


def to_inception_input(x):
    """[-1, 1] single-channel -> [0, 1] three-channel, which is what
    torchmetrics expects with normalize=True."""
    x = (x.clamp(-1, 1) + 1) / 2
    return x.repeat(1, 3, 1, 1)


def load_trainer(checkpoint_path, device):
    text_encoder = CLIPEncoder(
        model_name=Config.model_name,
        max_length=Config.max_length,
        diffusion_dim=Config.diffusion_dim,
        use_projection=Config.use_projection,
    )
    context_dim = Config.diffusion_dim if Config.use_projection else text_encoder.embedding_dim

    unet = Unet(
        dim=Config.unet_dim,
        dim_mults=Config.dim_mults,
        channels=Config.channels,
        context_dim=context_dim,
        self_condition=Config.self_condition,
        use_linear_attn=Config.use_linear_attention,
        use_cross_attention=Config.use_cross_attention,
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
        device=device,
        results_folder=Config.results_folder,
        scheduler_type=Config.scheduler_type,
        scheduler_params=Config.scheduler_params,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    ckpt_cfg = checkpoint.get("model_config")
    if ckpt_cfg is not None:
        current_cfg = build_model_config(context_dim)
        mismatch = {k: (v, current_cfg.get(k)) for k, v in ckpt_cfg.items() if current_cfg.get(k) != v}
        if mismatch:
            raise SystemExit(f"Checkpoint architecture does not match config (checkpoint, current): {mismatch}")

    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.ema_model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    if "text_projection_state_dict" in checkpoint and Config.use_projection:
        text_encoder.projection.load_state_dict(checkpoint["text_projection_state_dict"])
    else:
        print("Warning: checkpoint has no text projection weights; conditioning will be meaningless.")

    trainer.model.eval()
    trainer.ema_model.eval()
    return trainer


def build_dataset(split, split_ratio):
    if split == "all":
        train = IUXrayDataset(Config.data_dir, Config.image_size, split_ratio, True,
                              Config.max_samples, Config.frontal_only)
        val = IUXrayDataset(Config.data_dir, Config.image_size, split_ratio, False,
                            Config.max_samples, Config.frontal_only)
        return torch.utils.data.ConcatDataset([train, val])
    return IUXrayDataset(Config.data_dir, Config.image_size, split_ratio, split == "train",
                         Config.max_samples, Config.frontal_only)


def collect_captions(dataset):
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        return [s["caption"] for d in dataset.datasets for s in d.samples]
    return [s["caption"] for s in dataset.samples]


def main():
    parser = argparse.ArgumentParser(description="FID/KID evaluation for the text-conditional diffusion model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of images to generate for the fake distribution.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=Config.n_steps)
    parser.add_argument("--guidance-scale", type=float, default=Config.guidance_scale)
    parser.add_argument("--reference-split", choices=["train", "val", "all"], default="all",
                        help="Real images the generated distribution is compared against.")
    parser.add_argument("--caption-split", choices=["train", "val", "all"], default="val",
                        help="Captions the samples are conditioned on; val keeps them unseen.")
    parser.add_argument("--split-ratio", type=float, default=Config.split_ratio,
                        help="Must match the value used at training time, or 'val' is not held out.")
    parser.add_argument("--feature", type=int, default=2048, choices=[64, 192, 768, 2048],
                        help="Inception feature dim. 768 is less biased when reals are scarce.")
    parser.add_argument("--kid-subset-size", type=int, default=100)
    parser.add_argument("--no-ema", action="store_true", help="Sample from the raw weights instead of EMA.")
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--output", type=str, default="results/eval_fid_kid.json")

    args = parser.parse_args()
    set_seed(args.seed)
    device = Config.device

    trainer = load_trainer(args.checkpoint, device)

    reference = build_dataset(args.reference_split, args.split_ratio)
    caption_pool = collect_captions(build_dataset(args.caption_split, args.split_ratio))
    if not caption_pool:
        raise SystemExit(f"No captions in split '{args.caption_split}'.")

    n_real = len(reference)
    n_unique_captions = len(set(caption_pool))
    print(f"Reference reals: {n_real} | caption pool: {len(caption_pool)} ({n_unique_captions} unique)")
    if n_real < args.feature:
        print(f"Warning: {n_real} reals with feature={args.feature} makes FID severely biased; "
              f"use --feature 768 or a larger --reference-split.")
    if n_unique_captions < 100:
        print(f"Warning: only {n_unique_captions} unique captions; FID will partly measure "
              f"noise diversity rather than conditional coverage.")

    # KID draws subsets from both distributions, so it cannot exceed either side
    kid_subset = min(args.kid_subset_size, args.n_samples, n_real)

    fid = FrechetInceptionDistance(feature=args.feature, normalize=True).to(device)
    kid = KernelInceptionDistance(feature=args.feature, subset_size=kid_subset,
                                  normalize=True).to(device)

    real_loader = DataLoader(reference, batch_size=args.batch_size, shuffle=False,
                             collate_fn=custom_collate, num_workers=2)
    for batch in tqdm(real_loader, desc="Real features"):
        x = to_inception_input(batch["pixel_values"].to(device))
        fid.update(x, real=True)
        kid.update(x, real=True)

    # Cycle the caption pool with a fixed shuffle so every caption is used a
    # comparable number of times when n_samples exceeds the pool.
    rng = random.Random(args.seed)
    captions = caption_pool[:]
    rng.shuffle(captions)
    captions = [captions[i % len(captions)] for i in range(args.n_samples)]

    generated = 0
    with tqdm(total=args.n_samples, desc="Sampling") as bar:
        while generated < args.n_samples:
            chunk = captions[generated:generated + args.batch_size]
            samples = trainer.sample(
                batch_size=len(chunk),
                captions=chunk,
                n_steps=args.n_steps,
                guidance_scale=args.guidance_scale,
                use_ema=not args.no_ema,
                show_progress=False,
            )
            x = to_inception_input(samples.to(device))
            fid.update(x, real=False)
            kid.update(x, real=False)
            generated += len(chunk)
            bar.update(len(chunk))

    fid_value = fid.compute().item()
    kid_mean, kid_std = kid.compute()

    results = {
        "checkpoint": args.checkpoint,
        "fid": fid_value,
        "kid_mean": kid_mean.item(),
        "kid_std": kid_std.item(),
        "n_generated": args.n_samples,
        "n_real": n_real,
        "n_unique_captions": n_unique_captions,
        "reference_split": args.reference_split,
        "caption_split": args.caption_split,
        "split_ratio": args.split_ratio,
        "n_steps": args.n_steps,
        "guidance_scale": args.guidance_scale,
        "feature": args.feature,
        "kid_subset_size": kid_subset,
        "ema": not args.no_ema,
        "seed": args.seed,
    }

    print(f"\nFID  {fid_value:.2f}")
    print(f"KID  {kid_mean.item():.4f} +/- {kid_std.item():.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
