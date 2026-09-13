# Conditional DDPM for Text-to-Medical-Image Synthesis

Text-conditional DDPM that generates 256x256 chest X-ray images from radiology report text, using a frozen BiomedCLIP text encoder with a trainable projection and a cross-attention UNet. Supports DDIM sampling, classifier-free guidance, and EMA weights.

## 0. Prerequisites

* NVIDIA GPU with CUDA support (recommended: at least 16GB VRAM)
* CUDA Toolkit (12.1 recommended)
* Python 3.12

## 1. Environment Setup

1. Install [uv](https://docs.astral.sh/uv/) (no root required)

    ```
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ```

2. Create a virtual environment and install dependencies

    ```
    uv venv .venv --python 3.12
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```

    `open_clip_torch` is installed from PyPI. The text encoder enables token-level outputs at runtime (`model.text.output_tokens = True`), so no modification of the open_clip source is needed.

3. In every new shell session, activate the environment before running anything:

    ```
    source .venv/bin/activate
    ```

## 2. Dataset Download
  * Access the Indiana University Chest X-ray Collection from [Open-i](https://openi.nlm.nih.gov/faq).
  * For this project, you must use both the `PNG images` and `Reports` of the dataset.
  * Organize the data with following directory structure:

    ```
    data/
      /IU-XRay
        /NLMCXR_png
        /ecgen-radiology
    ```

## 3. Training

1. Navigate to `config.py` and modify the hyperparameters in the `Config` class based on your settings (shared by `train.py` and `inference.py`).
2. Start the training process with the following command

   ```python3 train.py```

Notes:

* **Auto-resume**: rerunning `train.py` automatically resumes from the latest checkpoint in `results/checkpoints/` (`model-latest.pt` or the newest `model-epoch-N.pt`).
* **Time-limited GPU sessions**: training stops gracefully after `Config.max_train_hours` (default 7.5h, fitting an 8h GPU allocation) and checkpoints `model-latest.pt` every `Config.checkpoint_every_min` (default 30 min). Just rerun `train.py` in the next session to continue. Only the newest `Config.keep_checkpoints` per-epoch checkpoints are kept on disk.
* **Monitoring**: training/validation curves and sample images are logged to TensorBoard (`tensorboard --logdir results/tensorboard`); per-epoch GT-vs-generated comparisons are saved under `results/visualization/`.

## 4. Inference

Generate medical images from text descriptions:

```
python inference.py --checkpoint </path/to/checkpoint.pt> \
    --caption "<caption>" \
    --output <img.png> \
    --n_steps <n_steps> \
    --guidance_scale <scale> \
    --seed <seed> \
    --batch_size <batch_size>
```

* `--n_steps` below the training timesteps (1000) uses DDIM sampling (default 250); 1000 runs full DDPM.
* `--guidance_scale` controls classifier-free guidance strength (default 3.0; 1.0 disables guidance).
* Sampling uses EMA weights when present in the checkpoint.

## 5. Evaluation (FID / KID)

Per-epoch PSNR/SSIM only tracks a rough trend: generated samples are not pixel-aligned
with ground truth, and SSIM saturates within a few epochs. Distribution metrics are the
real measure of sample quality.

```
python evaluate.py --checkpoint </path/to/checkpoint.pt> \
    --n-samples <n> \
    --batch-size <batch_size> \
    --n-steps <n_steps> \
    --guidance-scale <scale> \
    --reference-split {train,val,all} \
    --caption-split {train,val,all} \
    --features 2048 768 \
    --save-samples results/eval-samples
```

Samples `--n-samples` images conditioned on captions from `--caption-split` (default
`val`, i.e. unseen reports) and compares them against the real images in
`--reference-split` (default `all`, for the largest possible reference). Prints FID and
KID (mean ± std) and writes them with the full run configuration to
`results/eval_fid_kid.json`.

* `--split-ratio` must match the value used at training time, or `val` is not actually held out.
* `--features` scores at several Inception feature dims in one sampling pass. **2048 is the
  standard FID reported in the literature; smaller dims are on a completely different scale
  and are not comparable to it** (on the same images, 2048 gave FID 81.8 where 768 gave 0.49).
  Smaller dims are only useful as a relative signal when the reference set is tiny.
* `--save-samples` caches the generated images so metrics can be recomputed at other feature
  dims without repeating the sampling pass, which dominates the runtime.
* `--n-samples` should be at least the largest feature dim, or the covariance FID estimates is
  rank-deficient. The default job uses 2000 against a 3,851-image reference set; report that as
  FID-2k rather than comparing it to FID-50k numbers.
* KID's subset size is clamped to the smaller of the two distributions; it is more
  reliable than FID at this dataset's scale (~3.8k frontal images).
* Sampling dominates the runtime: `n_samples × n_steps` UNet forward passes.
