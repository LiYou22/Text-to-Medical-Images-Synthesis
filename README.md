# Conditional DDPM for Text-to-Medical-Image Synthesis

Text-conditional DDPM that generates 256x256 chest X-ray images from radiology report text, using a frozen BiomedCLIP text encoder with a trainable projection and a cross-attention UNet. Supports DDIM sampling, classifier-free guidance, and EMA weights.

## 0. Prerequisites

* NVIDIA GPU with CUDA support (recommended: at least 16GB VRAM)
* CUDA Toolkit (12.1 recommended)
* Python 3.12

## 1. Environment Setup

1. Create a virtual environment

    ```
    conda create -n <env_name> python==3.12
    conda activate <env_name>
    ```

2. Install dependencies

    ```pip install -r requirements.txt```

    `open_clip_torch` is installed from PyPI. The text encoder enables token-level outputs at runtime (`model.text.output_tokens = True`), so no modification of the open_clip source is needed.

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

> **Note on old pretrained weights**: checkpoints trained before the conditioning/sampling overhaul (e.g., the previously linked Google Drive weights) are incompatible with the current code — they lack the trained text projection and EMA weights, so generations will not be meaningful. Retrain to produce usable checkpoints.
