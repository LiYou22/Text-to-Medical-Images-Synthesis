# Conditonal DDPM for Text-to-Medical-Images-Synthesis

## 0. Environment Setup

1. Create a virtual environment

    ```conda create -n <env_name> python==3.12```

2. Install dependencies

    ```pip install -r requirements.txt```

3. For this project, you need to setup the `open_clip` dependency based on following guideline:
  * Go to the github page of [open_clip](https://github.com/mlfoundations/open_clip).
  * Clone the repository and put it in the root path of the project directory
  * In `/open_clip/src/open_clip/model.py`, change the `output_tokens` attribute of class `CLIPTextCfg` to `True`. This way, the text encoder will return embeddings of tokens rather than a globally-pooled embedding.

## 1. Dataset Download
  * Access the Indiana University Chest X-ray Collection from [Open-i](https://openi.nlm.nih.gov/faq).
  * For this project, you must use both the `PNG images` and `Reports` of the dataset.
  * Organize the data with following directory structure:
    
    ```
    data/
      /IU-XRay
        /NLMCXR_png      # Contains all PNG format chest X-ray images
        /Reports         # Contains corresponding medical reports
    ```

## 2. Training

1. Navigate to the `train.py` and modify the hyperparameters in the `Config` class based on your settings.
2. Start the training process with the following command
   ```python3 train.py```

## 3.  Inference with Pre-trained Models

1. Download pre-trained weights from [Google Drive](https://drive.google.com/file/d/1CZSjKLUYmv8malavn2qfcxEKy_ZH_DgG/view?usp=drive_link).
2. Save the checkpoint file.
3. Generate medical images from text descriptions using the following command:  
   
   ```python inference.py --checkpoint </path/to/your/checkpoint.pt> --caption <caption> --output <img.png> --n_steps <n_steps> --seed <seed>```
