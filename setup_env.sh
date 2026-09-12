export PROJ=/projects/cvpr/YouLi/Text-to-Medical-Images-Synthesis
cd $PROJ

# 1) install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2) set up virtual environments
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt

# 3) install torch-cu128
uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# 4) Download BiomedCLIP weights
python - <<'PY'
from open_clip import create_model_from_pretrained, get_tokenizer
m = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
create_model_from_pretrained(m); get_tokenizer(m)
print("BiomedCLIP cached.")
PY

# 5) Download InceptionV3 weights
python - <<'PY'
from torchmetrics.image.fid import FrechetInceptionDistance
FrechetInceptionDistance(feature=768, normalize=True)
print("InceptionV3 cached.")
PY
