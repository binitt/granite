# granite

## Setup

```bash
uv venv --python "C:\Python\Python313\python.exe"
uv run python -V
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt
```

## Login to Hugging Face

```bash
hf auth login
```

## Manual download

```bash
hf download ibm-granite/granite-speech-4.1-2b-plus
```

## Run

```bash
uv run python transcribe_plus.py
```
