import modal
from modal import Image
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

cache_path = "/vol/cache"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_ID = "openai/clip-vit-base-patch32"

def get_model_info(model_ID, device):
	model = CLIPModel.from_pretrained(model_ID).to(device)
	processor = CLIPProcessor.from_pretrained(model_ID)
	tokenizer = CLIPTokenizer.from_pretrained(model_ID)
	return model, processor, tokenizer

def download_models():
    from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

    model, processor, tokenizer = get_model_info(model_ID, device)

    model.save_pretrained(cache_path, safe_serialization=True)
    processor.save_pretrained(cache_path, safe_serialization=True)
    tokenizer.save_pretrained(cache_path, safe_serialization=True)


app = modal.App(name="emb")

image = (
    Image.debian_slim(python_version="3.11.4")
    .pip_install(
        "pymongo",
        "requests",
        "modal",
        "Pillow",
        "transformers",
        "torch",
        "numpy",
        "torch",
        "pandas",
    )
    .run_function(
        download_models,
    )
)

app.image = image
