"""
Additional inference script for stable diffusion
Edited by IAmxIAo
"""

from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import base64
import numpy as np

def process_data(data: dict) -> dict:
    g_cuda = None
    g_cuda = torch.Generator(device='cuda')
    
    return {
        "prompt": data.pop("prompt", data),
        "negative_prompt": data.pop("negative_prompt", ""),
        "num_images_per_prompt": min(data.pop("num_samples", 2),5),
        "guidance_scale": data.pop("guidance_scale", 7.5),
        "num_inference_steps": min(data.pop("num_inference_steps", 50), 50),
        "height": 512,
        "width": 512,
        "generator":g_cuda.manual_seed(data.pop("seed",15213))
    }


def model_fn(model_dir: str):
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    t2i_pipe = StableDiffusionPipeline.from_pretrained(
        model_dir,
        scheduler=scheduler, 
        safety_checker=None, 
        torch_dtype=torch.float16
    )
    if torch.cuda.is_available():
        t2i_pipe = t2i_pipe.to("cuda")

    t2i_pipe.enable_attention_slicing()
    return t2i_pipe


def predict_fn(data: dict, hgf_pipe) -> dict:
    with torch.autocast("cuda"):
        images = hgf_pipe(**process_data(data))["images"]

    # return dictionary, which will be json serializable
    return {
        "images": [
            base64.b64encode(np.array(image).astype(np.uint8)).decode("utf-8")
            for image in images
        ]
    }
