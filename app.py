import spaces
import gradio as gr
import os
import sys
from glob import glob
import time
from typing import Any, Union

import numpy as np
import torch

print(f'torch version:{torch.__version__}')

import trimesh
from huggingface_hub import snapshot_download
from PIL import Image
from accelerate.utils import set_seed

import subprocess
# import importlib, site, sys

# # Re-discover all .pth/.egg-link files
# for sitedir in site.getsitepackages():
#     site.addsitedir(sitedir)

# # Clear caches so importlib will pick up new modules
# importlib.invalidate_caches()

# def sh(cmd): subprocess.check_call(cmd, shell=True)

def install_cuda_toolkit():
    CUDA_TOOLKIT_URL = "https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run"
    CUDA_TOOLKIT_FILE = "/tmp/%s" % os.path.basename(CUDA_TOOLKIT_URL)
    subprocess.check_call(["wget", "-q", CUDA_TOOLKIT_URL, "-O", CUDA_TOOLKIT_FILE])
    subprocess.check_call(["chmod", "+x", CUDA_TOOLKIT_FILE])
    subprocess.check_call([CUDA_TOOLKIT_FILE, "--silent", "--toolkit"])

    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["PATH"] = "%s/bin:%s" % (os.environ["CUDA_HOME"], os.environ["PATH"])
    os.environ["LD_LIBRARY_PATH"] = "%s/lib:%s" % (
        os.environ["CUDA_HOME"],
        "" if "LD_LIBRARY_PATH" not in os.environ else os.environ["LD_LIBRARY_PATH"],
    )
    # Fix: arch_list[-1] += '+PTX'; IndexError: list index out of range
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
    print("==> finished installation")

print("installing cuda toolkit")
install_cuda_toolkit()
print("finished")

header_path = "/usr/local/cuda/include/cuda_runtime.h"
print(f"{header_path} exists:", os.path.exists(header_path))

# my_env = os.environ.copy()
subprocess.run(["pip", "install","diso"], check=True)


# # tell Python to re-scan site-packages now that the egg-link exists
# import importlib, site; site.addsitedir(site.getsitepackages()[0]); importlib.invalidate_caches()


from src.utils.data_utils import get_colored_mesh_composition, scene_to_parts, load_surfaces
from src.utils.render_utils import render_views_around_mesh, render_normal_views_around_mesh, make_grid_for_images_or_videos, export_renderings
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.utils.image_utils import prepare_image
from src.models.briarmbg import BriaRMBG

# Constants
MAX_NUM_PARTS = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

# Download and initialize models
partcrafter_weights_dir = "pretrained_weights/PartCrafter"
rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
snapshot_download(repo_id="wgsxm/PartCrafter", local_dir=partcrafter_weights_dir)
snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)

rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(DEVICE)
rmbg_net.eval()
pipe: PartCrafterPipeline = PartCrafterPipeline.from_pretrained(partcrafter_weights_dir).to(DEVICE, DTYPE)

@spaces.GPU()
@torch.no_grad()
def run_triposg(image: Image.Image,
                num_parts: int,
                seed: int,
                num_tokens: int,
                num_inference_steps: int,
                guidance_scale: float,
                max_num_expanded_coords: float,
                use_flash_decoder: bool,
                rmbg: bool):
    """
    Generate 3D part meshes from an input image.
    """
    if rmbg:
        img_pil = prepare_image(image, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
    else:
        img_pil = image

    set_seed(seed)
    start_time = time.time()
    outputs = pipe(
        image=[img_pil] * num_parts,
        attention_kwargs={"num_parts": num_parts},
        num_tokens=num_tokens,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_num_expanded_coords=max_num_expanded_coords,
        use_flash_decoder=use_flash_decoder,
    ).meshes
    duration = time.time() - start_time
    print(f"Generation time: {duration:.2f}s")

    # Ensure no None outputs
    for i, mesh in enumerate(outputs):
        if mesh is None:
            outputs[i] = trimesh.Trimesh(vertices=[[0,0,0]], faces=[[0,0,0]])

    # Merge and color
    merged = get_colored_mesh_composition(outputs)

    # Export meshes and return results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join("results", timestamp)
    os.makedirs(export_dir, exist_ok=True)
    for idx, mesh in enumerate(outputs):
        mesh.export(os.path.join(export_dir, f"part_{idx:02}.glb"))
    merged.export(os.path.join(export_dir, "object.glb"))

    return merged, export_dir

# Gradio Interface
def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# PartCrafter 3D Generation Demo")
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Input Image")
                num_parts = gr.Slider(1, MAX_NUM_PARTS, value=4, step=1, label="Number of Parts")
                seed = gr.Number(value=0, label="Random Seed", precision=0)
                num_tokens = gr.Slider(256, 2048, value=1024, step=64, label="Num Tokens")
                num_steps = gr.Slider(1, 100, value=50, step=1, label="Inference Steps")
                guidance = gr.Slider(1.0, 20.0, value=7.0, step=0.1, label="Guidance Scale")
                max_coords = gr.Text(value="1e9", label="Max Expanded Coords")
                flash_decoder = gr.Checkbox(value=False, label="Use Flash Decoder")
                remove_bg = gr.Checkbox(value=False, label="Remove Background (RMBG)")
                run_button = gr.Button("Generate 3D Parts")
            with gr.Column(scale=1):
                output_model = gr.Model3D(label="Merged 3D Object")
                output_dir = gr.Textbox(label="Export Directory")

        run_button.click(fn=run_triposg,
                         inputs=[input_image, num_parts, seed, num_tokens, num_steps,
                                 guidance, max_coords, flash_decoder, remove_bg],
                         outputs=[output_model, output_dir])
    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch()