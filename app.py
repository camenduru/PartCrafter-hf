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
import importlib, site, sys

# Re-discover all .pth/.egg-link files
for sitedir in site.getsitepackages():
    site.addsitedir(sitedir)

# Clear caches so importlib will pick up new modules
importlib.invalidate_caches()

def sh(cmd): subprocess.check_call(cmd, shell=True)

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
    # add for compiler header lookup
    os.environ["CPATH"] = f"{os.environ['CUDA_HOME']}/include" + (
        f":{os.environ['CPATH']}" if "CPATH" in os.environ else ""
    )
    # Fix: arch_list[-1] += '+PTX'; IndexError: list index out of range
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9;9.0"
    print("==> finished installation")

print("installing cuda toolkit")
install_cuda_toolkit()
print("finished")

header_path = "/usr/local/cuda/include/cuda_runtime.h"
print(f"{header_path} exists:", os.path.exists(header_path))

def sh(cmd_list, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.check_call(cmd_list, env=env)

# install with FORCE_CUDA=1
sh(["pip", "install", "diso"], {"FORCE_CUDA": "1"})
sh(["pip", "install", "torch-cluster", "-f", "https://data.pyg.org/whl/torch-2.7.0+126.html"])



# tell Python to re-scan site-packages now that the egg-link exists
import importlib, site; site.addsitedir(site.getsitepackages()[0]); importlib.invalidate_caches()


from src.utils.data_utils import get_colored_mesh_composition, scene_to_parts, load_surfaces
from src.utils.render_utils import render_views_around_mesh, render_normal_views_around_mesh, make_grid_for_images_or_videos, export_renderings
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.utils.image_utils import prepare_image
from src.models.briarmbg import BriaRMBG

# Constants
MAX_NUM_PARTS = 16
DEVICE = "cuda" 
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
def run_triposg(image_path: str,
                num_parts: int = 1,
                seed: int = 123,
                num_tokens: int = 1024,
                num_inference_steps: int = 50,
                guidance_scale: float = 7.0,
                max_num_expanded_coords: float = 1e9,
                use_flash_decoder: bool = False,
                rmbg: bool = True):
    """
    Generate 3D part meshes from an input image.
    """
    if rmbg:
        img_pil = prepare_image(image_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
    else:
        img_pil = Image.open(image_path)

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

    glb_path = os.path.join(export_dir, "object.glb")
    merged.export(glb_path)
    print(glb_path)

    # 1) Check for the file’s existence
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"No merged .glb found at {glb_path}")

    # 2) List every file in the folder
    all_files = os.listdir(export_dir)
    
    print(f"Files in {export_dir}: {all_files}")
    
    return glb_path, export_dir

# Gradio Interface
def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown(
        """ # PartCrafter – Structured 3D Mesh Generation via Compositional Latent Diffusion Transformers

        • Source: [Github](https://github.com/wgsxm/PartCrafter)  
        • HF Space by : [@alexandernasa](https://twitter.com/alexandernasa/)  """
        )
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="filepath", label="Input Image")
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
                examples = gr.Examples(
                    examples=[
                        [
                            "examples/np10_cc486e491a2c499f9fd2aad2b02c6ccb.png", 
                            1,
                            123,
                            1024,
                            50,
                            7.0,
                            1e9,
                            False,
                            True
                        ], 
                        
                    ],
                    inputs=[input_image, num_parts, seed, num_tokens, num_steps,
                                 guidance, max_coords, flash_decoder, remove_bg],
                    outputs=[output_model, output_dir],
                    fn=run_triposg,
                    cache_examples=True,
                )

        run_button.click(fn=run_triposg,
                         inputs=[input_image, num_parts, seed, num_tokens, num_steps,
                                 guidance, max_coords, flash_decoder, remove_bg],
                         outputs=[output_model, output_dir])
    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch()