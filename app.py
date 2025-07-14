import spaces
import gradio as gr
import os
import sys
from glob import glob
import time
from typing import Any, Union

import numpy as np
import torch
import uuid
import shutil

print(f'torch version:{torch.__version__}')

import trimesh
import glob
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

os.environ["PARTCRAFTER_PROCESSED"] = f"{os.getcwd()}/proprocess_results"


def sh(cmd_list, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.check_call(cmd_list, env=env)

# install with FORCE_CUDA=1
sh(["pip", "install", "diso"], {"FORCE_CUDA": "1"})
# sh(["pip", "install", "torch-cluster", "-f", "https://data.pyg.org/whl/torch-2.7.0+126.html"])



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

def first_file_from_dir(directory, ext):
    files = glob.glob(os.path.join(directory, f"*.{ext}"))
    return sorted(files)[0] if files else None


def explode_mesh(mesh, explosion_scale=0.4):    

    if isinstance(mesh, trimesh.Scene):
        scene = mesh
    elif isinstance(mesh, trimesh.Trimesh):
        print("Warning: Single mesh provided, can't create exploded view")
        scene = trimesh.Scene(mesh)
        return scene
    else:
        print(f"Warning: Unexpected mesh type: {type(mesh)}")
        scene = mesh

    if len(scene.geometry) <= 1:
        print("Only one geometry found - nothing to explode")
        return scene
    
    print(f"[EXPLODE_MESH] Starting mesh explosion with scale {explosion_scale}")
    print(f"[EXPLODE_MESH] Processing {len(scene.geometry)} parts")
    
    exploded_scene = trimesh.Scene()
    
    part_centers = []
    geometry_names = []
    
    for geometry_name, geometry in scene.geometry.items():
        if hasattr(geometry, 'vertices'):
            transform = scene.graph[geometry_name][0]
            vertices_global = trimesh.transformations.transform_points(
                geometry.vertices, transform)
            center = np.mean(vertices_global, axis=0)
            part_centers.append(center)
            geometry_names.append(geometry_name)
            print(f"[EXPLODE_MESH] Part {geometry_name}: center = {center}")
    
    if not part_centers:
        print("No valid geometries with vertices found")
        return scene
    
    part_centers = np.array(part_centers)
    global_center = np.mean(part_centers, axis=0)
    
    print(f"[EXPLODE_MESH] Global center: {global_center}")
    
    for i, (geometry_name, geometry) in enumerate(scene.geometry.items()):
        if hasattr(geometry, 'vertices'):
            if i < len(part_centers):
                part_center = part_centers[i]
                direction = part_center - global_center
                
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-6:
                    direction = direction / direction_norm
                else:
                    direction = np.random.randn(3)
                    direction = direction / np.linalg.norm(direction)
                
                offset = direction * explosion_scale
            else:
                offset = np.zeros(3)
            
            original_transform = scene.graph[geometry_name][0].copy()
            
            new_transform = original_transform.copy()
            new_transform[:3, 3] = new_transform[:3, 3] + offset
            
            exploded_scene.add_geometry(
                geometry, 
                transform=new_transform, 
                geom_name=geometry_name
            )
            
            print(f"[EXPLODE_MESH] Part {geometry_name}: moved by {np.linalg.norm(offset):.4f}")
    
    print("[EXPLODE_MESH] Mesh explosion complete")
    return exploded_scene
    

def get_duration(
    image_path,
    num_parts,
    seed,
    num_tokens,
    num_inference_steps,
    guidance_scale,
    use_flash_decoder,
    rmbg,
    session_id,
    progress,
    ):

    duration_seconds = 60

    if num_parts > 5:
        duration_seconds = 75
    elif num_parts > 10:
        duration_seconds = 90
    return int(duration_seconds)
        
    
@spaces.GPU(duration=get_duration)
@torch.no_grad()
def run_triposg(image_path: str,
                num_parts: int = 1,
                seed: int = 0,
                num_tokens: int = 1024,
                num_inference_steps: int = 50,
                guidance_scale: float = 7.0,
                use_flash_decoder: bool = False,
                rmbg: bool = True,
                session_id = None,
                progress=gr.Progress(track_tqdm=True),):

    """
    Generate 3D part meshes from an input image.
    """

    max_num_expanded_coords = 1e9

    if session_id is None:
        session_id = uuid.uuid4().hex
        
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


    export_dir = os.path.join(os.environ["PARTCRAFTER_PROCESSED"], session_id)

    # If it already exists, delete it (and all its contents)
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    
    os.makedirs(export_dir, exist_ok=True)

    parts = []
    
    for idx, mesh in enumerate(outputs):
        part = os.path.join(export_dir, f"part_{idx:02}.glb")
        mesh.export(part)
        parts.append(part)
        
    zip_path = os.path.join(os.environ["PARTCRAFTER_PROCESSED"], f"{session_id}.zip")
    
    # shutil.make_archive wants the base name without extension:
    base_name = zip_path[:-4]  # strip off '.zip'
    shutil.make_archive(base_name, 'zip', export_dir)
    
    # Merge and color
    merged = get_colored_mesh_composition(outputs)


    glb_path = os.path.join(export_dir, "object.glb")
    merged.export(glb_path)

    mesh_file = first_file_from_dir(export_dir, "glb")
    
    return mesh_file, export_dir, zip_path

def cleanup(request: gr.Request):

    sid = request.session_hash
    if sid:
        d1 = os.path.join(os.environ["PARTCRAFTER_PROCESSED"], sid)
        shutil.rmtree(d1, ignore_errors=True)
        
def start_session(request: gr.Request):

    return request.session_hash
    
def build_demo():
    css = """
        #col-container {
            margin: 0 auto;
            max-width: 1024px;
        }
        """
    theme = gr.themes.Ocean()
    
    with gr.Blocks(css=css, theme=theme) as demo:
        session_state = gr.State()
        demo.load(start_session, outputs=[session_state])

        with gr.Column(elem_id="col-container"):
            gr.HTML(
                """
                <div style="text-align: center;">
                    <p style="font-size:16px; display: inline; margin: 0;">
                        <strong>PartCrafter</strong> – Structured 3D Mesh Generation via Compositional Latent Diffusion Transformers
                    </p>
                    <a href="https://github.com/wgsxm/PartCrafter" style="display: inline-block; vertical-align: middle; margin-left: 0.5em;">
                        <img src="https://img.shields.io/badge/GitHub-Repo-blue" alt="GitHub Repo">
                    </a>
                </div>
                """
            )
            gr.Markdown(
            """ 
            • HF Space by : [@alexandernasa](https://twitter.com/alexandernasa/)  """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="filepath", label="Input Image")
                    num_parts = gr.Slider(1, MAX_NUM_PARTS, value=4, step=1, label="Number of Parts")
                    run_button = gr.Button("Generate 3D Parts", variant="primary")
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        seed = gr.Number(value=0, label="Random Seed", precision=0)
                        num_tokens = gr.Slider(256, 2048, value=1024, step=64, label="Num Tokens")
                        num_steps = gr.Slider(1, 100, value=50, step=1, label="Inference Steps")
                        guidance = gr.Slider(1.0, 20.0, value=7.0, step=0.1, label="Guidance Scale")
                        flash_decoder = gr.Checkbox(value=False, label="Use Flash Decoder")
                        remove_bg = gr.Checkbox(value=True, label="Remove Background (RMBG)")

                with gr.Column(scale=1):
                    gr.HTML(
                        """
                        <p style="opacity: 0.6; font-style: italic;">
                          The 3D Preview might take a few seconds to load the 3D model
                        </p>
                        """
                    )
                    with gr.Row(scale=2):
                        output_model = gr.Model3D(label="Merged 3D Object")
                        output_dir = gr.Textbox(label="Export Directory", visible=False)
                    with gr.Row(scale=1):
                        download_zip = gr.File(label="Download All Parts (zip)")
                    with gr.Row(scale=3):
                        examples = gr.Examples(
                            
                            examples=[
                                [
                                    "assets/images/np5_b81f29e567ea4db48014f89c9079e403.png", 
                                    5,
                                ], 
                                [
                                    "assets/images/np7_1c004909dedb4ebe8db69b4d7b077434.png", 
                                    7,
                                ], 
                                [
                                    "assets/images/np13_b07b4d858z.png", 
                                    13,
                                ], 
                                
                            ],
                            inputs=[input_image, num_parts],
                            outputs=[output_model, output_dir, download_zip],
                            fn=run_triposg,
                            cache_examples=True,
                        )
    
            run_button.click(fn=run_triposg,
                             inputs=[input_image, num_parts, seed, num_tokens, num_steps,
                                     guidance, flash_decoder, remove_bg, session_state],
                             outputs=[output_model, output_dir, download_zip])
        return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.unload(cleanup)
    demo.queue()
    demo.launch()