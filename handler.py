import runpod
import base64
import sys
import torch
import numpy as np
from io import BytesIO
from PIL import Image

# Add repo paths so we can import their modules
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Import pipelines directly from the repo's code
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

# --- Load Models (Global Scope for Caching) ---
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Shape Pipeline...")
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('./weights', torch_dtype=torch.float16).to(device)

print("Loading Paint Pipeline...")
# Configuration matches the official repo defaults
paint_config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
paint_pipeline = Hunyuan3DPaintPipeline(paint_config, device=device)

def handler(job):
    job_input = job['input']
    
    # 1. Decode Input Image
    image_b64 = job_input.get('image')
    if not image_b64:
        return {"error": "No image provided. Send JSON with {'image': 'base64string'}"}
    
    image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
    
    # 2. Generate Shape (Mesh)
    # The pipeline returns a Trimesh object
    mesh = shape_pipeline(image=image)[0]
    
    # 3. Generate Texture (Optional but recommended)
    if job_input.get('texture', True):
        mesh = paint_pipeline(mesh, image)
    
    # 4. Export to GLB (Binary)
    output_buffer = BytesIO()
    mesh.export(output_buffer, file_type='glb')
    output_b64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
    
    return {
        "glb_base64": output_b64
    }

runpod.serverless.start({"handler": handler})