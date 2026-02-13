# Start with RunPod's PyTorch 2.1/CUDA 12.1 image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

WORKDIR /app

# Install system dependencies required by the repo
RUN apt-get update && apt-get install -y \
    git ninja-build libgl1-mesa-glx libglib2.0-0 build-essential wget \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install runpod

# Compile the required Custom CUDA Extensions (Critical Step)
WORKDIR /app/hy3dpaint/custom_rasterizer
RUN pip install .

WORKDIR /app/hy3dpaint/DifferentiableRenderer
RUN bash compile_mesh_painter.sh

# Return to root
WORKDIR /app

# Download Model Weights during build (Faster cold starts)
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='tencent/Hunyuan3D-2.1', local_dir='./weights')"

# Copy your handler
COPY handler.py /app/handler.py

# Start command
CMD ["python3", "-u", "handler.py"]