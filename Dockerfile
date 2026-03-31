FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /app

# System dependencies for opencv/scikit-image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install faiss-gpu separately (replaces faiss-cpu in requirements.txt)
RUN pip install --no-cache-dir faiss-gpu

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Install patchcore package in development mode
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python"]
