# Base image
FROM python:3.11-slim

# Setup working directory
WORKDIR /app

# Copy requirements.txt and install dependencies

# copy from dir to `/app/requirements.txt`, `/app/` saved as WORKDIR
COPY docker/training/requirements.txt .

# RUN executes a command during image build. 
# pip install needs to run at build time so dependencies are baked into the image. 
RUN pip install --no-cache-dir -r requirements.txt 

# Copy the training script to the working directory
# copies to /app/train.py
COPY docker/training/train.py ./train.py

# Declare the volume mount point for models
# is a mount point for persistent storage. 
# Without it, any files written to that directory would be lost when the container stops. 
# With it, Docker can map it to host storage (or a named volume) so trained model files persist across container restarts.
VOLUME ["/app/models"]

# TODO: Set the command to run the training script
CMD ["python", "train.py"]
# CMD only runs when you start a container.