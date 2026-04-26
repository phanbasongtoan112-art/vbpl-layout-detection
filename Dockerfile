# Use the official Ultralytics Docker image as the base.
# This image already contains PyTorch, CUDA, OpenCV, and all core YOLOv8 dependencies,
# ensuring that GPU support works out of the box without manual environment setup.
FROM ultralytics/ultralytics:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install any additional Python dependencies your project might need
RUN pip install --no-cache-dir -r requirements.txt

# Copy the specific project directories into the container.
# We avoid copying the entire data/ directory (especially synthetic_dataset/images)
# to keep the image lean. The .dockerignore file ensures we only copy what's needed.
COPY crawler/ crawler/
COPY scripts/ scripts/
COPY data/raw_html/ data/raw_html/

# Copy the models directory. The .dockerignore ensures only models/best.pt is copied 
# if it exists, ignoring other large model checkpoints.
COPY models/ models/

# Define the default command to run when the container starts.
# We use CMD instead of ENTRYPOINT so that users can easily override it
# to run other scripts (e.g., `docker run dld-yolov8 python scripts/train_yolo.py`).
# The default action is to run the prediction script.
CMD ["python", "scripts/predict_layout.py"]
