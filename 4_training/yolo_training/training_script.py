"""YOLO training script for pig detection using Ultralytics.

Trains a YOLOv11x model on the synthetic dataset, evaluates on the validation set,
runs inference on a test image, and exports the model to ONNX format.
"""

from ultralytics import YOLO

# Load a pre-trained YOLOv11x model as the starting point
model = YOLO("yolo11x.pt")

# Train the model on the synthetic pig detection dataset
train_results = model.train(
    data="dataset.yaml",  # Path to dataset YAML with train/val splits and class names
    epochs=100,           # Number of training epochs
    device="0",           # GPU device index (use "cpu" for CPU-only training)
)

# Evaluate model performance on the validation set
metrics = model.val()

# Run inference on a test image to verify detections visually
results = model("test/test.jpg")
results[0].show()

# Export the trained model to ONNX format for deployment
path = model.export(format="onnx")
