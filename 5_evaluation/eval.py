"""Evaluate a trained YOLO model on the test split and export metrics to JSON.

Computes mAP50, mAP50-95, precision, and recall on the test set defined in
dataset.yaml, then saves the results to metrics_results.json.
"""

from ultralytics import YOLO
import json
import numpy as np

# Load the trained model weights
model = YOLO("experimentos/results_2/60_art/detect/train/weights/best.pt")

# Run validation on the test split
metrics = model.val(data="dataset.yaml", split="test")

# Compute mean precision and recall across all classes
precision = np.mean(metrics.box.p)
recall = np.mean(metrics.box.r)

# Display computed metrics
print("Computed metrics:")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Save metrics to a JSON file for further analysis
metrics_to_save = {
    "mAP50": metrics.box.map50,
    "mAP50-95": metrics.box.map,
    "Precision": precision,
    "Recall": recall,
}

with open("metrics_results.json", "w") as f:
    json.dump(metrics_to_save, f, indent=4)

print("Metrics saved to 'metrics_results.json'")
