from ultralytics import YOLO
import json
import numpy as np

# Carregar o modelo treinado
model = YOLO("experimentos/results_2/60_art/detect/train/weights/best.pt")

# Avaliar o modelo nos dados de teste
metrics = model.val(data="dataset.yaml", split="test")

# Calcular métricas médias para todas as classes
precision = np.mean(metrics.box.p)  # Precisão média
recall = np.mean(metrics.box.r)  # Recall médio

# Exibir as métricas
print("Métricas calculadas:")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Salvar as métricas em um arquivo JSON
metrics_to_save = {
    "mAP50": metrics.box.map50,
    "mAP50-95": metrics.box.map,
    "Precisão": precision,
    "Recall": recall,
}

with open("metrics_results.json", "w") as f:
    json.dump(metrics_to_save, f, indent=4)

print("Métricas salvas em 'metrics_results.json'")
