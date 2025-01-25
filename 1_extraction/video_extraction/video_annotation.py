import json
import os

# Parâmetros da imagem (substitua pelos valores do seu vídeo)
image_width = 3840  # Largura do frame
image_height = 2160  # Altura do frame

# Caminho para o arquivo JSON de anotações
json_path = "frames.json"
output_dir = "yolo_annotations"

# Criar o diretório de saída
os.makedirs(output_dir, exist_ok=True)

# Carregar as anotações do JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Processar cada objeto
for obj in data["objects"]:
    for frame in obj["frames"]:
        frame_number = frame["frameNumber"]
        bbox = frame["bbox"]
        is_ground_truth = frame.get("isGroundTruth", "0")

        # Verificar se é uma anotação válida
        if is_ground_truth != "1":
            continue

        # Considerando que as coordenadas são do centro
        x_center = bbox["x"] / image_width
        y_center = bbox["y"] / image_height
        width = bbox["width"] / image_width
        height = bbox["height"] / image_height

        # Usar 0 como class_id por padrão (substitua conforme necessário)
        class_id = 0
        yolo_annotation = f"{class_id} {x_center} {y_center} {width} {height}\n"

        # Nome do arquivo no formato "frame_XXXX.txt"
        output_file = os.path.join(output_dir, f"frame_{frame_number:04d}.txt")

        # Salvar a anotação no arquivo correspondente
        with open(output_file, "a") as out_f:
            out_f.write(yolo_annotation)

print(f"Anotações convertidas salvas em: {output_dir}")
