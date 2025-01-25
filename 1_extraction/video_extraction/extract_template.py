import os
import json
import random
from PIL import Image

# Função para verificar se dois retângulos se sobrepõem
def has_intersection(rect1, rect2):
    x1_min, y1_min, x1_max, y1_max = rect1['x'] - rect1['width'] / 2, rect1['y'] - rect1['height'] / 2, rect1['x'] + rect1['width'] / 2, rect1['y'] + rect1['height'] / 2
    x2_min, y2_min, x2_max, y2_max = rect2['x'] - rect2['width'] / 2, rect2['y'] - rect2['height'] / 2, rect2['x'] + rect2['width'] / 2, rect2['y'] + rect2['height'] / 2
    
    return not (x1_max <= x2_min or x1_min >= x2_max or y1_max <= y2_min or y1_min >= y2_max)

# Caminhos
frames_dir = './output/rec_b/frames'  # Diretório com os frames
annotations_path = 'frames.json'  # Caminho para o JSON
output_dir = 'annotated_parts'  # Diretório de saída para as regiões válidas

# Número de frames e partições
num_frames_to_process = 100  # Total de frames a serem processados
num_partitions = 5  # Número de partições

# Crie o diretório de saída, se necessário
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Carregue as anotações do JSON
with open(annotations_path, 'r') as f:
    annotations = json.load(f)

# Dicionário para armazenar regiões com conflitos por frame
conflict_regions = {}

# Itera sobre os frames
for obj in annotations['objects']:
    for frame_data in obj['frames']:
        frame_number = frame_data['frameNumber']
        bbox = frame_data['bbox']

        # Adiciona as regiões de cada frame para verificar conflitos
        if frame_number not in conflict_regions:
            conflict_regions[frame_number] = []
        conflict_regions[frame_number].append(bbox)

# Seleciona frames aleatórios para processamento, sem repetição
all_frame_numbers = list(conflict_regions.keys())
selected_frame_numbers = random.sample(all_frame_numbers, min(num_frames_to_process, len(all_frame_numbers)))

# Divide os frames selecionados em partições
partition_size = len(selected_frame_numbers) // num_partitions
partitions = [selected_frame_numbers[i * partition_size:(i + 1) * partition_size] for i in range(num_partitions)]

# Processa cada partição
for partition_idx, frame_numbers in enumerate(partitions, start=1):
    # Cria diretório para a partição
    partition_dir = os.path.join(output_dir, f'partition_{partition_idx}')
    if not os.path.exists(partition_dir):
        os.makedirs(partition_dir)

    # Processa os frames da partição
    for frame_number in frame_numbers:
        bboxes = conflict_regions[frame_number]
        # Lista para armazenar índices de regiões conflitantes
        conflicting_indices = set()

        # Verifica conflitos entre as regiões
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if has_intersection(bboxes[i], bboxes[j]):
                    conflicting_indices.add(i)
                    conflicting_indices.add(j)

        # Carrega o frame correspondente
        frame_filename = os.path.join(frames_dir, f'frame_{frame_number:04d}.jpg')
        if not os.path.exists(frame_filename):
            print(f"Frame {frame_filename} não encontrado.")
            continue

        with Image.open(frame_filename) as img:
            # Processa as regiões válidas (sem conflito)
            for idx, bbox in enumerate(bboxes):
                if idx not in conflicting_indices:
                    x_center = bbox['x']
                    y_center = bbox['y']
                    width = bbox['width']
                    height = bbox['height']

                    # Calcula os limites da região baseada no centro
                    left = x_center - width / 2
                    top = y_center - height / 2
                    right = x_center + width / 2
                    bottom = y_center + height / 2

                    # Recorta a região anotada
                    cropped_img = img.crop((left, top, right, bottom))

                    # Salva a região válida na partição correspondente
                    output_filename = os.path.join(partition_dir, f'frame_{frame_number:04d}_region_{idx}.jpg')
                    cropped_img.save(output_filename)

                    print(f"Região válida salva em: {output_filename}")

print("Extração concluída. Regiões com conflito foram ignoradas.")
