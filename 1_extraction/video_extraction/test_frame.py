import cv2
import json
import os
from PIL import Image

# Caminho para o diretório onde os frames foram salvos
frames_dir = './output/2024-11-06_15-31-30/frames'

# Caminho para o arquivo JSON com as anotações
annotations_path = 'frames.json'
# Diretório para salvar o frame anotado
output_dir = 'tested_frames'

# Frame que você deseja processar (número do frame)
frame_to_process = 2050  # Altere este número para o frame desejado

# Crie o diretório de saída, se necessário
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Carregue as anotações do JSON
with open(annotations_path, 'r') as f:
    annotations = json.load(f)

# Flag para verificar se o frame foi encontrado
frame_found = False

# Carregue o frame específico uma vez
frame_filename = os.path.join(frames_dir, f'frame_{frame_to_process:04d}.jpg')
image = cv2.imread(frame_filename)

if image is None:
    print(f"Frame {frame_filename} não encontrado.")
else:
    frame_found = True

    # Percorra cada objeto no JSON para desenhar todas as bounding boxes desse frame
    for obj in annotations['objects']:
        for frame_data in obj['frames']:
            if frame_data['frameNumber'] == frame_to_process:
                bbox = frame_data['bbox']
                
                # As coordenadas (x, y) representam o centro da bounding box
                x_center, y_center = bbox['x'], bbox['y']
                width, height = bbox['width'], bbox['height']
                
                # Calcule o canto superior esquerdo
                top_left = (int(x_center - width / 2), int(y_center - height / 2))
                bottom_right = (int(x_center + width / 2), int(y_center + height / 2))

                # Desenhe a bounding box na imagem
                color = (0, 255, 0)  # Cor verde para as bounding boxes
                thickness = 5  # Espessura da linha

                cv2.rectangle(image, top_left, bottom_right, color, thickness)

    # Converta a imagem do formato OpenCV para Pillow (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Exiba a imagem com Pillow
    pil_image.show()

    # Salve a imagem anotada (descomente se necessário)
    '''
    output_filename = os.path.join(output_dir, f'annotated_frame_{frame_to_process:04d}.jpg')
    cv2.imwrite(output_filename, image)
    print(f"Frame {frame_to_process} processado e salvo em '{output_dir}'.")
    '''

if not frame_found:
    print(f"O frame {frame_to_process} não foi encontrado nas anotações.")
