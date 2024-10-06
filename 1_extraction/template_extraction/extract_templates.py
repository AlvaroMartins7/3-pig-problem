import pandas as pd
from PIL import Image

# Função para verificar se dois retângulos se sobrepõem
def has_intersection(rect1, rect2):
    x1_min, y1_min, x1_max, y1_max = rect1['x'], rect1['y'], rect1['x'] + rect1['width'], rect1['y'] + rect1['height']
    x2_min, y2_min, x2_max, y2_max = rect2['x'], rect2['y'], rect2['x'] + rect2['width'], rect2['y'] + rect2['height']
    
    return not (x1_max <= x2_min or x1_min >= x2_max or y1_max <= y2_min or y1_min >= y2_max)

# Lê o arquivo CSV
df = pd.read_csv('data/12-00/12-00.csv')

# Itera pelas imagens únicas
for img_filename in df['filename'].unique():
    # Filtra as anotações dessa imagem específica
    img_annotations = df[df['filename'] == img_filename]
    
    # Lista para armazenar as regiões com possível conflito
    conflict_regions = set()

    # Percorre as anotações e verifica se há interseções
    for i, row1 in img_annotations.iterrows():
        rect1 = eval(row1['region_shape_attributes'])  # Converte a string em dicionário
        current_rect1 = {
            'x': rect1['x'],
            'y': rect1['y'],
            'width': rect1['width'],
            'height': rect1['height']
        }

        # Comparar com todas as outras regiões
        for j, row2 in img_annotations.iterrows():
            if i != j:  # Evitar comparar a região consigo mesma
                rect2 = eval(row2['region_shape_attributes'])
                current_rect2 = {
                    'x': rect2['x'],
                    'y': rect2['y'],
                    'width': rect2['width'],
                    'height': rect2['height']
                }
                
                if has_intersection(current_rect1, current_rect2):
                    # Adiciona ambas regiões na lista de conflito
                    conflict_regions.add(i)
                    conflict_regions.add(j)

    # Agora, extraímos apenas as regiões sem conflito
    for index, row in img_annotations.iterrows():
        if index not in conflict_regions:
            shape_attributes = eval(row['region_shape_attributes'])
            x = shape_attributes['x']
            y = shape_attributes['y']
            width = shape_attributes['width']
            height = shape_attributes['height']

            # Abre a imagem
            img_path = f'data/12-00/{img_filename}'
            with Image.open(img_path) as img:
                # Extrai a região anotada
                cropped_img = img.crop((x, y, x + width, y + height))
                
                # Salva a região recortada
                cropped_img.save(f'output/{img_filename}_region_{row["region_id"]}.jpg')

print("Extração concluída. Todas as regiões com conflito foram ignoradas.")
