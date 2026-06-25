import os
import json
import cv2
import matplotlib.pyplot as plt
import random

def visualizar_coco_anotacoes(
        arquivo_coco, 
        pasta_imagens, 
        max_imagens=10):
    
    with open(arquivo_coco, 'r') as f:
        dados = json.load(f)

    imagens_info = {img['id']: img for img in dados['images']}
    anotacoes = dados['annotations']
    categorias = {cat['id']: cat['name'] for cat in dados.get('categories', [])}

    # Agrupa anotações por image_id
    anotacoes_por_imagem = {}
    for ann in anotacoes:
        img_id = ann['image_id']
        if img_id not in anotacoes_por_imagem:
            anotacoes_por_imagem[img_id] = []
        anotacoes_por_imagem[img_id].append(ann)

    # Embaralha os IDs das imagens
    lista_ids = list(imagens_info.keys())
    random.shuffle(lista_ids)

    imagens_processadas = 0
    for img_id in lista_ids:
        if imagens_processadas >= max_imagens:
            break

        img_info = imagens_info[img_id]
        caminho_img = os.path.join(pasta_imagens, img_info['file_name'])

        if not os.path.exists(caminho_img):
            print(f"Imagem não encontrada: {caminho_img}")
            continue

        imagem = cv2.imread(caminho_img)
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        for ann in anotacoes_por_imagem.get(img_id, []):
            x, y, w, h = ann['bbox']
            categoria = categorias.get(ann['category_id'], "categoria desconhecida")
            cv2.rectangle(imagem, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
            cv2.putText(imagem, categoria, (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        plt.figure(figsize=(8, 4))
        plt.imshow(imagem)
        plt.title(f"Imagem: {img_info['file_name']}")
        plt.axis('off')
        plt.show()

        imagens_processadas += 1


visualizar_coco_anotacoes(
    arquivo_coco='/home/alvaro/Desktop/mestrado/3-pig-problem/2_processing/annotation_correction/output/rec_c.json',
    pasta_imagens='/home/alvaro/Desktop/mestrado/dados/frames/rec_c/frames',
    max_imagens=10  # opcional, mostra até 5 imagens
)


