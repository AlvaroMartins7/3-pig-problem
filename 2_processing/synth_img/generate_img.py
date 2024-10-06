import os
import argparse
import random
import datetime as dt
from concurrent.futures import ProcessPoolExecutor  # Import necessário para multiprocessamento

import synth_tools as st  # my custom module


def write_annotation(annot_folder, img_filename, data_array):
    # Criação do arquivo de anotação no formato YOLO
    file_path = os.path.join(annot_folder, f"{img_filename}.txt")
    with open(file_path, 'a') as file:
        data_line = ' '.join(map(str, data_array))
        file.write(data_line + '\n')


def process_template(background, template_folder, annot_folder, img_filename):
    # Processa um template e retorna sucesso ou falha
    template = st.get_random_image(template_folder).convert('RGBA')
    template = st.random_resize_template(template)
    position = st.get_random_position(background, template)
    if position is None:
        return False

    st.overlay_template(background, template, position)
    bbox = st.calculate_bounding_box(position, template)
    
    yolo_annot_params = st.set_yolo_annot_params(bbox[0], bbox[1], bbox[2], bbox[3], background.size[0], background.size[1], 0)
    write_annotation(annot_folder, img_filename, yolo_annot_params)
    
    if random.random() < 0.3:
        st.overlay_template(background, template, position)
    
    return True


def merge_images(background_folder, template_folder, annot_folder, img_filename):
    # Faz a fusão dos templates sobre o fundo
    background = st.get_random_image(background_folder).convert('RGBA')
    
    num_templates = random.randint(1, 3)
    for _ in range(num_templates):
        success = process_template(background, template_folder, annot_folder, img_filename)
        if not success:
            continue

    return background


def create_folder(out_path):
    # Cria pastas de saída para imagens e anotações
    actual_dt = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(out_path, actual_dt)
    image_path = os.path.join(out_path, 'images')
    annotation_path = os.path.join(out_path, 'labels')
    os.makedirs(image_path)
    os.makedirs(annotation_path)
    return image_path, annotation_path


def generate_image(index, bgns_path, tpls_path, image_path, annotation_path):
    # Função para gerar uma imagem e salvar
    img_filename = f"image_{index:03d}"
    synth_img = merge_images(bgns_path, tpls_path, annotation_path, img_filename)
    output_path = os.path.join(image_path, f"{img_filename}.png")
    synth_img.save(output_path, format='PNG')
    print(f"{img_filename} and labels were generated successfully.")


def generate_dataset(bgns_path, tpls_path, out_path, num_imgs, max_process):
    # Gera o dataset utilizando multiprocessamento
    image_path, annotation_path = create_folder(out_path)

    with ProcessPoolExecutor(max_workers=max_process) as executor:
        executor.map(generate_image, range(num_imgs), [bgns_path]*num_imgs, [tpls_path]*num_imgs, [image_path]*num_imgs, [annotation_path]*num_imgs)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate a dataset with templates.')
    parser.add_argument('--backgrounds-path', dest='bgns_path', type=str, required=True,
                        help='Path to the directory containing background images to be used.')
    parser.add_argument('--templates-path', dest='tpls_path', type=str, required=True,
                        help='Path to the directory containing template images to be used.')
    parser.add_argument('--out-path', dest='out_path', type=str, required=True,
                        help='Path to the directory to save the images generated to.')
    parser.add_argument('--num-imgs', dest='num_imgs', type=int, required=True,
                        help='Number of images to be generated.')
    parser.add_argument('--max-process', dest='max_process', type=int, default=1,
                        help='Maximum number of parallel processes')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    bgns_path = args.bgns_path
    tpls_path = args.tpls_path
    out_path = args.out_path
    num_imgs = args.num_imgs
    max_process = args.max_process

    generate_dataset(bgns_path, tpls_path, out_path, num_imgs, max_process)
