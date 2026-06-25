import os
import argparse
import yaml
import random
import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor
from PIL import ImageDraw
import synth_tools as st  # seu módulo utilitário


def write_yolo_annotation(annot_folder, img_filename, data_array):
    file_path = os.path.join(annot_folder, f"{img_filename}.txt")
    with open(file_path, 'a') as file:
        data_line = ' '.join(map(str, data_array))
        file.write(data_line + '\n')


def write_partial_coco_annotation(annot_folder, img_filename, coco_data):
    file_path = os.path.join(annot_folder, f"coco_partial_{img_filename}.json")
    with open(file_path, 'w') as f:
        json.dump(coco_data, f)


def save_template_artifacts(alpha_folder, erosion_folder, img_filename, template_idx,
                            alpha_images, erosion_images):
    if alpha_folder and alpha_images:
        for step_idx, alpha_img in enumerate(alpha_images):
            alpha_file = os.path.join(
                alpha_folder,
                f"{img_filename}_tpl{template_idx:02d}_alpha_step{step_idx}.png"
            )
            alpha_img.save(alpha_file, format='PNG')

    if erosion_folder and erosion_images:
        for step_idx, erosion_img in enumerate(erosion_images):
            erosion_file = os.path.join(
                erosion_folder,
                f"{img_filename}_tpl{template_idx:02d}_erosion_step{step_idx}.png"
            )
            erosion_img.save(erosion_file, format='PNG')


def save_composed_template(composed_folder, img_filename, template_idx, template_img):
    if not composed_folder:
        return

    composed_file = os.path.join(
        composed_folder,
        f"{img_filename}_tpl{template_idx:02d}_composed_final.png"
    )
    template_img.save(composed_file, format='PNG')


def save_photometric_snapshot(folder_path, img_filename, suffix, image):
    if not folder_path:
        return

    file_path = os.path.join(folder_path, f"{img_filename}_{suffix}.png")
    image.save(file_path, format='PNG')


def process_template(background, templates_cache, annot_folder, img_filename, template_idx,
                     rescale_factor,
                     debug, rotation_angle, h_flip, v_flip, norm_factor, annot_format,
                     coco_image_id, coco_annot_id, alpha_folder=None, erosion_folder=None,
                     composed_folder=None):
    # Escolhe um template aleatório do cache pré-carregado
    template = random.choice(templates_cache).copy()

    template, alpha_images, erosion_images = st.filter_template(template, collect_steps=True)
    save_template_artifacts(alpha_folder, erosion_folder, img_filename, template_idx,
                            alpha_images, erosion_images)
    # Save composed template before geometric transforms to keep original template resolution
    save_composed_template(composed_folder, img_filename, template_idx, template)
    template = st.rescale_img(template, rescale_factor)
    template = st.normalize_img(template, norm_factor)
    template = st.flip_img(template, h_flip, v_flip)
    template = st.rotate_img(template, rotation_angle)

    position = st.get_random_position(background, template)
    if position is None:
        return None

    st.overlay_template(background, template, position)
    bbox = st.calculate_bounding_box(position, template)

    if debug:
        draw = ImageDraw.Draw(background)
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="yellow", width=3)

    if random.random() < 0.3:
        st.overlay_template(background, template, position)

    if annot_format == 'yolo':
        yolo_annot = st.set_yolo_annot_params(bbox[0], bbox[1], bbox[2], bbox[3],
                                              background.size[0], background.size[1], 0)
        write_yolo_annotation(annot_folder, img_filename, yolo_annot)
        return None

    elif annot_format == 'coco':
        coco_annot = st.set_coco_annot_params(bbox[0], bbox[1], bbox[2], bbox[3],
                                   coco_image_id, 0, coco_annot_id)
        return coco_annot


def merge_images(background_folder, template_folder, annot_folder, img_filename, img_res, debug,
                 noise_bg, rescale_factor, rotation_angle, h_flip, v_flip, templ_number,
                 gamma_factor, contrast_factor, sharpness_factor, color_factor, norm_factor,
                 annot_format, templates_cache, alpha_folder=None, erosion_folder=None,
                 composed_folder=None, pre_photo_folder=None, post_photo_folder=None):

    if random.random() < noise_bg:
        background = st.noise_background(img_res).convert('RGBA')
    else:
        background = st.get_random_image(background_folder).convert('RGBA')
        background = st.resize_img(background, img_res)

    annotations = []
    image_id = int(img_filename.split('_')[-1])
    annot_id = image_id * 1000

    total_templates = random.randint(1, templ_number)
    for template_idx in range(total_templates):
        result = process_template(background, templates_cache, annot_folder, img_filename, template_idx,
                                  rescale_factor,
                                  debug, rotation_angle, h_flip, v_flip, norm_factor,
                                  annot_format, image_id, annot_id,
                                  alpha_folder=alpha_folder, erosion_folder=erosion_folder,
                                  composed_folder=composed_folder)
        if result:
            annotations.append(result)
            annot_id += 1

    pre_photometric = background.copy()
    save_photometric_snapshot(pre_photo_folder, img_filename, "before_photometric", pre_photometric)

    background = st.gamma_transform(background, gamma_factor)
    background = st.contrast_transform(background, contrast_factor)
    background = st.sharpness_transform(background, sharpness_factor)
    background = st.color_transform(background, color_factor)

    post_photometric = background.copy()
    save_photometric_snapshot(post_photo_folder, img_filename, "after_photometric", post_photometric)

    return background, annotations


def generate_image(index, bgns_path, tpls_path, image_path, annotation_path, img_res,
                   debug, noise_bg, rescale_factor, rotation_angle, h_flip, v_flip, templ_number,
                   gamma_factor, contrast_factor, sharpness_factor, color_factor, norm_factor,
                   annot_format, templates_cache, alpha_folder=None, erosion_folder=None,
                   composed_folder=None, pre_photo_folder=None, post_photo_folder=None):

    img_filename = f"{index:d}"
    synth_img, annotations = merge_images(bgns_path, tpls_path, annotation_path, img_filename, img_res, debug,
                                          noise_bg, rescale_factor, rotation_angle, h_flip, v_flip, templ_number,
                                          gamma_factor, contrast_factor, sharpness_factor, color_factor, norm_factor,
                                          annot_format, templates_cache,
                                          alpha_folder=alpha_folder, erosion_folder=erosion_folder,
                                          composed_folder=composed_folder,
                                          pre_photo_folder=pre_photo_folder,
                                          post_photo_folder=post_photo_folder)

    output_path = os.path.join(image_path, f"{img_filename}.png")
    synth_img.save(output_path, format='PNG')
    print(f"{img_filename} and labels were generated successfully.")

    if annot_format == 'coco':
        coco_data = {
            "images": [{
                "id": index,
                "file_name": f"{img_filename}.png",
                "width": img_res[0],
                "height": img_res[1]
            }],
            "annotations": annotations,
            "categories": [{"id": 0, "name": "object"}]
        }
        write_partial_coco_annotation(annotation_path, img_filename, coco_data)


import json
from glob import glob

def juntar_anotacoes_coco(annot_folder, arquivo_saida='anotacoes_completas.json'):
    arquivo_saida = os.path.join(annot_folder, arquivo_saida)
    imagens = []
    anotacoes = []
    categorias = []
    proximo_id_imagem = 0
    proximo_id_anotacao = 0

    arquivos_json = sorted(glob(os.path.join(annot_folder, "coco_partial_*.json")))

    for caminho_arquivo in arquivos_json:
        with open(caminho_arquivo, 'r') as f:
            dados = json.load(f)

        # Adiciona categorias (uma vez só)
        if not categorias and 'categories' in dados:
            categorias = dados['categories']

        # Mapeia os image_id locais para novos ids globais
        id_mapeamento = {}
        for img in dados.get('images', []):
            novo_id = proximo_id_imagem
            id_mapeamento[img['id']] = novo_id
            img['id'] = novo_id
            imagens.append(img)
            proximo_id_imagem += 1

        # Atualiza anotacoes com novos image_ids e ids únicos
        for ann in dados.get('annotations', []):
            ann['id'] = proximo_id_anotacao
            ann['image_id'] = id_mapeamento.get(ann['image_id'], ann['image_id'])
            anotacoes.append(ann)
            proximo_id_anotacao += 1

    # Cria o dicionário final no formato COCO
    anotacoes_coco = {
        "images": imagens,
        "annotations": anotacoes,
        "categories": categorias
    }

    with open(arquivo_saida, 'w') as f:
        json.dump(anotacoes_coco, f, indent=2)

    print(f"Arquivo salvo com sucesso em: {arquivo_saida}")


def create_folder(out_path):
    actual_dt = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(out_path, actual_dt)
    image_path = os.path.join(out_path, 'images')
    annotation_path = os.path.join(out_path, 'labels')
    alpha_path = os.path.join(out_path, 'templates_alpha')
    erosion_path = os.path.join(out_path, 'templates_erosion')
    composed_path = os.path.join(out_path, 'templates_composed')
    pre_photo_path = os.path.join(out_path, 'images_pre_photometric')
    post_photo_path = os.path.join(out_path, 'images_post_photometric')
    os.makedirs(image_path)
    os.makedirs(annotation_path)
    os.makedirs(alpha_path)
    os.makedirs(erosion_path)
    os.makedirs(composed_path)
    os.makedirs(pre_photo_path)
    os.makedirs(post_photo_path)
    return image_path, annotation_path, alpha_path, erosion_path, composed_path, pre_photo_path, post_photo_path


def generate_dataset(bgns_path, tpls_path, out_path, num_imgs, max_process, img_res,
                     debug, noise_bg, rescale_factor, rotation_angle, h_flip, v_flip,
                     templ_number, gamma_factor, contrast_factor, sharpness_factor,
                     color_factor, norm_factor, annot_format):

    image_path, annotation_path, alpha_path, erosion_path, composed_path, pre_photo_path, post_photo_path = create_folder(out_path)

    # Pré-carrega todos os templates em memória para acelerar o processo
    templates_cache = st.preload_templates(tpls_path)

    from functools import partial
    from concurrent.futures import ThreadPoolExecutor

    worker_func = partial(
        generate_image,
        bgns_path=bgns_path,
        tpls_path=tpls_path,
        image_path=image_path,
        annotation_path=annotation_path,
        img_res=img_res,
        debug=debug,
        noise_bg=noise_bg,
        rescale_factor=rescale_factor,
        rotation_angle=rotation_angle,
        h_flip=h_flip,
        v_flip=v_flip,
        templ_number=templ_number,
        gamma_factor=gamma_factor,
        contrast_factor=contrast_factor,
        sharpness_factor=sharpness_factor,
        color_factor=color_factor,
        norm_factor=norm_factor,
        annot_format=annot_format,
        templates_cache=templates_cache,
        alpha_folder=alpha_path,
        erosion_folder=erosion_path,
        composed_folder=composed_path,
        pre_photo_folder=pre_photo_path,
        post_photo_folder=post_photo_path
    )

    with ThreadPoolExecutor(max_workers=max_process) as executor:
        executor.map(worker_func, range(num_imgs))

    if annot_format == 'coco':
        juntar_anotacoes_coco(annotation_path)


def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate a dataset with templates.')
    parser.add_argument("--config", dest='config_path', type=str, required=True,
                        help="Path to YAML configuration file.")
    parser.add_argument('--num-imgs', dest='num_imgs', type=int, required=True,
                        help='Number of images to be generated.')
    parser.add_argument('--max-process', dest='max_process', type=int, default=1,
                        help='Maximum number of parallel processes.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode to show bounding boxes on images.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config_from_yaml(args.config_path)

    bgns_path = config['bgns_path']
    tpls_path = config['tpls_path']
    out_path = config['out_path']
    noise_bg = config['noise_bg']
    img_res = tuple(config['img_res'])
    orig_tpl_res = tuple(config['orig_tpl_res'])
    rescale_factor = config['rescale_factor']
    rotation_angle = config['rotation_angle']
    h_flip = config['h_flip']
    v_flip = config['v_flip']
    templ_number = config['templ_number']
    gamma_factor = config['gamma_factor']
    contrast_factor = config['contrast_factor']
    sharpness_factor = config['sharpness_factor']
    color_factor = config['color_factor']
    annot_format = config['annot_format']

    norm_factor = st.normalization_factor(orig_tpl_res, img_res)

    generate_dataset(bgns_path, tpls_path, out_path, args.num_imgs, args.max_process, img_res,
                     args.debug, noise_bg, rescale_factor, rotation_angle, h_flip, v_flip,
                     templ_number, gamma_factor, contrast_factor, sharpness_factor,
                     color_factor, norm_factor, annot_format)
