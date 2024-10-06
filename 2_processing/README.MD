1. Create a local virtual environment:

```shell
python -m venv .venv
```

2. Activate the virtual environment:

```shell
source .venv/bin/activate
```

3. Install the required dependencies:

```shell
pip install -r requirements.txt
```

An additional step is to install the `SegmentAnything (SAM)` tool from Meta.You can install it by running the following command:

```shell
pip install git+https://github.com/facebookresearch/segment-anything.git
```

After running the command, create a directory `data/checkpoints` in the root of the project. Then, download the pre-trained model from this [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and save it in the `data/checkpoints` directory.

## Dataset

The dataset used in this project provided by the [AIFARMS](https://github.com/AIFARMS/multi-camera-pig-tracking) and can be directly acessed via this [link](https://drive.google.com/drive/folders/1E2wW2aRENgy_TqlzfICn58ahbTHVIaK6). Download the directories `ISRL images` and `multicam-dataset` and place them into the created `data` directory.

## Templates extraction

Once you downloaded the dataset, you can extract the templates from the images using the `extract_templates.py` script. This script will extract the templates from the images and save them in the `data/templates` directory. To run the script, use the following command:

```shell
python -m tools.extract_template
```
