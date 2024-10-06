# EXTRAÇÃO DE TEMPLATES

## Objetivo

Extrair templates de datasets de imagem via suas anotações, lidando com o problema de sobreposição de objetos

## Dataset utilizado

O dataset utilizado foi produzido pela [AIFARMS](https://github.com/AIFARMS/multi-camera-pig-tracking) e pode ser acessado por este [link](https://drive.google.com/drive/folders/1E2wW2aRENgy_TqlzfICn58ahbTHVIaK6). Baixe o diretório `ISRL images` e coloque um diretório com imagens e anotações (exemplo: "12-00") e o coloque na pasta "data"

## Passos para usar a extração de templates:

### 1. Criar um ambiente virtual python:

```
python3 -m venv .venv
```

### 2. Ativar o ambiente criado:

```
source .venv/bin/activate
```

### 3. Instalar as dependências:

```
pip install -r requirements.txt
```

### 4. Se certificar de que o apontamento para o diretório de dataset está pronto

### 5. Rodar a extração:

```
python3 extract_templates.py
```
