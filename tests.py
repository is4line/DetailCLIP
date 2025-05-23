import zipfile
import os

# extraction zip
zip_path = "/home/onyxia/work/DetailCLIP/archive.zip"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/home/onyxia/work/DetailCLIP/data")

checkpoint_path = 'python DetailCLIP/eval_zeroshot.py --resume /home/onyxia/work/DetailCLIP/checkpoint_best.pt'

timm_path = '/usr/local/lib/python3.12/site-packages/timm/models'

dataset_path = "/home/onyxia/work/datasets/stanford_dogs/test"
titles = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
labels = [title[10:] for title in titles]
