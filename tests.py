import zipfile
import os
import numpy as np

# extraction zip
#zip_path = "/home/onyxia/work/datasets/stanford_cars.zip"
#with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#    zip_ref.extractall("/home/onyxia/work/datasets")

checkpoint_path = 'python DetailCLIP/eval_zeroshot.py --resume /home/onyxia/work/DetailCLIP/checkpoint_best.pt'

timm_path = '/usr/local/lib/python3.12/site-packages/timm/models'

dataset_path = "/home/onyxia/work/datasets/stanford_cars/test"
titles = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
print(titles)
test_data_dir = '/home/onyxia/work/datasets/stanford_dogs/test/'

def get_images(image_dir):
    img_width, img_height = 224, 224 
    channels = 3
    batch_size = 64
    num_images= 50
    image_arr_size= img_width * img_height * channels
    image_index = 0
    image_arr_size= img_width * img_height * channels
    images = np.ndarray(shape=(num_images, image_arr_size))
    labels = np.array([])                       

    for type in os.listdir(image_dir)[:50]:
        print(type)
        type_images = os.listdir(image_dir + type)
        labels= np.append(labels, type.split('-')[1])

    return (images, labels)


