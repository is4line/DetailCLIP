from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from PIL import Image
import torch
import os

processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

data_path = '/home/onyxia/work/datasets/split_CUB200_2011/test/'
for folder in os.listdir(data_path):
    for file in os.listdir(data_path+folder):
        raw_image = Image.open(data_path+folder+'/'+file).convert('RGB')
        pixel_values = processor(images=raw_image, return_tensors="pt").to(device).pixel_values

        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(file, generated_caption)