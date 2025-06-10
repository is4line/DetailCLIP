import requests
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import os
import torch

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

data_path = '/home/onyxia/work/datasets/split_CUB200_2011/test/'
for folder in os.listdir(data_path):
    print(folder)
    for file in os.listdir(data_path+folder):
        raw_image = Image.open(data_path+folder+'/'+file).convert('RGB')
        text = "a picture of a bird which has"
        inputs = processor(raw_image, text, return_tensors="pt").to(device)

        out = model.generate(**inputs)
        print(file, processor.decode(out[0], skip_special_tokens=True))
