import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


data_path = '/home/onyxia/work/datasets/split_CUB200_2011/test/'
for folder in os.listdir(data_path):
    print(folder)
    for file in os.listdir(data_path+folder):
        raw_image = Image.open(data_path+folder+'/'+file).convert('RGB')
        text = "a photography of a "
        inputs = processor(raw_image, text, return_tensors="pt")

        out = model.generate(**inputs)
        print(file, processor.decode(out[0], skip_special_tokens=True))
