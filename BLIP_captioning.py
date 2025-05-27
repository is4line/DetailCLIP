import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


data_path = '/home/onyxia/work/datasets/stanford_dogs/test/n02110958-pug/'

for file in os.listdir(data_path):
    raw_image = Image.open(data_path+file).convert('RGB')
    text = "a photography of a "
    inputs = processor(raw_image, text, return_tensors="pt")

    out = model.generate(**inputs)
    print(file, processor.decode(out[0], skip_special_tokens=True))
