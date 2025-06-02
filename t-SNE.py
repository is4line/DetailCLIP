from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import models
import torch
from collections import OrderedDict
import json
import os
import torchvision.transforms as transforms
from tokenizer import SimpleTokenizer
import datasets
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

model = getattr(models, 'ICLIP_VITB16')()
model.cuda()

# Creating model
ckpt_path = 'checkpoint_best.pt'

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state_dict = OrderedDict()
for k, v in ckpt['state_dict'].items():
    state_dict[k.replace('module.', '')] = v

old_args = ckpt['args']
print("=> creating model: {}".format(old_args.model))
model = getattr(models, old_args.model)()
model.cuda()
model.load_state_dict(state_dict, strict=True)
print("=> loaded resume checkpoint '{}' (epoch {})".format(ckpt_path, ckpt['epoch']))

cwd = os.path.dirname(os.path.realpath(__file__))
print(cwd)
with open(os.path.join(cwd, 'dataset_catalog.json')) as f:
    catalog = json.load(f)

with open(os.path.join(cwd, 'templates.json')) as f:
    all_templates = json.load(f)

with open(os.path.join(cwd, 'labels.json')) as f:
    all_labels = json.load(f)

# Data loading code
print("=> creating dataset")
tokenizer = SimpleTokenizer()
val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        lambda x: x.convert('RGB'),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])

results = []
for d in catalog:
    print('Evaluating {}'.format(d))
    val_dataset = datasets.get_downstream_dataset(catalog, name=d, is_train=False, transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=10, pin_memory=True, drop_last=False)

    templates = all_templates[d]
    labels = all_labels[d]

    model.eval()

    print('=> encoding captions')
    with torch.no_grad():
        text_features = []
        for label in labels:
            if isinstance(label, list):
                texts = [t.format(l) for t in templates for l in label]
            else:
                texts = [t.format(label) for t in templates]
            texts = tokenizer(texts).cuda(non_blocking=True)
            texts = texts.view(-1, 77).contiguous()
            class_embeddings = utils.get_model(model).encode_text(texts, ema=True)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)

        text_features = torch.stack(text_features, dim=0)

        for images, target in val_loader:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # encode images
            image_features = utils.get_model(model).encode_image(images, ema=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)


# t-SNE
text_features.cpu()
text_features = pd.DataFrame(text_features)
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(text_features)

plt.figure(figsize=(16,4))
ax1 = plt.subplot(1, 3, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=tsne_pca_results,
    legend="full",
    alpha=0.3,
    ax=ax1
)