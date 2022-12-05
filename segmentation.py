from semseg import show_models, show_heads, show_datasets
from pathlib import Path
from semseg.models import *
import torch
from torchvision import io
from torchvision import transforms as T
from PIL import Image
import cv2
import numpy as np


def show_image(image):
    if image.shape[2] != 3: image = image.permute(1, 2, 0)
    image = Image.fromarray(image.numpy())
    return image


ckpt = Path('./checkpoints/pretrained/segformer')
model = eval("SegFormer")(
    backbone="MiT-B3",
    num_classes=150
)

model.load_state_dict(torch.load(r"checkpoints/pretrained/segformer/segformer.b3.ade.pth"))
model.eval()

image_path = "stylized.jpg"
image = io.read_image(image_path)

image = T.CenterCrop((512, 512))(image)
image = image.float() / 255
image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
image = image.unsqueeze(0)

with torch.inference_mode():
    seg = model(image)
seg = seg.softmax(1).argmax(1).to(int)
print(seg.unique())
print(seg.shape)

im = seg.cpu().detach().numpy().T
print(type(im), im.shape, im.max(), im.min())
cv2.imshow("", im.astype(np.uint8))
cv2.waitKey(0)

# seg_map = palette[seg].squeeze().to(torch.uint8)
# show_image(seg)
