import torch
import cv2
import numpy as np
import base64
import json
import requests
from PIL import Image
from io import BytesIO
import clip

import os
from supabase import create_client, Client
from clip_onnx import clip_onnx

url: str = "https://hxetfoqzortxhqhscggo.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh4ZXRmb3F6b3J0eGhxaHNjZ2dvIiwicm9sZSI6ImFub24iLCJpYXQiOjE2ODE2MTI2NTksImV4cCI6MTk5NzE4ODY1OX0.Nj1mF3vfJB60toeopvlF0l81YsITvNMiW6bDfdctgJ4"
supabase: Client = create_client(url, key)

# onnx cannot work with cuda
model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)

# batch first
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).cpu() # [1, 3, 224, 224]
image_onnx = image.detach().cpu().numpy().astype(np.float32)

# batch first
text = clip.tokenize(["a diagram", "a dog", "a cat"]).cpu() # [3, 77]
text_onnx = text.detach().cpu().numpy().astype(np.int32)

visual_path = "clip_visual.onnx"
textual_path = "clip_textual.onnx"

onnx_model = clip_onnx(model, visual_path=visual_path, textual_path=textual_path)
onnx_model.convert2onnx(image, text, verbose=True)
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
onnx_model.start_sessions(providers=["CPUExecutionProvider"]) # cpu mode

image_features = onnx_model.encode_image(image_onnx)
text_features = onnx_model.encode_text(text_onnx)

logits_per_image, logits_per_text = onnx_model(image_onnx, text_onnx)
probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421067 0.00299571]]


# def load_image_bytes(url):
#     return requests.get(url, headers=headers).content

# def content_to_ndarray(im_bytes):
#     bytes_io = bytearray(im_bytes)
#     img = Image.open(BytesIO(bytes_io))
#     img = img.convert('RGB')
#     img = np.array(img)[:, :, ::-1].copy()
#     return img

# def load_image_from_url(url):
#     return content_to_ndarray(load_image_bytes(url))

# headers = {
#     'user-agent':
#         'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'
# }

# def init():
#     global model
#     sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
#     sam.to("cuda")
#     model = SamPredictor(sam)

# def url_to_image(url):
#     image = load_image_from_url(url)
#     return image

# def inference(image_path):
#     global model
#     image = url_to_image(image_path)
#     model.set_image(image)
#     image_embedding = model.get_image_embedding().cpu().numpy()
#     np.save('embedding.npy', image_embedding)

#     with open('embedding.npy', 'rb+') as f:
#         res = supabase.storage.from_('images').upload('embedding.npy', os.path.abspath('embedding.npy'))

#     return "done"