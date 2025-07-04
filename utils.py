import cv2
import torch
import numpy as np

def preprocess(img):
    img = cv2.resize(img, (28, 28))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    return img
