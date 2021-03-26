import torch
import numpy as np

class GetPixelLabel(object):

    def __init__(self):
        pass

    def __call__(self, img):
        img = np.asarray(img)
        img = torch.from_numpy(img.copy())
        img.apply_(lambda x: x if x!=255 else 0)
        return img

    def randomize_parameters(self):
        pass