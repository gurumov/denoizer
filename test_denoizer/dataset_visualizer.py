import os
import numpy as np
from PIL import Image

if __name__ == "__main__":
    dirr = "/Users/simon/test"
    for f in os.listdir(dirr):
        img_path = os.path.join(dirr, f)
        img = np.asarray(Image.open(img_path).convert('L'))
        # noise = np.random.standard_normal(size=img.shape)
        noise = np.random.normal(loc=0.0, scale=10.0,  size=img.shape)
        noisy = img + noise
        noisy = np.clip(noisy, a_min=0.0, a_max=255.0)
        Image.fromarray(img).show()
        Image.fromarray(noisy).show()
        a = input()
        
