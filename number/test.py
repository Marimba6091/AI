from PIL import Image
from Neuron_net import NN
from normalize_data import Img_to_matrix
import os
import numpy as np


data = []
for i in range(10):
    data.append(np.array(Img_to_matrix(Image.open(f"number/imgs/{i}.png")).get_data()))

output = []
nn = NN()
nn.set_data_from("number/train_data.json")
for i in data:
    output.append(nn.predict(i))
for x, y in enumerate(output):
    print(f"img {x}")
    for z, l in enumerate(y):
        print(f"{z} - {l:.4f}")