import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import time

from novelty_GAN.utils.cohomo import barycentric_additions

dim = 2

N = 200

novelty_radius = .1

scaling = (1, 1)

# generate random points in a box
dataset = torch.rand(N, dim)

noise = .01
# generate random points on a line
# vec = torch.tensor([[1, 1.]])
# dataset = torch.rand(N, 1)*vec + torch.normal(mean=0, std=noise, size=(N, dim))

plt.scatter(dataset[:, 0],
            dataset[:, 1],
            color='black',
            label='dataset',
            )
start = time.time()
temp = np.stack(tuple(
    barycentric_additions(
        dataset=dataset.numpy(),
        max_cohomo=1,
    )))
print('time:', time.time() - start)

plt.scatter(temp[:, 0],
            temp[:, 1],
            color='purple',
            marker='x',
            label='additions',
            )
plt.legend()
plt.show()
