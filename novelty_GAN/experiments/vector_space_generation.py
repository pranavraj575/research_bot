import torch
import matplotlib.pyplot as plt

from novelty_GAN.networks.ffn import FFN

rand_dim = 1
dim = 2

generator = FFN(input_dim=rand_dim,
                output_dim=dim,
                hidden_layers=[64, 64],
                )
discriminator = FFN(input_dim=rand_dim,
                    output_dim=dim,
                    hidden_layers=[64, 64],
                    output_activation=torch.nn.Sigmoid,
                    )
N = 100
# generate random points in a box
dataset = torch.rand(N, dim)

# generate random points on a line
vec = torch.rand(1, dim)
dataset = torch.rand(N, 1)*vec

plt.plot(dataset[:,0],dataset[:,1])
plt.show()
