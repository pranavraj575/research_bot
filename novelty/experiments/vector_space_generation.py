import torch
import matplotlib.pyplot as plt
from collections import deque
import copy

from novelty.networks.ffn import FFN

rand_dim = 10
dim = 2

generator = FFN(input_dim=rand_dim,
                output_dim=dim,
                hidden_layers=[64,64 ],
                )
discriminator = FFN(input_dim=dim,
                    output_dim=1,
                    hidden_layers=[64],
                    output_activation=torch.nn.Sigmoid,
                    )

discriminators = deque(maxlen=20)

optim_gen = torch.optim.Adam(generator.parameters())
optim_disc = torch.optim.Adam(discriminator.parameters())

N = 100

novelty_radius = .1

scaling = (1, 1)

# generate random points in a box
dataset = torch.rand(N, dim)

noise = .01
# generate random points on a line
# vec = torch.tensor([[1, 1.]])
# dataset = torch.rand(N, 1)*vec + torch.normal(mean=0, std=noise, size=(N, dim))

sample = 100
epochs = 5000
for i in range(epochs):
    optim_gen.zero_grad()
    optim_disc.zero_grad()
    generated = generator.forward(torch.rand(sample, rand_dim))
    true = dataset[torch.randint(0, N, (sample,)), :]

    stuff = torch.concatenate((generated.detach(), true), dim=0)

    truth_estimate = discriminator.forward(stuff)
    crit = torch.nn.BCELoss()
    # target is (0,0,0,0,1,1,1,1)
    target = torch.zeros(truth_estimate.shape, )
    target[sample:] = 1
    discrim_loss = crit.forward(input=truth_estimate,
                                target=target)
    discrim_loss.backward()
    optim_disc.step()
    discriminators.append(copy.deepcopy(discriminator.state_dict()))

    gen_target = torch.ones((sample, 1))
    gen_loss = 0
    for state_dict in discriminators:
        temp_disc = copy.deepcopy(discriminator)
        temp_disc.load_state_dict(state_dict=state_dict)
        crit = torch.nn.BCELoss()
        gen_loss += crit.forward(input=temp_disc.forward(generated),
                                 target=gen_target,
                                 )
    gen_loss = gen_loss/len(discriminators)
    differences = torch.linalg.norm(generated.unsqueeze(1) - dataset.unsqueeze(0), dim=-1)
    #differences=torch.topk(k=k,input=differences).values

    diff_loss = torch.mean(novelty_radius/differences)

    overall_loss = scaling[0]*gen_loss + scaling[1]*diff_loss

    overall_loss.backward()
    optim_gen.step()
    if i == epochs - 1 or not i%(epochs//10):
        discrim_dset = discriminator.forward(dataset)
        # find where discriminator thinks the dataset is true
        discrim_dset = torch.ge(discrim_dset.flatten(), .5)

        plt.scatter(dataset[discrim_dset, 0],
                    dataset[discrim_dset, 1],
                    color='green',
                    marker='x',
                    label='True Positives',
                    )
        plt.scatter(dataset[torch.logical_not(discrim_dset), 0],
                    dataset[torch.logical_not(discrim_dset), 1],
                    color='red',
                    marker='x',
                    label='False Negatives',
                    )

        generated = generator.forward(torch.rand(sample, rand_dim)).detach()

        discrim_gen = discriminator.forward(generated)
        # where discriminator thinks these are real
        discrim_gen = torch.ge(discrim_gen.flatten(), .5)

        plt.scatter(generated[discrim_gen, 0],
                    generated[discrim_gen, 1],
                    color='red',
                    label='False Postives',
                    )
        plt.scatter(generated[torch.logical_not(discrim_gen), 0],
                    generated[torch.logical_not(discrim_gen), 1],
                    color='green',
                    label='True Negatives',
                    )
        plt.legend()
        plt.show()
