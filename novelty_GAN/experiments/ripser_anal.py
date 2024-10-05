import numpy as np
import torch, os, sys,time
import matplotlib.pyplot as plt

from novelty_GAN.utils.cohomo import barycentric_additions, stitch_together, default_radius_bounds

show = False
DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))

save_dir = os.path.join(DIR, 'temp', 'cohomo_test')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

dim = 2

N = 40
torch.random.manual_seed(420)
np.random.seed(69)
novelty_radius = .1

scaling = (1, 1)

# generate random points in a box
dataset = torch.rand(N, dim)

noise = .1
# generate random points on a line
vec = torch.tensor([[1, 1.]])
dataset = torch.rand(N, 1)*vec + torch.normal(mean=0, std=noise, size=(N, dim))

min_radius, max_radius = default_radius_bounds(dataset=dataset, )
print('rad bounds', min_radius, max_radius)

start = time.time()
additions = np.stack(tuple(
    stitch_together(
        dataset=dataset.numpy(),
        max_cohomo=1,
    )))
print('time:', time.time() - start)

plt.scatter(dataset[:, 0],
            dataset[:, 1],
            color='black',
            label='dataset',
            )
plt.scatter(additions[:, 0],
            additions[:, 1],
            color='purple',
            marker='x',
            alpha=.5,
            label='additions',
            )
plt.legend()
plt.savefig(os.path.join(save_dir, 'augmented_dataset.png'))
if show:
    plt.show()
plt.close()

from ripser import ripser
from novelty_GAN.utils.cohomo import plot_diagrams

result = ripser(dataset, do_cocycles=False)
diagrams = result['dgms']
plot_diagrams(diagrams, show=False, ax=plt.gca())
plt.title('original')
plt.savefig(os.path.join(save_dir, 'original_persistence.png'))
if show:
    plt.show()
plt.close()

result = ripser(np.concatenate((dataset, additions), axis=0), do_cocycles=False)
diagrams = result['dgms']
plot_diagrams(diagrams, show=False, ax=plt.gca())
plt.savefig(os.path.join(save_dir, 'augmented_persistence.png'))
if show:
    plt.show()
plt.close()
