from torchvision import datasets, transforms
import time, torch, os, sys

from PIL import Image

from novelty.networks.ffn import FFN
import matplotlib.pyplot as plt
from novelty.networks.autoencoder import MNIST_Autoencoder

device = "cuda" if torch.cuda.is_available() else "cpu"

DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))

dataset_dir = os.path.join(DIR, 'data')

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
])
dataset1 = datasets.EMNIST(dataset_dir, train=True, download=True,
                           transform=transform,
                           split='balanced',
                           )
dataset2 = datasets.EMNIST(dataset_dir, train=False,
                           transform=transform,
                           split='balanced',
                           )

train_kwargs = {'batch_size': 128,
                'shuffle': True,
                # 'pin_memory': True,
                # 'pin_memory_device': device,
                }
test_kwargs = {'batch_size': 128,
               # 'pin_memory': True,
               # 'pin_memory_device': device,
               }
train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
temp_dir = os.path.join(DIR, 'temp')
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

temp = 4
for img, label in datasets.EMNIST(dataset_dir, train=True, download=False, split='balanced'):
    img: Image.Image
    print(label)
    img.save(os.path.join(temp_dir, 'test.png'))
    temp -= 1
    if temp < 0:
        break

guess_dir = os.path.join(temp_dir, 'guess')
if not os.path.exists(guess_dir):
    os.makedirs(guess_dir)

## RECONSTRUCTION TEST
print('training reconstruction')
aut = MNIST_Autoencoder(embedding_dim=8, device=device)
optim = torch.optim.Adam(aut.parameters())
start = time.time()
for inp, labels in train_loader:
    optim.zero_grad()
    inp = inp.to(device)
    enc, dec = aut.forward(inp)
    crit = torch.nn.MSELoss()
    loss = crit.forward(input=dec, target=inp)
    loss.backward()
    optim.step()
    print(loss.item(), end='\r')
print()
print('train time:', time.time() - start)
for inp, _ in test_loader:
    inp = inp.to(device)
    _, dec = aut.forward(inp)

    trans = transforms.ToPILImage()
    for i, (guess, real) in enumerate(zip(dec, inp)):
        guess, real = trans(guess), trans(real)
        guess.save(os.path.join(guess_dir, str(i) + '_guess.png'))
        real.save(os.path.join(guess_dir, str(i) + '_real.png'))
    break

# using fixed encoder for classification, also grab the encodings of each point

print('training classification with fixed encoder')
points = [[] for _ in range(47)]
head = FFN(input_dim=aut.embedding_dim, output_dim=47, hidden_layers=[64, 64], device=device)
optim = torch.optim.Adam(head.parameters())
start = time.time()
for inp, labels in train_loader:
    inp, labels = inp.to(device), labels.to(device)
    optim.zero_grad()
    enc, _ = aut.forward(inp)
    logits = head.forward(enc)

    crit = torch.nn.CrossEntropyLoss()
    loss = crit.forward(input=logits, target=labels)
    loss.backward()
    optim.step()
    print(loss.item(), end='\r')
    for vec, label in zip(enc, labels.view(-1, 1)):
        points[label].append(vec.detach().cpu())
print()
print('train time:', time.time() - start)
for inp, labels in test_loader:
    inp, labels = inp.to(device), labels.to(device)
    enc, dec = aut.forward(inp)
    logits = head.forward(enc)
    guesses = torch.argmax(logits, dim=1)
    print('success prop:', (torch.sum(guesses == labels)/len(guesses)).item())
    break

print([len(t) for t in points])
for i, pts in enumerate(points):
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    plt.scatter(x, y, marker='.', label=i)
plt.legend()
plt.savefig(os.path.join(temp_dir, '2d_embeddings.png'))
plt.close()
