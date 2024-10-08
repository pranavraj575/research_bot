import shutil

from torchvision import datasets, transforms
import time, torch, os, sys

from research_bot.networks.ffn import FFN
from research_bot.networks.autoencoder import MNIST_Autoencoder

reset = False
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.random.manual_seed(69)
DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))

dataset_dir = os.path.join(DIR, 'data')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomRotation(degrees=(90, 90)),
])
dataset1 = datasets.EMNIST(dataset_dir, train=True, download=True,
                           transform=transform,
                           split='balanced',
                           )
dataset2 = datasets.EMNIST(dataset_dir, train=False,
                           transform=transform,
                           split='balanced',
                           )

train_kwargs = {'batch_size': 512,
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
embedding_dim = 8
temp_dir = os.path.join(DIR, 'temp', 'emnist_' + str(embedding_dim))
test_dir = os.path.join(temp_dir, 'test')
for t in temp_dir, test_dir:
    if not os.path.exists(t):
        os.makedirs(t)

num_classes = 47

done = [False for _ in range(num_classes)]
for img, label in datasets.EMNIST(dataset_dir,
                                  transform=transform,
                                  train=True,
                                  download=False,
                                  split='balanced',
                                  ):
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(test_dir, str(int(label)) + '.png'))
    done[int(label)] = True
    if all(done):
        break
guess_dir = os.path.join(temp_dir, 'guess')
if not os.path.exists(guess_dir):
    os.makedirs(guess_dir)

model_file = os.path.join(temp_dir, 'model.pkl')
## RECONSTRUCTION TEST
aut = MNIST_Autoencoder(embedding_dim=embedding_dim, device=device)
if not reset and os.path.exists(model_file):
    aut.load_state_dict(torch.load(model_file, weights_only=True))
else:
    optim = torch.optim.Adam(aut.parameters())
    print('training reconstruction')
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
    torch.save(aut.state_dict(), model_file)
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
head_file = os.path.join(temp_dir, 'head.pkl')
head = FFN(input_dim=aut.embedding_dim,
           output_dim=num_classes,
           hidden_layers=[64, 64],
           device=device,
           )
if not reset and os.path.exists(head_file):
    head.load_state_dict(torch.load(head_file, weights_only=True))
else:
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
    print()
    print('train time:', time.time() - start)
    torch.save(head.state_dict(), head_file)
embed_file = os.path.join(temp_dir, 'embed.pkl')
if not reset and os.path.exists(embed_file):
    points = torch.load(embed_file, weights_only=True)
else:
    print('recording test embeddings')
    points = [[] for _ in range(num_classes)]
    start = time.time()
    for i, (inp, labels) in enumerate(test_loader):
        print(i, '/', len(test_loader), end='\r')
        inp, labels = inp.to(device), labels.to(device)
        enc, _ = aut.forward(inp)
        for vec, label in zip(enc, labels.view(-1, 1)):
            points[label].append(vec.detach().cpu().numpy())

    print([len(t) for t in points])
    points = torch.stack([torch.tensor(t[torch.randint(0, len(t), (1,))]) for t in points], dim=0)
    print(points)
    print(points.shape)
    print('time:', time.time() - start)
    torch.save(points, embed_file)
for inp, labels in test_loader:
    inp, labels = inp.to(device), labels.to(device)
    enc, dec = aut.forward(inp)
    logits = head.forward(enc)
    guesses = torch.argmax(logits, dim=1)
    print('success prop:', (torch.sum(guesses == labels)/len(guesses)).item())
    break

from ripser import ripser
import numpy as np
from research_bot.novelty_gen.cohomo import plot_diagrams, default_radius_bounds, stitch_together
import matplotlib.pyplot as plt

max_cohomo = 3
test_cohomo = 6

points = points[:10].numpy()
min_rad, max_rad = default_radius_bounds(points)
min_rad=2.9
print('rad bounds (', min_rad, ',', max_rad, ')')
result = ripser(points, maxdim=test_cohomo)
print([('hom', i, 'cnt', len(arr)) for i, arr in enumerate(result['dgms'])])
plot_diagrams(result['dgms'])
plt.savefig(os.path.join(temp_dir, 'sampled_cohomo.png'))
plt.close()

additions = np.stack(list(
    stitch_together(points,
                    min_radius=min_rad,
                    max_radius=max_rad,
                    max_cohomo=max_cohomo,
                    depth=3,
                    check_simplices=False,
                    )
))
print('number of additions', len(additions))
imgs = aut.decode(torch.tensor(additions).to(device))

creation_dir = os.path.join(temp_dir, 'creation')
if os.path.exists(creation_dir):
    shutil.rmtree(creation_dir)
os.makedirs(creation_dir)

for i, img in enumerate(imgs):
    trans = transforms.ToPILImage()
    trans(img).save(os.path.join(creation_dir, str(i) + '.png'))

result = ripser(np.concatenate((points, additions), axis=0), maxdim=max_cohomo)
print([('hom', i, 'cnt', len(arr)) for i, arr in enumerate(result['dgms'])])
plot_diagrams(result['dgms'])
plt.savefig(os.path.join(temp_dir, 'updated_cohomo.png'))
plt.close()
