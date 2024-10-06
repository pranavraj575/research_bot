from torchvision import datasets, transforms
import torch, os, sys
from torch import nn
from PIL import Image
from novelty.networks.ffn import FFN
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))

dataset_dir = os.path.join(DIR, 'data')

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
])
dataset1 = datasets.MNIST(dataset_dir, train=True, download=True,
                          transform=transform
                          )
dataset2 = datasets.MNIST(dataset_dir, train=False,
                          transform=transform
                          )

train_kwargs = {'batch_size': 128, 'shuffle': True}
test_kwargs = {'batch_size': 128}
train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
temp_dir = os.path.join(DIR, 'temp')
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

for img, label in datasets.MNIST(dataset_dir, train=True, download=False):
    img: Image.Image
    print(label)
    img.save(os.path.join(temp_dir, 'test.png'))
    break


class Autoencoder(nn.Module):
    def __init__(self,
                 input_shape=(1, 28, 28),
                 encoder_kernels=None,
                 encoder_channels=None,
                 encoder_hidden=None,
                 embedding_dim=32,
                 decoder_hidden=None,
                 decoding_init_shape=None,
                 decoder_channels=None,
                 decoder_kernels=None,
                 ):
        super().__init__()
        self.embedding_dim = embedding_dim
        if encoder_channels is None:
            encoder_channels = [4, 8]
        if encoder_kernels is None:
            encoder_kernels = [5 for _ in encoder_channels]
        if encoder_hidden is None:
            encoder_hidden = [128]

        encoder_bits = []
        old_channel = input_shape[0]
        img_shape = input_shape[1:]
        for kernel, channel in zip(encoder_kernels, encoder_channels):
            img_shape = [t - kernel + 1 for t in img_shape]
            encoder_bits.append(nn.Conv2d(old_channel, channel, kernel_size=kernel))
            old_channel = channel
            encoder_bits.append(nn.ReLU())

        encoder_bits.append(nn.Flatten())
        old_dim = img_shape[0]*img_shape[1]*old_channel
        for layer in encoder_hidden:
            encoder_bits.append(nn.Linear(old_dim, layer))
            encoder_bits.append(nn.ReLU())
            old_dim = layer
        encoder_bits.append(nn.Linear(old_dim, embedding_dim))
        self.encoder = nn.Sequential(*encoder_bits)

        if decoder_hidden is None:
            decoder_hidden = [128, 4000]
        if decoding_init_shape is None:
            decoding_init_shape = (10, 20, 20)
        if decoder_channels is None:
            decoder_channels = [8, 1]
        if decoder_kernels is None:
            decoder_kernels = [5 for _ in decoder_channels]

        decoder_bits = []
        old_dim = embedding_dim
        for layer in decoder_hidden:
            decoder_bits.append(nn.Linear(old_dim, layer))
            decoder_bits.append(nn.ReLU())
            old_dim = layer
        decoder_bits.append(nn.Unflatten(1, decoding_init_shape))
        old_channel = decoding_init_shape[0]

        for kernel, channel in zip(decoder_kernels, decoder_channels):
            decoder_bits.append(nn.ConvTranspose2d(old_channel, channel, kernel_size=kernel))
            old_channel = channel
            decoder_bits.append(nn.ReLU())

        self.decoder = nn.Sequential(*decoder_bits)
        decoder_bits.append(nn.Sigmoid())

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return enc, dec


guess_dir = os.path.join(temp_dir, 'guess')
if not os.path.exists(guess_dir):
    os.makedirs(guess_dir)

## RECONSTRUCTION TEST
print('training reconstruction')
aut = Autoencoder(embedding_dim=2)
optim = torch.optim.Adam(aut.parameters())
for inp, labels in train_loader:
    optim.zero_grad()
    enc, dec = aut.forward(inp)
    crit = nn.MSELoss()
    loss = crit.forward(input=dec, target=inp)
    loss.backward()
    optim.step()
    print(loss.item(), end='\r')
print()
for inp, _ in test_loader:
    _, dec = aut.forward(inp)

    trans = transforms.ToPILImage()
    for i, (guess, real) in enumerate(zip(dec, inp)):
        guess, real = trans(guess), trans(real)
        guess.save(os.path.join(guess_dir, str(i) + '_guess.png'))
        real.save(os.path.join(guess_dir, str(i) + '_real.png'))
    break

# using fixed encoder for classification, also grab the encodings of each point

print('training classification with fixed encoder')
points = [[] for _ in range(10)]
head = FFN(input_dim=aut.embedding_dim, output_dim=10, hidden_layers=[64, 64])
optim = torch.optim.Adam(head.parameters())

for inp, labels in train_loader:
    optim.zero_grad()
    enc, _ = aut.forward(inp)
    logits = head.forward(enc)

    crit = nn.CrossEntropyLoss()
    loss = crit.forward(input=logits, target=labels)
    loss.backward()
    optim.step()
    print(loss.item(), end='\r')
    for vec, label in zip(enc, labels.view(-1, 1)):
        points[label].append(vec.detach())
print()
for inp, labels in test_loader:
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
plt.savefig(os.path.join(temp_dir,'2d_embeddings.png'))
plt.close()

## CLASSIFICATION TEST, NO USE OF DECODER
aut = Autoencoder(embedding_dim=10)
optim = torch.optim.Adam(aut.parameters())

for inp, labels in train_loader:
    optim.zero_grad()
    enc, dec = aut.forward(inp)
    crit = nn.CrossEntropyLoss()
    loss = crit.forward(input=enc, target=labels)
    loss.backward()
    optim.step()
    print(loss.item(), end='\r')
print()
for inp, labels in test_loader:
    enc, dec = aut.forward(inp)
    guesses = torch.argmax(enc, dim=1)
    print('success prop:', (torch.sum(guesses == labels)/len(guesses)).item())
    break
