from torch import nn


class Autoencoder(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 device=None,
                 ):
        super().__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        assert self.encoder.embedding_dim == self.decoder.embedding_dim
        self.embedding_dim = self.encoder.embedding_dim

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return enc, dec

    def decode(self, enc):
        return self.decoder(enc)

    def encode(self, x):
        return self.encoder(x)


class MNISTEncoder(nn.Module):
    def __init__(self,
                 input_shape=(1, 28, 28),
                 encoder_kernels=None,
                 encoder_channels=None,
                 encoder_hidden=None,
                 embedding_dim=32,
                 device=None,
                 ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
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
            encoder_bits.append(nn.Conv2d(old_channel, channel, kernel_size=kernel, device=self.device))
            old_channel = channel
            encoder_bits.append(nn.ReLU())

        encoder_bits.append(nn.Flatten())
        old_dim = img_shape[0]*img_shape[1]*old_channel
        for layer in encoder_hidden:
            encoder_bits.append(nn.Linear(old_dim, layer, device=self.device))
            encoder_bits.append(nn.ReLU())
            old_dim = layer
        encoder_bits.append(nn.Linear(old_dim, embedding_dim, device=self.device))
        self.encoder = nn.Sequential(*encoder_bits)

    def forward(self, x):
        enc = self.encoder(x)
        return enc


class MNISTDecoder(nn.Module):
    def __init__(self,
                 embedding_dim=32,
                 decoder_hidden=None,
                 decoding_init_shape=None,
                 decoder_channels=None,
                 decoder_kernels=None,
                 device=None,
                 ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device

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
            decoder_bits.append(nn.Linear(old_dim, layer, device=self.device))
            decoder_bits.append(nn.ReLU())
            old_dim = layer
        decoder_bits.append(nn.Unflatten(1, decoding_init_shape))
        old_channel = decoding_init_shape[0]

        for kernel, channel in zip(decoder_kernels, decoder_channels):
            decoder_bits.append(nn.ConvTranspose2d(old_channel, channel, kernel_size=kernel, device=self.device))
            old_channel = channel
            decoder_bits.append(nn.ReLU())

        self.decoder = nn.Sequential(*decoder_bits)
        decoder_bits.append(nn.Sigmoid())

    def forward(self, enc):
        dec = self.decoder(enc)
        return dec


class MNIST_Autoencoder(Autoencoder):
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
                 device=None,
                 ):
        encoder = MNISTEncoder(input_shape=input_shape,
                               encoder_kernels=encoder_kernels,
                               encoder_channels=encoder_channels,
                               encoder_hidden=encoder_hidden,
                               embedding_dim=embedding_dim,
                               device=device,
                               )
        decoder = MNISTDecoder(embedding_dim=embedding_dim,
                               decoder_hidden=decoder_hidden,
                               decoding_init_shape=decoding_init_shape,
                               decoder_channels=decoder_channels,
                               decoder_kernels=decoder_kernels,
                               device=device,
                               )
        super().__init__(encoder=encoder, decoder=decoder, device=device)
