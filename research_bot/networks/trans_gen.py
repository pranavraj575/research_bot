import torch
from torch import nn

from research_bot.networks.positional_encoder import PositionalAppender


class TransPointGen(nn.Module):
    def __init__(self,
                 dim,
                 pos_encodings=10,
                 embedding_dim=512,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=.1,
                 max_len=5000,
                 base_period=None,
                 device=None,
                 ):
        super().__init__()
        self.device = device
        # linear map from single origin point to embedding dim
        self.interface_origin = nn.Linear(in_features=dim,
                                          out_features=embedding_dim,
                                          device=device,
                                          )
        # pos enc of neighbor points
        self.pos_enc = PositionalAppender(d_model=dim,
                                          additional_dim=pos_encodings,
                                          dropout=dropout,
                                          max_len=max_len,
                                          base_period=base_period,
                                          device=device,
                                          )
        # linear map from neighbor points to embedding dim
        self.interface = nn.Linear(in_features=self.pos_enc.output_dim,
                                   out_features=embedding_dim,
                                   device=device,
                                   )
        # transformer to get contextual points
        self.trans = nn.Transformer(d_model=embedding_dim,
                                    nhead=nhead,
                                    num_encoder_layers=1,
                                    num_decoder_layers=num_decoder_layers,
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout,
                                    batch_first=True,
                                    device=device,
                                    )
        self.interface_out = nn.Linear(in_features=embedding_dim,
                                       out_features=dim,
                                       device=device,
                                       )
        self.tokens = nn.Embedding(
            num_embeddings=2,
            embedding_dim=embedding_dim,
            device=device,
        )

        # to get MASK and CLS, use self.tokens(self.MASK), self.tokens(self.CLS)
        self.MASK = 0
        self.CLS = 1

    def add_noise(self, points, stdev=None):
        """
        returns noisy points
        Args:
            points: (N,K,D)
            stdev: None or float or (D,)
                stdev on each dimension
        """
        if stdev is None:
            # (D,)
            stdev = torch.std(points.view(-1, points.shape[-1]), dim=0)
        if not torch.is_tensor(stdev):
            # (1,)
            stdev = torch.tensor([stdev], device=self.device)
        # (1,1,1) or (1,1,D)
        stdev = stdev.unsqueeze(0).unsqueeze(0)

        return torch.normal(mean=points, std=stdev, )

    def forward(self, points, neighbors, replace_with_MASK=None, ignore_mask=None):
        """
        batch size N, input dim D, sequence dim K, embedding dim E
        Args:
            points: (N,1,D)
            neighbors: (N,K,D)
            replace_with_MASK: (N,K) boolean or None, true if we replace em with mask
            ignore_mask: (N,K) boolean or None
        Returns:
            cls (N,E) and guesses (N,K,D)
        """
        # (N,1,E)
        src = self.interface_origin.forward(points)
        N = src.shape[0]
        # (N,1,E)
        cls = self.tokens(torch.tensor([self.CLS for _ in range(N)], device=self.device)).unsqueeze(1)

        # (N,K,E)
        embedded_nbhs = self.interface(self.pos_enc.forward(neighbors))
        if replace_with_MASK is not None:
            embedded_nbhs[replace_with_MASK] = self.tokens.forward(torch.tensor(self.MASK))

        # (N,K+1,E)
        tgt = torch.cat((cls, embedded_nbhs), dim=1)

        if ignore_mask is not None:
            # (N,K+1)
            # dont mask the first element, as this is the cls embedding
            ignore_mask = torch.cat((torch.zeros(N, 1), ignore_mask), dim=1)

        # (N,K+1,E)
        out = self.trans.forward(src=src,
                                 tgt=tgt,
                                 tgt_key_padding_mask=ignore_mask,
                                 )
        # (N,E)
        cls_out = out[:, 0, :]
        # (N,K,D)
        gen_out = self.interface_out.forward(out[:, 1:, :])
        return cls_out, gen_out


if __name__ == '__main__':
    dim = 3
    trans = TransPointGen(dim=dim, embedding_dim=16)

    cls_out, gen_out = trans.forward(points=torch.rand(2, 1, dim),
                                     neighbors=torch.rand(2, 4, dim),
                                     )
    print(cls_out.shape, gen_out.shape)
    print(trans.add_noise(torch.zeros(1, 2, dim), stdev=1.))
