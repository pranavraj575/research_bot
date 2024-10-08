import torch, math
from torch import nn


class AbstractPositionalEncoding(nn.Module):
    pass


class IdentityEncoding(AbstractPositionalEncoding):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.output_dim = None

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        return x


class ClassicPositionalEncoding(AbstractPositionalEncoding):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device=None):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.output_dim = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        pe = torch.zeros(1, max_len, d_model, device=device)
        pe[0, :, 0::2] = torch.sin(position*div_term)
        pe[0, :, 1::2] = torch.cos(position*div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PositionalAppender(AbstractPositionalEncoding):

    def __init__(self,
                 d_model,
                 additional_dim=None,
                 dropout=0.1,
                 max_len=5000,
                 base_period=None,
                 device=None,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.output_dim = d_model + additional_dim
        if additional_dim is None:
            additional_dim = d_model
        if base_period is None:
            base_period = -2*math.log(10000.0)/additional_dim
        self.additional_dim = additional_dim

        position = torch.arange(max_len).unsqueeze(1)
        div_term_even = torch.exp(torch.arange(0, additional_dim, 2)*base_period/2)
        div_term_odd = torch.exp((torch.arange(1, additional_dim, 2) - 1)*base_period/2)

        pe = torch.zeros(1, max_len, additional_dim, device=device)
        pe[0, :, 0::2] = torch.sin(position*div_term_even)
        pe[0, :, 1::2] = torch.cos(position*div_term_odd)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        add_shape = (x.size(0), x.size(1), self.additional_dim)
        return torch.cat((x, self.pe[:, :x.size(1), :].broadcast_to(add_shape)), dim=-1)


if __name__ == '__main__':
    d_model = 1
    p = PositionalAppender(d_model=d_model,
                           additional_dim=5,
                           base_period=-math.log(2),
                           dropout=0)
    print(p.forward(torch.zeros(1, 5, d_model)))
