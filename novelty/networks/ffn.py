"""
simple feed forward NN
"""
import torch
from torch import nn

class FFN(nn.Module):
    """
    simple feed forward network with ReLU activation and specified hidden layers
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_layers=None,
                 activation=nn.ReLU,
                 output_activation=None,
                 device=None,
                 ):
        super().__init__()
        self.device = device
        if hidden_layers is None:
            hidden_layers = []
        self.nn_layers = nn.ModuleList()
        hidden_layers = [input_dim] + list(hidden_layers)
        for i in range(len(hidden_layers) - 1):
            self.nn_layers.append(nn.Linear(hidden_layers[i],
                                            hidden_layers[i + 1],
                                            device=self.device,
                                            ))
            self.nn_layers.append(activation())
        self.nn_layers.append(nn.Linear(hidden_layers[-1],
                                        output_dim,
                                        device=self.device,
                                        ))
        if output_activation is not None:
            self.nn_layers.append(output_activation())

    def forward(self, X):
        """
        :param X: (*, input_dim)
        :return: (*, output_dim)
        """
        for layer in self.nn_layers:
            X = layer(X)
        return X

if __name__ == '__main__':
    input_dim = 4

    test = FFN(input_dim=input_dim,
               output_dim=2,
               hidden_layers=[69, 420],
               output_activation=torch.nn.Sigmoid,
               )
    optim = torch.optim.Adam(params=test.parameters())
    for i in range(1000):
        optim.zero_grad()
        stuff = torch.rand(3, input_dim)
        loss = torch.linalg.norm(test(stuff)-.69)
        loss.backward()
        optim.step()
        if not i%10:
            print(loss)
