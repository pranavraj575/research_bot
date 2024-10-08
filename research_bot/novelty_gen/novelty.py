import torch


class Novelty:
    """
    takes as input a pytorch dataset (N x M) array, each M-vector is a datapoint
    attempts to generate novel points that match the vibe of the datset
    """

    def __init__(self, dataset=None):
        self.dataset = dataset
        self.N, self.dim = None, None
        if dataset is not None:
            self.set_dataset(dataset)

    def generate_novelties(self, n, dataset=None, update=False):
        """
        returns a set of novel points to add to dataset
        Args:
            n: number of points to generate
            dataset: dataset to use, if None, uses self.dataset
            update: whether to update self.dataset with novelties
        """
        if dataset is None:
            dataset = self.dataset
        additions = self._generate_novelties(n=n, dataset=dataset)
        if update:
            self.update_dataset(additions=additions)
        return additions

    def _generate_novelties(self, n, dataset):
        """
        returns a set of novel points to add to dataset
        currently just takes a box around dataset and samples some random points
        Args:
            n: number of points to generate
            dataset: dataset to use
        Returns:
            (n,D) where D is the dimension of points
        """
        dim = dataset.shape[1]
        low = torch.min(dataset, dim=0, keepdim=True).values
        high = torch.max(dataset, dim=0, keepdim=True).values
        additions = torch.rand(n, dim)*(high - low) + low

        return additions

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.N, self.dim = self.dataset.shape

    def update_dataset(self, additions):
        if self.dataset is None:
            self.set_dataset(dataset=additions)
        else:
            self.set_dataset(torch.cat((self.dataset, additions), dim=0))

    def train(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    torch.random.manual_seed(69)
    n = Novelty(torch.rand(30, 2))
    additions = n.generate_novelties(3)

    plt.scatter(n.dataset[:, 0], n.dataset[:, 1])
    plt.scatter(additions[:, 0], additions[:, 1])

    plt.show()
