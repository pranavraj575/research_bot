import torch, math

from research_bot.novelty_gen.novelty import Novelty
from research_bot.networks.trans_gen import TransPointGen


class TransSymmetry(Novelty):
    def __init__(self,
                 dataset=None,
                 dim=None,
                 pos_encodings=10,
                 embedding_dim=512,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=.1,
                 max_len=5000,
                 base_period=-math.log(2),
                 device=None,
                 relative_mode=True,
                 lr=.001,
                 ):
        super().__init__(dataset=dataset)
        if dim is None:
            assert self.dim is not None
        else:
            self.dim = dim
        self.relative_mode = relative_mode
        self.max_len = max_len
        self.model = TransPointGen(dim=self.dim,
                                   pos_encodings=pos_encodings,
                                   embedding_dim=embedding_dim,
                                   nhead=nhead,
                                   num_decoder_layers=num_decoder_layers,
                                   dim_feedforward=dim_feedforward,
                                   dropout=dropout,
                                   max_len=self.max_len,
                                   base_period=base_period,
                                   device=device,
                                   )
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)

    def set_dataset(self, dataset):
        super().set_dataset(dataset=dataset)
        min_dists = []
        for i, t in enumerate(dataset):
            dists = torch.linalg.norm(dataset - t.view(1, -1), dim=-1)
            dists[i] = torch.inf
            min_dists.append(torch.min(dists))
        self.min_neighbor_dist_estimate = torch.median(torch.tensor(min_dists))

    def _generate_novelties(self, n, dataset):
        stuff = []
        for _ in range(n):
            center_pt_idx = self.select_pt_by_sparsity(radius=None, dataset=dataset)
            # (D) and (K,D)
            center_pt, neibhors = self.ctr_pt_and_nbrs(center_pt_idx=center_pt_idx,
                                                       relative=self.relative_mode,
                                                       )
            plt.scatter(center_pt[0],center_pt[1],color='purple',zorder=69)
            # (K+1,D)
            neibhors = torch.cat((torch.zeros(1, self.dim), neibhors), dim=0)
            # (1,K+1,D)
            neibhors = neibhors.unsqueeze(0)

            # (1,K+1), all False except for the first element
            replace_with_MASK = torch.zeros(neibhors.shape[:2], dtype=torch.bool)
            replace_with_MASK[0, 0] = True

            # (1,K+1,D)
            _, guesses = self.model.forward(points=center_pt.unsqueeze(0).unsqueeze(0),
                                            neighbors=neibhors,
                                            replace_with_MASK=replace_with_MASK,
                                            )
            # (D,)
            generated = guesses[0, 0].detach()
            if self.relative_mode:
                generated = generated + center_pt
            stuff.append(generated)
        return torch.stack(stuff, dim=0)

    def select_pt_by_sparsity(self, radius=None, dataset=None):
        if dataset is None:
            dataset = self.dataset
        if radius is None:
            radius = self.min_neighbor_dist_estimate
        counts = []
        for i, t in enumerate(dataset):
            dists = torch.linalg.norm(dataset - t.view(1, -1), dim=-1)
            cnt = torch.sum(dists < radius)
            counts.append(cnt.item())
        sparsity = torch.tensor(counts, dtype=torch.float)
        sparsity = 1 + torch.max(sparsity) - sparsity
        return torch.multinomial(sparsity, 1).item()

    def ctr_pt_and_nbrs(self,
                        center_pt_idx=None,
                        radius=None,
                        limited_len=True,
                        relative=True,
                        dataset=None,
                        ):
        """
        Args:
            center_pt: if specified, uses this center point
        """
        if radius is None:
            radius = self.min_neighbor_dist_estimate
        if center_pt_idx is None:
            center_pt_idx = torch.randint(self.N, ())
        if dataset is None:
            dataset = self.dataset
        center_pt = dataset[center_pt_idx]

        dists = torch.linalg.norm(dataset - center_pt.view(1, -1), dim=-1)
        dists[center_pt_idx] = torch.inf

        nb_idxs = torch.where(torch.le(dists, radius))[0]
        if limited_len:
            # grab a random subset if nb_idxs is too large
            nb_idxs = nb_idxs[torch.randperm(len(nb_idxs))[:self.max_len]]
        else:
            nb_idxs = nb_idxs[torch.randperm(len(nb_idxs))]
        neighbors = dataset[nb_idxs, :]
        if relative:
            # get relative points from center point
            neighbors = neighbors - center_pt
        return center_pt, neighbors

    def _gen_training_batch(self, batch_size, radius):
        all_centers = []
        all_nbs = []
        max_nb_size = 1
        while len(all_centers) < batch_size:
            center_pt, nbrs = self.ctr_pt_and_nbrs(radius=radius,
                                                   limited_len=True,
                                                   relative=self.relative_mode,
                                                   )

            if nbrs.shape[0]:
                # this will run on average at least half the time
                all_centers.append(center_pt)
                all_nbs.append(nbrs)
                max_nb_size = max(max_nb_size, len(nbrs))

        # (B,1,D)
        center_pts = torch.stack(all_centers, dim=0).unsqueeze(1)
        # (B,K,D), (B,K)
        neighbors = torch.zeros(batch_size, max_nb_size, self.dim)
        mask = torch.ones(batch_size, max_nb_size, dtype=torch.bool)
        for i, nbhood in enumerate(all_nbs):
            len_nbhood = len(nbhood)
            neighbors[i, :len_nbhood, :] = nbhood
            # unmask the used values
            mask[i, :len_nbhood] = False
        return center_pts, neighbors, mask

    def _get_losses(self,
                    center_pts,
                    neighbors,
                    mask,
                    radius,
                    mask_prob=.5,
                    replacemnet_probs=(.8, .1, .1),
                    risk_factor=.5,
                    ):
        """
        calculates loss
        currently for chosen delta, we use HuberLoss(delta) - risk_factor*delta*L1Loss
        we would like to use huber loss since it encourages matching the correct point,
            but does not give a large penalty if the guess is too far away
        we subtract some from the loss to further encourage risk
        also delta is chosen as the min_neighbor_dist_estimate
        Args:
            risk_factor: 0<= risk_factor < 1, amount to subsidize risky guesses
        """
        in_neighbors, mask_indices, replace_with_MASK = self._randomly_mask(
            neighbors=neighbors,
            neighbor_mask=mask,
            mask_prob=mask_prob,
            replacement_props=replacemnet_probs,
            radius=radius,
        )
        cls, guesses = self.model.forward(points=center_pts,
                                          neighbors=in_neighbors,
                                          replace_with_MASK=replace_with_MASK,
                                          ignore_mask=mask,
                                          )
        L1crit = torch.nn.L1Loss()
        delta = self.min_neighbor_dist_estimate.item()
        crit = torch.nn.HuberLoss(delta=delta)
        loss = crit.forward(
            input=guesses[mask_indices],
            target=neighbors[mask_indices],
        ) - L1crit.forward(
            input=guesses[mask_indices],
            target=neighbors[mask_indices],
        )*delta*risk_factor
        return loss

    def _randomly_mask(self, neighbors, neighbor_mask, mask_prob, replacement_props, radius):
        """
        Args:
            neighbors: size (N,K,D) tensor to randomly mask
            neighbor_mask: (N,K), True if masked
            mask_prob: proportion of elements to mask (note that we will always mask at least one per batch)
            replacement_props: proportion of ([MASK], random element, same element) to replace masked elements with
                tuple of three floats
        """
        masked_neighbors = neighbors.clone()
        # (N,K)
        which_to_mask = torch.bernoulli(torch.ones_like(neighbor_mask)*mask_prob)
        for i, msk in enumerate(neighbor_mask):
            valid_idxs = torch.where(torch.logical_not(msk))[0]
            which_to_mask[i, valid_idxs[torch.randint(0, len(valid_idxs), ())]] = True

        mask_indices = torch.where(torch.eq(which_to_mask, 1))
        map_tracker = torch.zeros_like(neighbor_mask, dtype=torch.long)
        map_tracker[mask_indices] = torch.multinomial(torch.tensor(replacement_props),
                                                      len(mask_indices[0]),
                                                      replacement=True,
                                                      ) + 1
        # this will be a tensor of 1s, 2s, and 3s, indicating whether to replace each mask with
        #   [MASK], a random element, or not to replace it
        # 0 is no replacements

        # replace these with MASK later
        replace_with_MASK = torch.eq(map_tracker, 1)

        rand_rep = torch.where(torch.eq(map_tracker, 2))
        num_rand = len(rand_rep[0])
        if num_rand:
            # (num_rand,)
            r = radius/2 + radius*torch.rand(num_rand)/2
            # (num_rand,dim)
            vec = torch.rand(num_rand, self.dim)
            vec = vec/torch.linalg.norm(vec, dim=-1).view(-1, 1)
            masked_neighbors[rand_rep] = vec*r.view(-1, 1)

        return masked_neighbors, mask_indices, replace_with_MASK

    def train(self,
              epochs=1,
              batch_size=128,
              radius=None,
              mask_probs=None,
              replacemnet_probs=(.8, .1, .1),
              ):
        if radius is None:
            radius = 3*self.min_neighbor_dist_estimate

        for epoch in range(epochs):
            print('epoch:', epoch, end='\r')
            self.optim.zero_grad()
            center_pts, neighbors, mask = self._gen_training_batch(batch_size=batch_size,
                                                                   radius=radius,
                                                                   )
            if mask_probs is None:
                mask_probs = [i/neighbors.shape[1] for i in range(neighbors.shape[1] + 1)]
            mask_prob = mask_probs[torch.randint(0, len(mask_probs), ())]
            loss = self._get_losses(center_pts=center_pts,
                                    neighbors=neighbors,
                                    mask=mask,
                                    radius=radius,
                                    mask_prob=mask_prob,
                                    replacemnet_probs=replacemnet_probs,
                                    )
            loss.backward()
            self.optim.step()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    torch.random.manual_seed(619)
    n = 10
    dataset = torch.zeros(n, n, 2)
    dataset[:, :, 0] = torch.arange(n, dtype=torch.float).view(-1, 1)
    dataset[:, :, 1] = torch.arange(n, dtype=torch.float).view(1, -1)
    dataset = dataset.reshape(-1, 2)
    sample = 69
    dataset = dataset[torch.randperm(n*n)[:sample]]
    dataset = torch.normal(dataset, .1)

    trans = TransSymmetry(dataset=dataset,
                          embedding_dim=128,
                          num_decoder_layers=5,
                          dim_feedforward=256,
                          lr=.01,
                          )
    trans.train(epochs=200)
    additions = trans._generate_novelties(n=10, dataset=trans.dataset)

    plt.scatter(dataset[:, 0], dataset[:, 1])
    plt.scatter(additions[:, 0], additions[:, 1])
    plt.show()
