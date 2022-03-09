import torch
from torch import nn

from data_module import get_features


class SafeEmbedding(nn.Embedding):
    def forward(self, input):
        output = torch.empty((*input.size(), self.embedding_dim),
                             device=input.device,
                             dtype=self.weight.dtype)

        seen = input < self.num_embeddings
        unseen = seen.logical_not()

        output[seen] = super().forward(input[seen])
        output[unseen] = self.weight.mean(dim=0)
        return output


class Net(nn.Module):
    def __init__(self, args):
        super().__init__()

        # TODO init
        self.emb = SafeEmbedding(args.n_emb, args.emb_dim)

        in_size = args.emb_dim + len(get_features())
        szs = [in_size] + args.szs

        layers = [
            layer
            for in_sz, out_sz in zip(szs[:-1], szs[1:])
            for layer in [
                nn.Linear(in_sz, out_sz),
                nn.BatchNorm1d(out_sz),
                nn.SiLU(),
            ]
        ]
        self.layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(szs[-1], 1)

        for m in self.modules():
            if isinstance(m, (nn.Linear, SafeEmbedding)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_id, x_feat):

        x_emb = self.emb(x_id)
        assert x_emb.dim() == 2, x_emb.size()

        out = torch.cat((x_emb, x_feat), dim=1)

        out = self.layers(out)
        return self.out_layer(out)
