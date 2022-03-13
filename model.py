import torch
from torch import nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer as _BertLayer


class SafeEmbedding(nn.Embedding):
    "Handle unseen id"

    def forward(self, input):
        output = torch.empty((*input.size(), self.embedding_dim),
                             device=input.device,
                             dtype=self.weight.dtype)

        seen = input < self.num_embeddings
        unseen = seen.logical_not()

        output[seen] = super().forward(input[seen])
        output[unseen] = torch.zeros_like(self.weight[0])
        # output[unseen] = self.weight.mean(dim=0).detach()
        return output


class FlattenBatchNorm1d(nn.BatchNorm1d):
    "BatchNorm1d that treats (N, C, L) as (N*C, L)"

    def forward(self, input):
        sz = input.size()
        return super().forward(input.view(-1, sz[-1])).view(*sz)


class BertLayer(_BertLayer):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)[0]


class Net(nn.Module):
    def __init__(self, args, n_feature):
        super().__init__()

        self.emb = SafeEmbedding(args.n_emb, args.emb_dim)

        in_size = args.emb_dim + n_feature
        szs = [in_size] + args.szs

        self.layers = nn.Sequential(*self.get_layers(args, szs))

        self.post_init()

    def get_layers(self, args, szs):
        layers = []

        for layer_i, (in_sz, out_sz) in enumerate(zip(szs[:-1], szs[1:])):
            layers.append(nn.Linear(in_sz, out_sz))
            layers.append(FlattenBatchNorm1d(out_sz))
            layers.append(nn.SiLU(inplace=True))

            if args.dropout > 0.0:
                layers.append(nn.Dropout(p=args.dropout, inplace=True))

            if layer_i in args.mhas:
                layers.append(BertLayer(BertConfig(
                    num_attention_heads=8,
                    hidden_size=out_sz,
                    intermediate_size=out_sz)))

        layers.append(nn.Linear(szs[-1], 1))
        return layers

    def forward(self, x_id, x_feat):
        x_emb = self.emb(x_id)
        out = torch.cat((x_emb, x_feat), dim=-1)
        return self.layers(out).squeeze(-1)

    def post_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, SafeEmbedding)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
