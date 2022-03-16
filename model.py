import torch
from torch import nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer as _BertLayer

from utils import pos_encoding


class SafeEmbedding(nn.Embedding):
    "Handle unseen id"

    def forward(self, input):
        output = torch.empty((*input.size(), self.embedding_dim),
                             device=input.device,
                             dtype=self.weight.dtype)

        seen = input < self.num_embeddings
        unseen = seen.logical_not()

        output[seen] = super().forward(input[seen])
        output[unseen] = torch.zeros_like(
            self.weight[0]).expand(unseen.sum(), -1)
        return output


class FlattenBatchNorm1d(nn.BatchNorm1d):
    "BatchNorm1d that treats (N, C, L) as (N*C, L)"

    def forward(self, input):
        sz = input.size()
        return super().forward(input.view(-1, sz[-1])).view(*sz)


class BertLayer(_BertLayer):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)[0]


class MemBertLayer(BertLayer):
    def __init__(self, *args, n_mem=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_mem = n_mem

    def forward(self, hidden, mem=None, **kwargs):
        assert hidden.dim() == 3, hidden.size()
        # 1 x L x D
        assert hidden.size(0) == 1, hidden.size()
        if mem is None:
            mem = pos_encoding(self.n_mem, hidden.size(2),
                               device=hidden.device,
                               dtype=hidden.dtype).unsqueeze(0)
        hidden = torch.cat([mem, hidden], dim=1)
        hidden = super().forward(hidden, **kwargs)
        mem, hidden = hidden[:, :mem.size(1)], hidden[:, mem.size(1):]
        return hidden, mem


class BasicLayer(nn.Module):
    def __init__(self, args, in_sz, out_sz, mha=False):
        super().__init__()
        self.args = args

        layers = [
            nn.Linear(in_sz, out_sz),
            FlattenBatchNorm1d(out_sz),
            nn.SiLU(),
        ]
        if args.dropout > 0.0:
            layers.append(nn.Dropout(p=args.dropout))
        self.layers = nn.Sequential(*layers)

        self.mha = self._maybe_get_mha(args, out_sz, mha)

    def _maybe_get_mha(self, args, size, mha):
        if not mha:
            return None
        bert_layer, kwargs = BertLayer, {}
        if args.n_mem:
            bert_layer = MemBertLayer
            kwargs['n_mem'] = args.n_mem
        return bert_layer(BertConfig(num_attention_heads=8,
                                     hidden_size=size,
                                     intermediate_size=size),
                          **kwargs)

    def forward(self, input, mem=None):
        output = self.layers(input)
        if self.mha is not None:
            args = [] if mem is None else [mem]
            return self.mha(output, *args)
        return output


class Net(nn.Module):
    """return (output, mem) if use_memory else output"""

    def __init__(self, args, n_embed, n_feature):
        super().__init__()

        self.emb = SafeEmbedding(n_embed, args.emb_dim)

        in_size = args.emb_dim + n_feature
        szs = [in_size] + args.szs

        self.mem_placeholder = None
        self.basic_layers = self._get_layers(args, szs)
        self.fc = nn.Linear(szs[-1], 1)

        # self._post_init()

    def _get_layers(self, args, szs):
        layers = nn.ModuleList([
            BasicLayer(args, in_sz, out_sz, layer_i in args.mhas)
            for layer_i, (in_sz, out_sz) in enumerate(zip(szs[:-1], szs[1:]))
        ])
        assert sum(
            isinstance(layer.mha, MemBertLayer)
            for layer in layers
        ) <= 1, 'Support at most one MemBertLayer'

        return layers

    def forward(self, x_id, x_feat, mem=None):
        x_emb = self.emb(x_id)
        output = torch.cat((x_emb, x_feat), dim=-1)

        for layer in self.basic_layers:
            if isinstance(layer.mha, MemBertLayer):
                output, mem = layer(output, mem=mem)
            else:
                output = layer(output)
        output = self.fc(output).squeeze(-1)

        if mem is not None:
            return output, mem
        return output

    def _post_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, SafeEmbedding)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
