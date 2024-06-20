import torch
import torch.nn as nn
import torch.nn.functional as F
from Mamba_v4 import VMamba_v4, MambaConfig
from Mamba_v2 import VMamba_v2
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device=device):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


class Embedding(nn.Module):
    def __init__(self, patch_size, emb_features, is_dropout, batch_size):
        super(Embedding, self).__init__()
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.is_dropout = is_dropout
        self.linear = nn.Linear((patch_size ** 2) * 3, emb_features, bias=False)

    def forward(self, x):
        num_patches = int(x.shape[-1] / self.patch_size) ** 2
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x_2 = x.contiguous().view(self.batch_size, 3, -1, self.patch_size, self.patch_size).permute(0, 2, 1, 3,
                                                                                                    4).contiguous()
        x_2 = x_2.view(self.batch_size, num_patches, -1)
        x_2 = self.linear(x_2)
        return x_2


class Classifier(nn.Module):
    def __init__(self, num_cls, d_model, is_dropout):
        super(Classifier, self).__init__()
        self.is_dropout = is_dropout
        self.dropout = nn.Dropout(0.1)
        self.linear_total1 = nn.Linear(d_model, num_cls, bias=False)

    def forward(self, axial):
        op = self.linear_total1(axial)
        if self.is_dropout:
            op = self.dropout(op)
        return F.softmax(op, -1)


class Model(nn.Module):
    def __init__(self, d_model, state_size, seq_len, batch_size, is_dropout,
                 patch_size, emb_out_features, num_cls, num_blocks, version):
        super(Model, self).__init__()
        self.batch_size = batch_size

        self.config = MambaConfig(
            d_model=d_model,
            n_layers=num_blocks,
            dt_rank='auto',
            d_state=state_size,
            expand_factor=2,
            d_conv=4,
            inner_layernorms=True,
            use_cuda=True,
        )

        self.cls_head = nn.Parameter(torch.zeros((1, 1, emb_out_features)).to(device))
        self.cls_tail = nn.Parameter(torch.zeros((1, 1, emb_out_features)).to(device))
        self.PE = nn.Parameter(torch.zeros(1, seq_len, emb_out_features).to(device))
        self.Emb_x = Embedding(patch_size, emb_out_features, is_dropout, batch_size)

        if version == 'v4':
            self.Mamba = VMamba_v4(config=self.config)
        else:
            self.Mamba = VMamba_v2(config=self.config)

        self.classifier = Classifier(num_cls, d_model, is_dropout)
        self.norm = RMSNorm(d_model, device=device)

    def forward(self, x):
        emb_x = self.Emb_x(x)
        cls_head = self.cls_head.expand(self.batch_size, -1, -1)
        cls_tail = self.cls_tail.expand(self.batch_size, -1, -1)
        emb_x = torch.cat((cls_head, emb_x, cls_tail), dim=1)
        emb_x = emb_x + self.PE

        out_x = self.Mamba(emb_x)

        # norm the output
        out_x = self.norm(out_x)

        # with cls_token
        op = self.classifier(out_x[:, 0] + out_x[0, -1])

        return op

