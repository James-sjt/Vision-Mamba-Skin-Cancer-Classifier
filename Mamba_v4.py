import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pscan import pscan


@dataclass
class MambaConfig:
    d_model: int
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16
    expand_factor: int = 2
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False

    pscan: bool = True
    use_cuda: bool = False

    multi_directional: bool = True

    divide_output: bool = True

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class VMamba_v4(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def step(self, x, caches):
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = VMambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x

    def step(self, x, cache):
        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class VMambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=config.d_conv - 1)

        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        if config.multi_directional:
            # A_b
            A_b = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
            self.A_log_b = nn.Parameter(torch.log(A_b))
            self.A_log_b._no_weight_decay = True
            self.conv1d_b = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                      kernel_size=config.d_conv, bias=config.conv_bias,
                                      groups=config.d_inner,
                                      padding=config.d_conv - 1)
            self.x_proj_b = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
            self.dt_proj_b = nn.Linear(config.dt_rank, config.d_inner, bias=True)
            self.D_b = nn.Parameter(torch.ones(config.d_inner))
            self.D_b._no_weight_decay = True

            # A_c
            A_c = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
            self.A_log_c = nn.Parameter(torch.log(A_c))
            self.A_log_c._no_weight_decay = True
            self.conv1d_c = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                      kernel_size=config.d_conv, bias=config.conv_bias,
                                      groups=config.d_inner,
                                      padding=config.d_conv - 1)
            self.x_proj_c = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
            self.dt_proj_c = nn.Linear(config.dt_rank, config.d_inner, bias=True)
            self.D_c = nn.Parameter(torch.ones(config.d_inner))
            self.D_c._no_weight_decay = True

            # A_d
            A_d = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
            self.A_log_d = nn.Parameter(torch.log(A_d))
            self.A_log_d._no_weight_decay = True
            self.conv1d_d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                      kernel_size=config.d_conv, bias=config.conv_bias,
                                      groups=config.d_inner,
                                      padding=config.d_conv - 1)
            self.x_proj_d = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
            self.dt_proj_d = nn.Linear(config.dt_rank, config.d_inner, bias=True)
            self.D_d = nn.Parameter(torch.ones(config.d_inner))
            self.D_d._no_weight_decay = True

        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        _, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = F.silu(x)
        y = self.ssm(x=x,
                     z=z,
                     A_log=self.A_log,
                     D=self.D,
                     x_proj=self.x_proj,
                     dt_proj=self.dt_proj)

        if self.config.multi_directional:
            xz_b = xz.flip([1])
            x_b, z_b = xz_b.chunk(2, dim=-1)
            x_b = x_b.transpose(1, 2)
            x_b = self.conv1d_b(x_b)[:, :, :L]
            x_b = x_b.transpose(1, 2)
            x_b = F.silu(x_b)
            y_b = self.ssm(x=x_b,
                           z=z_b,
                           A_log=self.A_log_b,
                           D=self.D_b,
                           x_proj=self.x_proj_b,
                           dt_proj=self.dt_proj_b)

            xz_c = xz[:, 1: -1].view(xz.shape[0], int(xz[:, 1: -1].shape[1] ** 0.5), int(xz[:, 1: -1].shape[1] ** 0.5),
                                                                                         xz.shape[-1]).transpose(1, 2)
            xz_c = xz_c.flatten(1, 2)
            xz_c = torch.cat((xz[:, 0].unsqueeze(1), xz_c, xz[:, -1].unsqueeze(1)), dim=1)

            x_c, z_c = xz_c.chunk(2, dim=-1)
            x_c = x_c.transpose(1, 2)
            x_c = self.conv1d_c(x_c)[:, :, :L]
            x_c = x_c.transpose(1, 2)
            x_c = F.silu(x_c)
            y_c = self.ssm(x=x_c,
                           z=z_c,
                           A_log=self.A_log_c,
                           D=self.D_c,
                           x_proj=self.x_proj_c,
                           dt_proj=self.dt_proj_c)

            xz_d = xz_c.flip([1])
            x_d, z_d = xz_d.chunk(2, dim=-1)
            x_d = x_d.transpose(1, 2)
            x_d = self.conv1d_d(x_d)[:, :, :L]
            x_d = x_d.transpose(1, 2)
            x_d = F.silu(x_d)
            y_d = self.ssm(x=x_d,
                           z=z_d,
                           A_log=self.A_log_d,
                           D=self.D_d,
                           x_proj=self.x_proj_d,
                           dt_proj=self.dt_proj_d)

            head_c, tail_c = y_c[:, 0].unsqueeze(1), y_c[:, -1].unsqueeze(1)
            head_d, tail_d = y_d[:, -1].unsqueeze(1), y_d[:, 0].unsqueeze(1)

            # transform y_c
            y_c = y_c[:, 1: -1].view(y_c.shape[0], int(y_c[:, 1: -1].shape[1] ** 0.5),
                                     int(y_c[:, 1: -1].shape[1] ** 0.5), y_c.shape[-1]).transpose(1, 2)
            y_c = y_c.flatten(1, 2)
            y_c = torch.cat((head_c, y_c, tail_c), dim=1)

            # transform y_d
            y_d = y_d[:, 1: -1].view(y_d.shape[0], int(y_d[:, 1: -1].shape[1] ** 0.5),
                                     int(y_d[:, 1: -1].shape[1] ** 0.5), y_d.shape[-1]).transpose(1, 2)
            y_d = y_d.flatten(1, 2)
            y_d = torch.cat((tail_d, y_d, head_d), dim=1)

        if self.config.use_cuda:
            if not self.config.multi_directional:
                return self.out_proj(y)
            else:
                if self.config.divide_output:
                    return self.out_proj((y + y_b.flip([1]) + y_c + y_d.flip([1])) / 4)
                else:
                    return self.out_proj(y + y_b.flip([1]) + y_c + y_d.flip([1]))

        z = F.silu(z)
        y = y * z
        if not self.config.multi_directional:
            return self.out_proj(y)
        else:
            z_b = F.silu(z_b)
            y_b = y_b * z_b

            # transform z_c, z_d
            head_c, tail_c = z_c[:, 0].unsqueeze(1), z_c[:, -1].unsqueeze(1)
            head_d, tail_d = z_d[:, -1].unsqueeze(1), z_d[:, 0].unsqueeze(1)

            # transform z_c
            z_c = z_c[:, 1: -1].view(z_c.shape[0], int(z_c[:, 1: -1].shape[1] ** 0.5),
                                     int(z_c[:, 1: -1].shape[1] ** 0.5), z_c.shape[-1]).transpose(1, 2)
            z_c = z_c.flatten(1, 2)
            z_c = torch.cat((head_c, z_c, tail_c), dim=1)

            # transform z_d
            z_d = z_d[:, 1: -1].view(z_d.shape[0], int(z_d[:, 1: -1].shape[1] ** 0.5),
                                     int(z_d[:, 1: -1].shape[1] ** 0.5), z_d.shape[-1]).transpose(1, 2)
            z_d = z_d.flatten(1, 2)
            z_d = torch.cat((tail_d, z_d, head_d), dim=1)

            z_c = F.silu(z_c)
            y_c = y_c * z_c

            z_d = F.silu(z_d)
            y_d = y_d * z_d

            if self.config.divide_output:
                return self.out_proj((y + y_b.flip([1]) + y_c + y_d.flip([1])) / 4)
            else:
                return self.out_proj(y + y_b.flip([1]) + y_c + y_d.flip([1]))

    def ssm(self, x, z, A_log, D, x_proj, dt_proj):
        A = -torch.exp(A_log.float())
        D = D.float()

        deltaBC = x_proj(x)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = dt_proj.weight @ delta.transpose(1, 2)
        if self.config.use_cuda:
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True,
                                         delta_bias=dt_proj.bias.float())
            y = y.transpose(1, 2)

        else:
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + dt_proj.bias)

            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):  # use pscan.py
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        BX = deltaB * (x.unsqueeze(-1))

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        BX = deltaB * (x.unsqueeze(-1))

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y

    def step(self, x, cache):
        h, inputs = cache

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=1)

        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv - 1]

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)

        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        A = -torch.exp(
            self.A_log.float())
        D = self.D.float()

        deltaBC = self.x_proj(x)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta))

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)

        BX = deltaB * (x.unsqueeze(-1))

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)

        h = deltaA * h + BX

        y = (h @ C.unsqueeze(-1)).squeeze(2)

        y = y + D * x

        return y, h


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
