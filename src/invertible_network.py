import numpy as np
import torch as torch


class InvertibleBlock(torch.nn.Module):
    def __init__(self, dim: int):
        assert dim > 1, "invertible network block requires at least two dimensions"
        super().__init__()
        self._dim = dim
        self._upper_lane_dim = np.random.randint(low=1, high=dim - 1)  # keep at least one dimension for the upper/lower lane
        self._lower_lane_dim = self._dim - self._upper_lane_dim
        assert self._upper_lane_dim + self._lower_lane_dim == self._dim, "upper plus lower lane dimension must match the total dimension"
        self._s1 = InvertibleBlock._make_sub_module(self._upper_lane_dim, self._lower_lane_dim)
        self._t1 = InvertibleBlock._make_sub_module(self._upper_lane_dim, self._lower_lane_dim)
        self._s2 = InvertibleBlock._make_sub_module(self._lower_lane_dim, self._upper_lane_dim)
        self._t2 = InvertibleBlock._make_sub_module(self._lower_lane_dim, self._upper_lane_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u1 = x[..., :self._upper_lane_dim]
        u2 = x[..., self._upper_lane_dim:]
        v1 = u1 * torch.exp(self._s2(u2)) + self._t2(u2)
        v2 = u2 * torch.exp(self._s1(v1)) + self._t1(v1)
        return torch.cat([v1, v2], dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        v1 = y[..., :self._upper_lane_dim]
        v2 = y[..., self._upper_lane_dim:]
        u2 = (v2 - self._t1(v1)) * torch.exp(-self._s1(v1))
        u1 = (v1 - self._t2(u2)) * torch.exp(-self._s2(u2))
        return torch.cat([u1, u2], dim=-1)

    @staticmethod
    def _make_sub_module(in_dim: int, out_dim: int) -> torch.nn.Module:
        hidden_dim = in_dim + out_dim
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, out_dim),
        )


class InvertibleNetwork(torch.nn.Module):
    def __init__(self, latent_dim: int, observation_dim: int, num_blocks: int):
        assert latent_dim >= observation_dim, "latent should have at least as many dimensions as the observations"
        super().__init__()
        self._latent_dim = latent_dim
        self._observation_dim = observation_dim
        self._blocks = torch.nn.ModuleList(InvertibleBlock(self._latent_dim) for _ in range(num_blocks))

    def forward(self, x: torch.Tensor, strip_padding: bool = True) -> torch.Tensor:
        result = x
        for block in self._blocks:
            result = block(result)
        if strip_padding:
            return result[..., :self._observation_dim]
        return result

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        result = y
        for block in self._blocks[::-1]:
            result = block.inverse(result)
        return result
