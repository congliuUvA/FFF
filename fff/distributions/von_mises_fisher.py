import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution, constraints
from torch.distributions.utils import lazy_property

try:
    from hyperspherical_vae.distributions import VonMisesFisher
    from hyperspherical_vae.ops.ive import ive
except ImportError:
    VonMisesFisher = object
    warnings.warn("Could not import hyperspherical_vae, VonMisesFisherMixtureDistribution will not be available.")


class VonMisesFisherMixtureDistribution(Distribution, nn.Module):
    def __init__(self, manifold, n_modes=1, device=None, init="random_uniform", learnable=False, *args, **kwargs):
        nn.Module.__init__(self)
        super().__init__(*args, event_shape=manifold.shape, **kwargs)
        self.n_modes = n_modes
        self.manifold = manifold
        self.dim = manifold.shape[0]
        self.mixture_weights_logits = nn.Parameter(torch.ones(n_modes, device=device) / n_modes,
                                                   requires_grad=learnable)

        if isinstance(init, str):
            if init == "random_uniform":
                locs = self.manifold.random_uniform(n_samples=n_modes).to(device).reshape(n_modes, self.dim)
                scales = torch.ones(n_modes, 1, device=device) * np.sqrt(2) * n_modes
            elif init == "pole":
                locs = torch.zeros(n_modes, self.dim, device=device)
                locs[:, 0] = 1
                scales = torch.ones(n_modes, 1, device=device) * np.sqrt(2)
            elif init == "fixed_uniform":
                if n_modes > 2 * self.dim:
                    raise ValueError(
                        f"Too many modes for fixed_uniform init: {n_modes}. Can only take {2 * self.dim} modes.")
                locs = torch.zeros(n_modes, self.dim, device=device)
                for i in range(n_modes):
                    locs[i, i % self.dim] = 1 - (i % 2) * 2
                scales = torch.ones(n_modes, 1, device=device) * np.sqrt(2) * n_modes
            else:
                raise ValueError(f"Unknown init {init}")
        elif isinstance(init, tuple):
            locs = init[:, 0].to(device)
            scales = init[:, 1].to(device)
            assert locs.shape == (n_modes, self.dim)
            assert scales.shape == (n_modes, 1)
        else:
            raise ValueError(f"Unknown init {init}")
        self.von_mises_fisher_distributions = []
        for loc, scale in zip(locs, scales):
            self.von_mises_fisher_distributions.append(VonMisesFisherTrainable(
                self.manifold,
                nn.Parameter(loc, requires_grad=learnable),
                nn.Parameter(scale, requires_grad=learnable)))

        self.von_mises_fisher_distributions = nn.ModuleList(self.von_mises_fisher_distributions)

    @property
    def device(self):
        return self.mixture_weights_logits.device

    def rsample(self, sample_shape=torch.Size()):
        mixture_distribution = torch.distributions.Categorical(probs=self.mixture_weights)
        mixture_indices = mixture_distribution.sample(sample_shape)
        unique_indices, counts = mixture_indices.unique(return_counts=True)
        samples = torch.zeros(*sample_shape, self.dim, device=self.device)

        # Sample from each vmf the number of times indicated by counts
        for idx, count in zip(unique_indices, counts):
            samples[mixture_indices == idx] = self.von_mises_fisher_distributions[idx].rsample(torch.Size([count]))
        return samples

    def log_prob(self, value):
        log_probs = torch.stack([vmf.log_prob(value) for vmf in self.von_mises_fisher_distributions], dim=0)
        # adjust by mixture weight
        log_probs += torch.log(self.mixture_weights.unsqueeze(1))
        return torch.logsumexp(log_probs, dim=0)

    @property
    def locs(self):
        return torch.stack([vmf.loc for vmf in self.von_mises_fisher_distributions], dim=0)

    @property
    def scales(self):
        return torch.stack([vmf.scale for vmf in self.von_mises_fisher_distributions], dim=0)

    @property
    def mean(self):
        return sum([vmf.mean * w for w, vmf in zip(self.mixture_weights, self.von_mises_fisher_distributions)])

    @property
    def mixture_weights(self):
        return torch.softmax(self.mixture_weights_logits, dim=0)


class VonMisesFisherTrainable(nn.Module, VonMisesFisher):
    # disable constraint checking by pytorch
    arg_constraints = {}
    _arg_constraints = {"loc": torch.distributions.constraints.real, "scale": torch.distributions.constraints.positive}

    def __init__(self, manifold, loc, scale, validate_args=None, k=1, *args, **kwargs):
        nn.Module.__init__(self)

        self.dtype = loc.dtype
        self.loc_unprojected = loc
        self.manifold = manifold
        self._VonMisesFisher__m = loc.shape[-1]
        self.k = k

        Distribution.__init__(self, self.loc.size(), validate_args=False)
        if validate_args:
            self.validate_args()
        self.log_scale = nn.Parameter(torch.log(scale), requires_grad=scale.requires_grad)

    @property
    def device(self):
        return self.loc_unprojected.device

    @property
    def _VonMisesFisher__e1(self):
        e1 = torch.zeros_like(self.loc_unprojected)
        e1[..., 0] = 1
        return e1

    def validate_args(self):
        try:
            arg_constraints = self._arg_constraints
        except NotImplementedError:
            arg_constraints = {}
            warnings.warn(
                f"{self.__class__} does not define `arg_constraints`. "
                + "Please set `arg_constraints = {}` or initialize the distribution "
                + "with `validate_args=False` to turn off validation."
            )
        for param, constraint in arg_constraints.items():
            if constraints.is_dependent(constraint):
                continue  # skip constraints that cannot be checked
            if param not in self.__dict__ and isinstance(
                    getattr(type(self), param), lazy_property
            ):
                continue  # skip checking lazily-constructed args
            value = getattr(self, param)
            valid = constraint.check(value)
            if not valid.all():
                raise ValueError(
                    f"Expected parameter {param} "
                    f"({type(value).__name__} of shape {tuple(value.shape)}) "
                    f"of distribution {repr(self)} "
                    f"to satisfy the constraint {repr(constraint)}, "
                    f"but found invalid values:\n{value}"
                )

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    @property
    def loc(self):
        loc = self.manifold.projection(self.loc_unprojected)
        return loc

    def _log_normalization(self):
        scale = self.scale
        output = -(
                ((self._VonMisesFisher__m) / 2 - 1) * torch.log(scale)
                - ((self._VonMisesFisher__m) / 2) * math.log(2 * math.pi)
                - (scale + torch.log(ive((self._VonMisesFisher__m) / 2 - 1, scale)))
        )
        return output
