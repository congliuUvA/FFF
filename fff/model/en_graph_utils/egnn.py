from torch import nn
import torch
import math
import math
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
import torch

import functools
import itertools
import operator
from copy import deepcopy
import torch
from typing import Optional, Tuple
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, radius_graph
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

class CosineCutoff(nn.Module):
    
    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()
        
        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2)


class GaussianSmearing(nn.Module):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=True):
        super(GaussianSmearing, self).__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(0, self.cutoff, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


rbf_class_mapping = {"gauss": GaussianSmearing, "expnorm": ExpNormalSmearing}


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


act_class_mapping = {"ssp": ShiftedSoftplus, "silu": nn.SiLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "swish": Swish}


class Sphere(nn.Module):
    
    def __init__(self, l=2):
        super(Sphere, self).__init__()
        self.l = l
        
    def forward(self, edge_vec):
        edge_sh = self._spherical_harmonics(self.l, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2])
        return edge_sh
        
    @staticmethod
    def _spherical_harmonics(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

        sh_1_0, sh_1_1, sh_1_2 = x, y, z
        
        if lmax == 1:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2], dim=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], dim=-1)


class VecLayerNorm(nn.Module):
    def __init__(self, hidden_channels, trainable, norm_type="max_min"):
        super(VecLayerNorm, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.eps = 1e-12
        
        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)
        
        if norm_type == "rms":
            self.norm = self.rms_norm
        elif norm_type == "max_min":
            self.norm = self.max_min_norm
        else:
            self.norm = self.none_norm
        
        self.reset_parameters()

    def reset_parameters(self):
        weight = torch.ones(self.hidden_channels)
        self.weight.data.copy_(weight)
    
    def none_norm(self, vec):
        return vec
        
    def rms_norm(self, vec):
        # vec: (num_atoms, 3 or 5, hidden_channels)
        dist = torch.norm(vec, dim=1)
        
        if (dist == 0).all():
            return torch.zeros_like(vec)
        
        dist = dist.clamp(min=self.eps)
        dist = torch.sqrt(torch.mean(dist ** 2, dim=-1))
        return vec / F.relu(dist).unsqueeze(-1).unsqueeze(-1)
    
    def max_min_norm(self, vec):
        # vec: (num_atoms, 3 or 5, hidden_channels)
        dist = torch.norm(vec, dim=1, keepdim=True)
        
        if (dist == 0).all():
            return torch.zeros_like(vec)
        
        dist = dist.clamp(min=self.eps)
        direct = vec / dist
        
        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)
        
        return F.relu(dist) * direct

    def forward(self, vec):
        # vec: (num_atoms, 3 or 8, hidden_channels)
        if vec.shape[1] == 3:
            vec = self.norm(vec)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.shape[1] == 8:
            vec1, vec2 = torch.split(vec, [3, 5], dim=1)
            vec1 = self.norm(vec1)
            vec2 = self.norm(vec2)
            vec = torch.cat([vec1, vec2], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("VecLayerNorm only support 3 or 8 channels")


class Distance(nn.Module):
    def __init__(self, cutoff, max_num_neighbors=32, loop=True):
        super(Distance, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.loop = loop

    def forward(self, pos, edge_index, edge_mask=None):
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.loop:
            mask = edge_mask
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight = torch.norm(edge_vec, dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)
        if edge_mask is not None:
            edge_weight = edge_weight * edge_mask.squeeze()

        return edge_index, edge_weight, edge_vec


class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, cutoff, max_z=6):
        super(NeighborEmbedding, self).__init__(aggr="add")
        self.embedding = nn.Linear(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr, edge_mask=None, node_mask=None):
        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)
        x_neighbors = self.embedding(z)
        # propagate_type: (x: Tensor, W: Tensor, edge_mask: Tensor)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, edge_mask=edge_mask, size=None)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        if node_mask is not None:
            x_neighbors = x_neighbors * node_mask
        return x_neighbors

    def message(self, x_j, W, edge_mask):
        out = x_j * W
        if edge_mask is not None:
            out = out * edge_mask
        return out
    
class EdgeEmbedding(MessagePassing):
    
    def __init__(self, num_rbf, hidden_channels):
        super(EdgeEmbedding, self).__init__(aggr=None)
        self.edge_proj = nn.Linear(num_rbf, hidden_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_proj.weight)
        self.edge_proj.bias.data.fill_(0)
        
    def forward(self, edge_index, edge_attr, x, edge_mask=None, node_mask=None):
        # propagate_type: (x: Tensor, edge_attr: Tensor, edge_mask: Tensor, node_mask: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_mask=edge_mask, node_mask=node_mask)
        if node_mask is not None:
            out = out * node_mask
        return out
    
    def message(self, x_i, x_j, edge_attr, edge_mask):
        out = (x_i + x_j) * self.edge_proj(edge_attr)
        if edge_mask is not None:
            out = out * edge_mask
        return out
    
    def aggregate(self, features, index):
        # no aggregate
        return features
    
# copied from the itertools docs
def _powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


class ShortLexBasisBladeOrder:
    def __init__(self, n_vectors):
        self.index_to_bitmap = torch.empty(2**n_vectors, dtype=int)
        self.grades = torch.empty(2**n_vectors, dtype=int)
        self.bitmap_to_index = torch.empty(2**n_vectors, dtype=int)

        for i, t in enumerate(_powerset([1 << i for i in range(n_vectors)])):
            bitmap = functools.reduce(operator.or_, t, 0)
            self.index_to_bitmap[i] = bitmap
            self.grades[i] = len(t)
            self.bitmap_to_index[bitmap] = i
            del t  # enables an optimization inside itertools.combinations


def set_bit_indices(x: int):
    """Iterate over the indices of bits set to 1 in `x`, in ascending order"""
    n = 0
    while x > 0:
        if x & 1:
            yield n
        x = x >> 1
        n = n + 1


def count_set_bits(bitmap: int) -> int:
    """Counts the number of bits set to 1 in bitmap"""
    count = 0
    for i in set_bit_indices(bitmap):
        count += 1
    return count


def canonical_reordering_sign_euclidean(bitmap_a, bitmap_b):
    """
    Computes the sign for the product of bitmap_a and bitmap_b
    assuming a euclidean metric
    """
    a = bitmap_a >> 1
    sum_value = 0
    while a != 0:
        sum_value = sum_value + count_set_bits(a & bitmap_b)
        a = a >> 1
    if (sum_value & 1) == 0:
        return 1
    else:
        return -1


def canonical_reordering_sign(bitmap_a, bitmap_b, metric):
    """
    Computes the sign for the product of bitmap_a and bitmap_b
    given the supplied metric
    """
    bitmap = bitmap_a & bitmap_b
    output_sign = canonical_reordering_sign_euclidean(bitmap_a, bitmap_b)
    i = 0
    while bitmap != 0:
        if (bitmap & 1) != 0:
            output_sign *= metric[i]
        i = i + 1
        bitmap = bitmap >> 1
    return output_sign


def gmt_element(bitmap_a, bitmap_b, sig_array):
    """
    Element of the geometric multiplication table given blades a, b.
    The implementation used here is described in :cite:`ga4cs` chapter 19.
    """
    output_sign = canonical_reordering_sign(bitmap_a, bitmap_b, sig_array)
    output_bitmap = bitmap_a ^ bitmap_b
    return output_bitmap, output_sign


def construct_gmt(index_to_bitmap, bitmap_to_index, signature):
    n = len(index_to_bitmap)
    array_length = int(n * n)
    coords = torch.zeros((3, array_length), dtype=torch.int)
    k_list = coords[0, :]
    l_list = coords[1, :]
    m_list = coords[2, :]

    # use as small a type as possible to minimize type promotion
    mult_table_vals = torch.zeros(array_length)

    for i in range(n):
        bitmap_i = index_to_bitmap[i]

        for j in range(n):
            bitmap_j = index_to_bitmap[j]
            bitmap_v, mul = gmt_element(bitmap_i, bitmap_j, signature)
            v = bitmap_to_index[bitmap_v]

            list_ind = i * n + j
            k_list[list_ind] = i
            l_list[list_ind] = v
            m_list[list_ind] = j

            mult_table_vals[list_ind] = mul

    return torch.sparse_coo_tensor(
        indices=coords, values=mult_table_vals, size=(n, n, n)
    )

class CliffordAlgebra(nn.Module):
    def __init__(self, metric):
        super().__init__()

        self.register_buffer("metric", torch.as_tensor(metric))
        self.num_bases = len(metric)
        self.bbo = ShortLexBasisBladeOrder(self.num_bases)
        self.dim = len(self.metric)
        self.n_blades = len(self.bbo.grades)
        cayley = (
            construct_gmt(
                self.bbo.index_to_bitmap, self.bbo.bitmap_to_index, self.metric
            )
            .to_dense()
            .to(torch.get_default_dtype())
        )
        self.grades = self.bbo.grades.unique()
        self.register_buffer(
            "subspaces",
            torch.tensor(tuple(math.comb(self.dim, g) for g in self.grades)),
        )
        self.n_subspaces = len(self.grades)
        self.grade_to_slice = self._grade_to_slice(self.subspaces)
        self.grade_to_index = [
            torch.tensor(range(*s.indices(s.stop))) for s in self.grade_to_slice
        ]

        self.register_buffer(
            "bbo_grades", self.bbo.grades.to(torch.get_default_dtype())
        )
        self.register_buffer("even_grades", self.bbo_grades % 2 == 0)
        self.register_buffer("odd_grades", ~self.even_grades)
        self.register_buffer("cayley", cayley)

    def geometric_product(self, a, b, blades=None):
        cayley = self.cayley

        if blades is not None:
            blades_l, blades_o, blades_r = blades
            assert isinstance(blades_l, torch.Tensor)
            assert isinstance(blades_o, torch.Tensor)
            assert isinstance(blades_r, torch.Tensor)
            cayley = cayley[blades_l[:, None, None], blades_o[:, None], blades_r]

        return torch.einsum("...i,ijk,...k->...j", a, cayley, b)


    def _grade_to_slice(self, subspaces):
        grade_to_slice = list()
        subspaces = torch.as_tensor(subspaces)
        for grade in self.grades:
            index_start = subspaces[:grade].sum()
            index_end = index_start + math.comb(self.dim, grade)
            grade_to_slice.append(slice(index_start, index_end))
        return grade_to_slice

    @functools.cached_property
    def _alpha_signs(self):
        return torch.pow(-1, self.bbo_grades)

    @functools.cached_property
    def _beta_signs(self):
        return torch.pow(-1, self.bbo_grades * (self.bbo_grades - 1) / 2)

    @functools.cached_property
    def _gamma_signs(self):
        return torch.pow(-1, self.bbo_grades * (self.bbo_grades + 1) / 2)

    def alpha(self, mv, blades=None):
        signs = self._alpha_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv.clone()

    def beta(self, mv, blades=None):
        signs = self._beta_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv.clone()

    def gamma(self, mv, blades=None):
        signs = self._gamma_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv.clone()

    def zeta(self, mv):
        return mv[..., :1]

    def embed(self, tensor: torch.Tensor, tensor_index: torch.Tensor) -> torch.Tensor:
        mv = torch.zeros(
            *tensor.shape[:-1], 2**self.dim, device=tensor.device, dtype=tensor.dtype
        )
        mv[..., tensor_index] = tensor
        return mv

    # def embed_grade(self, tensor: torch.Tensor, grade: int) -> torch.Tensor:
    #     mv = torch.zeros(*tensor.shape[:-1], 2**self.dim, device=tensor.device)
    #     s = self.grade_to_slice[grade]
    #     mv[..., s] = tensor
    #     return mv

    def embed_grade(self, tensor: torch.Tensor, grade: int) -> torch.Tensor:
        if grade == 0:
            right_zeros = torch.zeros(*tensor.shape[:-1], 2**self.dim-1, device=tensor.device)
            mv = torch.cat((tensor, right_zeros), dim=-1)
        if grade == 1:
            left_zeros = torch.zeros(*tensor.shape[:-1], 1, device=tensor.device)
            right_zeros = torch.zeros(*tensor.shape[:-1], 4, device=tensor.device)
            mv = torch.cat((left_zeros, tensor, right_zeros), dim=-1)
        return mv


    def get(self, mv: torch.Tensor, blade_index: tuple[int]) -> torch.Tensor:
        blade_index = tuple(blade_index)
        return mv[..., blade_index]

    def get_grade(self, mv: torch.Tensor, grade: int) -> torch.Tensor:
        s = self.grade_to_slice[grade]
        return mv[..., s]

    def b(self, x, y, blades=None):
        if blades is not None:
            assert len(blades) == 2
            beta_blades = blades[0]
            blades = (
                blades[0],
                torch.tensor([0]),
                blades[1],
            )
        else:
            blades = torch.tensor(range(self.n_blades))
            blades = (
                blades,
                torch.tensor([0]),
                blades,
            )
            beta_blades = None

        return self.geometric_product(
            self.beta(x, blades=beta_blades),
            y,
            blades=blades,
        )

    def q(self, mv, blades=None):
        if blades is not None:
            blades = (blades, blades)
        return self.b(mv, mv, blades=blades)

    def _smooth_abs_sqrt(self, input, eps=1e-16):
        return (input**2 + eps) ** 0.25

    def norm(self, mv, blades=None):
        return self._smooth_abs_sqrt(self.q(mv, blades=blades))

    def norms(self, mv, grades=None):
        if grades is None:
            grades = self.grades
        return [
            self.norm(self.get_grade(mv, grade), blades=self.grade_to_index[grade])
            for grade in grades
        ]

    def qs(self, mv, grades=None):
        if grades is None:
            grades = self.grades
        return [
            self.q(self.get_grade(mv, grade), blades=self.grade_to_index[grade])
            for grade in grades
        ]

    def sandwich(self, u, v, w):
        return self.geometric_product(self.geometric_product(u, v), w)

    def output_blades(self, blades_left, blades_right):
        blades = []
        for blade_left in blades_left:
            for blade_right in blades_right:
                bitmap_left = self.bbo.index_to_bitmap[blade_left]
                bitmap_right = self.bbo.index_to_bitmap[blade_right]
                bitmap_out, _ = gmt_element(bitmap_left, bitmap_right, self.metric)
                index_out = self.bbo.bitmap_to_index[bitmap_out]
                blades.append(index_out)

        return torch.tensor(blades)

    def random(self, n=None):
        if n is None:
            n = 1
        return torch.randn(n, self.n_blades)

    def random_vector(self, n=None):
        if n is None:
            n = 1
        vector_indices = self.bbo_grades == 1
        v = torch.zeros(n, self.n_blades, device=self.cayley.device)
        v[:, vector_indices] = torch.randn(
            n, vector_indices.sum(), device=self.cayley.device
        )
        return v

    def parity(self, mv):
        is_odd = torch.all(mv[..., self.even_grades] == 0)
        is_even = torch.all(mv[..., self.odd_grades] == 0)

        if is_odd ^ is_even:  # exclusive or (xor)
            return is_odd
        else:
            raise ValueError("This is not a homogeneous element.")

    def eta(self, w):
        return (-1) ** self.parity(w)

    def alpha_w(self, w, mv):
        return self.even_grades * mv + self.eta(w) * self.odd_grades * mv

    def inverse(self, mv, blades=None):
        mv_ = self.beta(mv, blades=blades)
        return mv_ / self.b(mv, mv_)

    def rho(self, w, mv):
        """Applies the versor w action to mv."""
        return self.sandwich(w, self.alpha_w(w, mv), self.inverse(w))

    def reduce_geometric_product(self, inputs):
        return functools.reduce(self.geometric_product, inputs)

    def versor(self, order=None, normalized=True):
        if order is None:
            order = self.dim if self.dim % 2 == 0 else self.dim - 1
        vectors = self.random_vector(order)
        versor = self.reduce_geometric_product(vectors[:, None])
        if normalized:
            versor = versor / self.norm(versor)[..., :1]
        return versor

    def rotor(self):
        return self.versor()

    @functools.cached_property
    def geometric_product_paths(self):
        gp_paths = torch.zeros((self.dim + 1, self.dim + 1, self.dim + 1), dtype=bool)

        for i in range(self.dim + 1):
            for j in range(self.dim + 1):
                for k in range(self.dim + 1):
                    s_i = self.grade_to_slice[i]
                    s_j = self.grade_to_slice[j]
                    s_k = self.grade_to_slice[k]

                    m = self.cayley[s_i, s_j, s_k]
                    gp_paths[i, j, k] = (m != 0).any()

        return gp_paths


def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor, dim=0):
    """
    Unsqueeze last dimensions of tensor to match another tensor's number of dimensions.

    Args:
        tensor (torch.Tensor): tensor to unsqueeze
        like (torch.Tensor): tensor whose dimensions to match
        dim: int: starting dim, default: 0.
    """
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    else:
        return tensor[dim * (slice(None),) + (None,) * n_unsqueezes]

        
def cl_flatten(h):
    batch_size = h.shape[0]
    return h.reshape(batch_size, -1)

def cl_split(h, algebra=CliffordAlgebra((1, 1, 1))):
    num_bases = 2 ** algebra.dim
    batch_size = h.shape[0]
    return h.reshape(batch_size, -1, num_bases)

EPS = 1e-6


class MVLayerNorm(nn.Module):
    def __init__(self, algebra, channels):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.a = nn.Parameter(torch.ones(1, channels))

    def forward(self, input):
        norm = self.algebra.norm(input)[..., :1].mean(dim=1, keepdim=True) + EPS
        a = unsqueeze_like(self.a, norm, dim=2)
        return a * input / norm


class EGCL(MessagePassing):
    def __init__(
        self,
        algebra,
        in_features,
        hidden_features,
        out_features,
        edge_attr_features=0,
        node_attr_features=0,
        residual=True,
        normalization_init=0,
        aggr="mean",
    ):
        super().__init__(aggr=aggr)
        self.residual = residual
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.edge_attr_features = edge_attr_features
        self.node_attr_features = node_attr_features

        self.edge_model = CEMLP(
            algebra,
            self.in_features + self.edge_attr_features,
            self.hidden_features,
            self.out_features,
            normalization_init=normalization_init,
        )

        self.node_model = CEMLP(
            algebra,
            self.in_features + self.out_features + node_attr_features,
            self.hidden_features,
            self.out_features,
            normalization_init=normalization_init,
        )
        self.algebra = algebra
        # self.norm = MVLayerNorm(self.algebra, self.hidden_features)
        

    def message(self, h_i, h_j, edge_attr=None, algebra=CliffordAlgebra((1.0, 1.0, 1.0)), edge_mask=None):
        h_i, h_j = cl_split(h_i, algebra=algebra), cl_split(h_j, algebra=algebra)
        if edge_attr is None:  # Unused.
            input = h_i - h_j
        else:
            input = torch.cat([h_i - h_j, edge_attr], dim=1)
        h_msg = self.edge_model(input)
        h_msg = cl_flatten(h_msg)
        if edge_mask is not None:
            h_msg = h_msg * edge_mask
        return h_msg

    def update(self, h_agg, h, node_attr, algebra=CliffordAlgebra((1.0, 1.0, 1.0)), node_mask=None):
        h_agg, h = cl_split(h_agg, algebra=algebra), cl_split(h, algebra=algebra)
        if node_attr is not None:
            input_h = torch.cat([h, h_agg, node_attr], dim=1)
        else:
            input_h = torch.cat([h, h_agg], dim=1)
        out_h = self.node_model(input_h)

        if self.residual:
            out_h = h + out_h
        # out_h = self.norm(out_h)
        out_h = cl_flatten(out_h)

        if node_mask is not None:
            out_h = out_h * node_mask
        return out_h
    
    def forward(self, h, edge_index, edge_attr=None, node_attr=None, algebra=CliffordAlgebra((1.0, 1.0, 1.0)), edge_mask=None, node_mask=None):
        h = cl_flatten(h)
        x = self.propagate(h=h, edge_index=edge_index, edge_attr=edge_attr, node_attr=node_attr, algebra=algebra, node_mask=node_mask, edge_mask=edge_mask)
        x = cl_split(x, algebra=algebra)

        return x
    

class CliffordNormalization(nn.Module):
    def __init__(self, algebra):
        super().__init__()
        self.algebra = algebra

    def forward(self, input):
        return input / self.algebra.norm(input)
    

class MVSiLU(nn.Module):
    def __init__(self, algebra, channels, invariant="mag2", exclude_dual=False):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.exclude_dual = exclude_dual
        self.invariant = invariant
        self.a = nn.Parameter(torch.ones(1, channels, algebra.dim + 1))
        self.b = nn.Parameter(torch.zeros(1, channels, algebra.dim + 1))

        if invariant == "norm":
            self._get_invariants = self._norms_except_scalar
        elif invariant == "mag2":
            self._get_invariants = self._mag2s_except_scalar
        else:
            raise ValueError(f"Invariant {invariant} not recognized.")

    def _norms_except_scalar(self, input):
        return self.algebra.norms(input, grades=self.algebra.grades[1:])

    def _mag2s_except_scalar(self, input):
        return self.algebra.qs(input, grades=self.algebra.grades[1:])

    def forward(self, input):
        norms = self._get_invariants(input)
        norms = torch.cat([input[..., :1], *norms], dim=-1)
        a = unsqueeze_like(self.a, norms, dim=2)
        b = unsqueeze_like(self.b, norms, dim=2)
        norms = a * norms + b
        norms = norms.repeat_interleave(self.algebra.subspaces, dim=-1)
        return torch.sigmoid(norms) * input
    
EPS = 1e-6


class NormalizationLayer(nn.Module):
    def __init__(self, algebra, features, init: float = 0):
        super().__init__()
        self.algebra = algebra
        self.in_features = features

        self.a = nn.Parameter(torch.zeros(self.in_features, algebra.n_subspaces) + init)

    def forward(self, input):
        assert input.shape[1] == self.in_features

        norms = torch.cat(self.algebra.norms(input), dim=-1)
        s_a = torch.sigmoid(self.a)
        norms = s_a * (norms - 1) + 1  # Interpolates between 1 and the norm.
        norms = norms.repeat_interleave(self.algebra.subspaces, dim=-1)
        normalized = input / (norms + EPS)

        return normalized
    
class SteerableGeometricProductLayer(nn.Module):
    def __init__(
        self, algebra, features, weight=None, include_first_order=True, normalization_init=0
    ):
        super().__init__()

        self.algebra = algebra
        self.features = features
        self.include_first_order = include_first_order

        if normalization_init is not None:
            self.normalization = NormalizationLayer(
                algebra, features, normalization_init
            )
        else:
            self.normalization = nn.Identity()
        self.linear_right = MVLinear(algebra, features, features, bias=False)
        if include_first_order:
            self.linear_left = MVLinear(algebra, features, features, bias=True)

        self.product_paths = algebra.geometric_product_paths
        self.weight = nn.Parameter(torch.empty(features, self.product_paths.sum())) if weight is None else weight

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=1 / (math.sqrt(self.algebra.dim + 1)))

    def _get_weight(self):
        weight = torch.zeros(
            self.features,
            *self.product_paths.size(),
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        weight[:, self.product_paths] = self.weight
        subspaces = self.algebra.subspaces
        weight_repeated = (
            weight.repeat_interleave(subspaces, dim=-3)
            .repeat_interleave(subspaces, dim=-2)
            .repeat_interleave(subspaces, dim=-1)
        )
        return self.algebra.cayley * weight_repeated

    def forward(self, input):
        input_right = self.linear_right(input)
        input_right = self.normalization(input_right)

        weight = self._get_weight()

        if self.include_first_order:
            return (
                self.linear_left(input)
                + torch.einsum("bni, nijk, bnk -> bnj", input, weight, input_right)
            ) / math.sqrt(2)

        else:
            return torch.einsum("bni, nijk, bnk -> bnj", input, weight, input_right)

    
class MVLinear(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        out_features,
        subspaces=True,
        bias=True,
    ):
        super().__init__()

        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features
        self.subspaces = subspaces

        if subspaces:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, algebra.n_subspaces)
            )
            self._forward = self._forward_subspaces
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(1, out_features, 1))
            self.b_dims = (0,)
        else:
            self.register_parameter("bias", None)
            self.b_dims = ()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def _forward(self, input):
        return torch.einsum("bm...i, nm->bn...i", input, self.weight)

    def _forward_subspaces(self, input):
        weight = self.weight.repeat_interleave(self.algebra.subspaces, dim=-1)
        return torch.einsum("bm...i, nmi->bn...i", input, weight)

    def forward(self, input):
        result = self._forward(input)

        if self.bias is not None:
            bias = self.algebra.embed(self.bias, self.b_dims)
            result += unsqueeze_like(bias, result, dim=2)
        return result
    
class CEMLP(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        hidden_features,
        out_features,
        n_layers=2,
        normalization_init=0,
        residual=False
    ):
        super().__init__()
        self.algebra = algebra
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.n_layers = n_layers
        self.residual = residual

        layers = []

        # Add geometric product layers.
        for i in range(n_layers - 1):
            layers.append(
                nn.Sequential(
                    MVLinear(self.algebra, in_features, hidden_features),
                    MVSiLU(self.algebra, hidden_features),
                    SteerableGeometricProductLayer(
                        self.algebra,
                        hidden_features,
                        normalization_init=normalization_init,
                    ),
                    MVLayerNorm(self.algebra, hidden_features),
                )
            )
            in_features = hidden_features

        # Add final layer.
        layers.append(
            nn.Sequential(
                MVLinear(self.algebra, in_features, out_features),
                MVSiLU(self.algebra, out_features),
                SteerableGeometricProductLayer(
                    self.algebra,
                    out_features,
                    normalization_init=normalization_init,
                ),
                MVLayerNorm(self.algebra, out_features),
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            if not self.residual:
                x = layer(x)
            else:
                x_out = layer(x)
                x = x + x_out
        return x

class CustomCliffordNormalization(nn.Module):
    def __init__(self, algebra, features, init: float = 0):
        super().__init__()
        self.algebra = algebra
        self.in_features = features

        self.a = nn.Parameter(torch.zeros(self.in_features, algebra.n_subspaces) + init)

    def forward(self, input):
        assert input.shape[1] == self.in_features

        norms = torch.cat(self.algebra.norms(input), dim=-1)
        s_a = torch.sigmoid(self.a)
        norms = s_a * (norms - 1) + 1  # Interpolates between 1 and the norm.
        norms = norms.repeat_interleave(self.algebra.subspaces, dim=-1)
        normalized = input / norms

        return normalized
    

class MVGeometricProduct(nn.Module):
    def __init__(self, algebra, features, normalization_init=0):
        super().__init__()

        self.algebra = algebra
        self.features = features
        self.normalization_init = normalization_init

        self.product_paths = algebra.geometric_product_paths
        self.weight = nn.Parameter(torch.empty(features, self.product_paths.sum()))

        self.normalization = self.setup_normalization(normalization_init)

        self.reset_parameters()

    def setup_normalization(self, init):
        if init is None:
            return nn.Identity()
        elif init == "norm":
            return CliffordNormalization(self.algebra)
        else:
            return CustomCliffordNormalization(self.algebra, self.features, init)

    def reset_parameters(self):
        torch.nn.init.normal_(
            self.weight, std=1 / (math.sqrt(2) * math.sqrt(self.algebra.dim + 1))
        )

    def _get_weight(self):
        weight = torch.zeros(
            self.features,
            *self.product_paths.size(),
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        weight[:, self.product_paths] = self.weight
        subspaces = self.algebra.subspaces
        weight_repeated = (
            weight.repeat_interleave(subspaces, dim=-3)
            .repeat_interleave(subspaces, dim=-2)
            .repeat_interleave(subspaces, dim=-1)
        )
        return self.algebra.cayley * weight_repeated

    def forward(self, v, w):
        weight = self._get_weight()
        w = self.normalization(w)
        return torch.einsum("bni, nijk, bnk -> bnj", v, weight, w)


class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord
    
class CliffordEquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, hidden_nvf=8):
        super(CliffordEquivariantUpdate, self).__init__()
        self.algebra = CliffordAlgebra((1,1,1))
        input_edge = hidden_nf * 2 + edges_in_d
        self.pos_embedding = MVLinear(self.algebra, 1, hidden_nvf, subspaces=False)
        self.pos_edge_model = CEMLP(self.algebra, hidden_nvf, hidden_nvf, hidden_nvf)
        self.pos_update_model = MVLinear(self.algebra, hidden_nvf, hidden_nvf)
        self.coord_mlp = self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nvf))
        self.h_update_mlp = nn.Sequential(
            nn.Linear(hidden_nvf + hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf))
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, edge_attr, edge_mask):
        row, col = edge_index
        input_cov_tensor = coord[row] - coord[col]
        input_inv_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        edge_cov_message = self.pos_edge_model(input_cov_tensor)
        edge_inv_message = self.coord_mlp(input_inv_tensor)
        
        if edge_mask is not None:
            edge_inv_message = edge_inv_message * edge_mask
            edge_cov_message = cl_flatten(edge_cov_message) * edge_mask
        agg_inv = unsorted_segment_sum(edge_inv_message, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        agg_cov = unsorted_segment_sum(edge_cov_message, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        agg_cov = cl_split(agg_cov)
        agg_cov[:, :, 0] = agg_cov[:, :, 0] + agg_inv
        h_update = self.h_update_mlp(torch.cat((agg_cov[..., 0], h), dim=-1))
        coord = coord + self.pos_update_model(agg_cov)
        
        return h_update, coord

    def forward(self, h, coord, edge_index, edge_attr=None, node_mask=None, edge_mask=None):
        h, coord = self.coord_model(h, coord, edge_index, edge_attr, edge_mask)
        if node_mask is not None:
            coord = cl_split(cl_flatten(coord) * node_mask)
            h = h * node_mask
        return h, coord

class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask,
                                               edge_mask=edge_mask)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x

class CliffordEquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, hidden_nvf=8, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        super(CliffordEquivariantBlock, self).__init__()
        self.algebra = CliffordAlgebra((1,1,1))
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", CliffordEquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, hidden_nvf=hidden_nvf,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))
        self.to(self.device)

    def forward(self, h, pos, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        row, col = edge_index
        distances = self.algebra.norm(pos[row] - pos[col]).squeeze()
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask,
                                               edge_mask=edge_mask)
        _, pos = self._modules["gcl_equiv"](h, pos, edge_index, edge_attr, node_mask, edge_mask)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, pos


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range / n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(
                hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                act_fn=act_fn, n_layers=inv_sublayers,
                attention=attention, norm_diff=norm_diff, tanh=tanh,
                coords_range=coords_range, norm_constant=norm_constant,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method
            ))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask,
                                                   edge_attr=distances)

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x

class CEGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, hidden_nvf=8, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super(CEGNN, self).__init__()
        self.algebra = CliffordAlgebra((1,1,1))
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range / n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 1 + hidden_nvf
        self.pos_embedding = MVLinear(self.algebra, in_features=1, out_features=hidden_nvf, subspaces=False)
        self.pos_embedding_out = MVLinear(self.algebra, in_features=hidden_nvf, out_features=1)
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, CliffordEquivariantBlock(
                hidden_nf, hidden_nvf=hidden_nvf, edge_feat_nf=edge_feat_nf, device=device,
                act_fn=act_fn, n_layers=inv_sublayers,
                attention=attention, norm_diff=norm_diff, tanh=tanh,
                coords_range=coords_range, norm_constant=norm_constant,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method
            ))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None):
        x_mean = x.mean(dim=0, keepdim=True)
        x = x - x_mean
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        h = self.embedding(h)
        pos = self.algebra.embed_grade(x.unsqueeze(1), 1)
        pos = self.pos_embedding(pos)
        for i in range(0, self.n_layers):
            h, pos = self._modules["e_block_%d" % i](h, pos, edge_index, node_mask=node_mask, edge_mask=edge_mask,
                                                   edge_attr=distances)

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        x = pos[..., 0, 1:4] + x_mean
        return h, x


class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 act_fn=nn.SiLU(), n_layers=4, attention=False,
                 normalization_factor=1, out_node_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, act_fn=act_fn,
                attention=attention))
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies) / max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        # norm[norm == 0] = 1
        result = result / norm
    return result


class CliffordMACE(nn.Module):

    def __init__(
        self,
        in_node_nf, 
        in_edge_nf, 
        hidden_nf, 
        device='cpu', 
        act_fn=nn.SiLU(), 
        n_layers=3, 
        attention=False,
        norm_diff=True, 
        out_node_nf=None, 
        tanh=False, 
        coords_range=15, 
        norm_constant=1, 
        inv_sublayers=2,
        sin_embedding=False, 
        normalization_factor=100, 
        aggregation_method='sum',
        num_heads=8,
        num_layers=4,
        scalar_hidden_channels=32,
        vec_hidden_channels=32,
        num_rbf=32,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        attn_activation="silu",
        max_z=6,
        cutoff=5.0,
        max_num_neighbors=32,
        num_mace_basis=12,
        num_mace_prods=4,
        mean=None,
        std=None, 
        energy_weight=0.05,
        force_weight=0.95,
    ):
        super(CliffordMACE, self).__init__()
        self.algebra = CliffordAlgebra((1,1,1))
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.scalar_hidden_channels = scalar_hidden_channels
        self.vec_hidden_channels = vec_hidden_channels
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.max_z = max_z
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
    
        self.embedding = nn.Linear(in_node_nf, scalar_hidden_channels)
        self.distance = Distance(cutoff, max_num_neighbors=max_num_neighbors, loop=True)
        self.distance_expansion = rbf_class_mapping[rbf_type](cutoff, num_rbf, trainable_rbf)
        self.neighbor_embedding = NeighborEmbedding(scalar_hidden_channels, num_rbf, cutoff, max_z).jittable()
        self.edge_embedding = EdgeEmbedding(num_rbf, scalar_hidden_channels).jittable()

        self.edge_clifford_embedding = MVLinear(self.algebra, in_features=1, out_features=vec_hidden_channels, subspaces=False)

        self.vis_mp_layers = nn.ModuleList()
        vis_mp_kwargs = dict(
            num_heads=num_heads, 
            scalar_hidden_channels=scalar_hidden_channels, 
            vec_hidden_channels=vec_hidden_channels,
            activation=activation, 
            attn_activation=attn_activation, 
            cutoff=cutoff, 
        )
        vis_mp_class = CliffordMACE_MP

        for _ in range(num_layers - 1):
            layer = vis_mp_class(last_layer=False, **vis_mp_kwargs).jittable()
            self.vis_mp_layers.append(layer)
        self.vis_mp_layers.append(vis_mp_class(last_layer=True, **vis_mp_kwargs).jittable())

        self.out_norm = nn.LayerNorm(scalar_hidden_channels)
        self.vec_out_norm = MVLayerNorm(self.algebra, channels=vec_hidden_channels)
        self.vec_out = MVLinear(self.algebra, in_features=vec_hidden_channels, out_features=1)
        self.scalar_out = nn.Sequential(
                nn.Linear(scalar_hidden_channels, max_z)
        )
        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        for layer in self.vis_mp_layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()
        
    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None):
        pos = x
        pos_mean = pos.mean(dim=0, keepdim=True)
        pos = pos - pos_mean
        edge_index = torch.stack(edge_index, dim=0)
        # Embedding Layers
        x = self.embedding(h)
        edge_index, edge_weight, edge_vec = self.distance(pos, edge_index, edge_mask) # edge_weights are the distances
        edge_attr = self.distance_expansion(edge_weight)
        # edge_vec = edge_vec / torch.norm(edge_vec, dim=1).unsqueeze(1)  # normalization
        # edge_vec =  torch.nan_to_num(edge_vec)
        if edge_mask is not None:
            edge_vec = edge_vec * edge_mask
        edge_vec = self.algebra.embed_grade(edge_vec.unsqueeze(1), 1)
        edge_vec = self.edge_clifford_embedding(edge_vec)
        vec = torch.zeros(x.size(0), self.vec_hidden_channels, 2**self.algebra.dim,  device=x.device) # it will add up with spherical harmonical function values of edges
        x = self.neighbor_embedding(h, x, edge_index, edge_weight, edge_attr, edge_mask=edge_mask, node_mask=node_mask) # message passing with edge weights only on atomic type on nodes
        edge_attr = self.edge_embedding(edge_index, edge_attr, x) # message passing with edge weights only on atomic type on edges
        
        i = 0
        # ViS-MP Layers
        for attn in self.vis_mp_layers[:-1]:
            dx, dvec, dedge_attr, edge_vec = attn(x, vec, edge_index, edge_weight, edge_attr, edge_vec, edge_mask=edge_mask, node_mask=node_mask)
            x = x + dx
            vec = vec + dvec
            edge_attr = dedge_attr + edge_attr

        dx, dvec, _, _ = self.vis_mp_layers[-1](x, vec, edge_index, edge_weight, edge_attr, edge_vec, edge_mask=edge_mask, node_mask=node_mask)
        x = x + dx
        vec = vec + dvec
        x = self.scalar_out(x)
        vec = self.vec_out(vec)
        vec = cl_split(vec)[..., 0, 1:4]
        if node_mask is not None:
            x = x * node_mask
            vec = vec + pos_mean
            vec = vec * node_mask
        return x, vec
    

class CliffordMACE_MP(MessagePassing):
    def __init__(
        self,
        num_heads,
        scalar_hidden_channels,
        vec_hidden_channels,
        activation,
        attn_activation,
        cutoff,
        last_layer=False,
    ):
        super(CliffordMACE_MP, self).__init__(aggr="mean", node_dim=0)
        assert scalar_hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({scalar_hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )
        self.algebra = CliffordAlgebra((1,1,1))
        self.num_heads = num_heads
        self.scalar_hidden_channels = scalar_hidden_channels
        self.vec_hidden_channels = vec_hidden_channels
        self.head_dim = scalar_hidden_channels // num_heads
        self.last_layer = last_layer
        
        self.layernorm = nn.LayerNorm(scalar_hidden_channels)
        self.vec_layernorm = MVLayerNorm(self.algebra, vec_hidden_channels)
        
        self.s_act = act_class_mapping[activation]()
        # self.v_act = MVSiLU(self.algebra, channels=2*vec_hidden_channels)
        self.attn_activation = act_class_mapping[attn_activation]()

        self.cutoff = CosineCutoff(cutoff)
        self.vec_proj = MVLinear(self.algebra, vec_hidden_channels, vec_hidden_channels * 3)

        self.q_proj = nn.Linear(scalar_hidden_channels, scalar_hidden_channels)
        self.k_proj = nn.Linear(scalar_hidden_channels, scalar_hidden_channels)
        self.v_proj = nn.Linear(scalar_hidden_channels, scalar_hidden_channels)
        self.dk_proj = nn.Linear(scalar_hidden_channels, scalar_hidden_channels)
        self.dv_proj = nn.Linear(scalar_hidden_channels, scalar_hidden_channels)

        self.s_proj = nn.Linear(scalar_hidden_channels, vec_hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = nn.Linear(scalar_hidden_channels, scalar_hidden_channels)
            self.v2s_3 = nn.Linear(vec_hidden_channels, scalar_hidden_channels, bias=False) if self.scalar_hidden_channels != self.vec_hidden_channels else nn.Identity()
            self.node_gp_layer = MVGeometricProduct(self.algebra, vec_hidden_channels)
            self.edge_gp_layer = CEMLP(self.algebra, vec_hidden_channels, vec_hidden_channels, vec_hidden_channels, n_layers=1)


        self.o_proj = nn.Linear(scalar_hidden_channels, scalar_hidden_channels * 3)
        self.reset_parameters()
       
        
        self.v2s_1 = nn.Linear(vec_hidden_channels, scalar_hidden_channels, bias=False) if self.scalar_hidden_channels != self.vec_hidden_channels else nn.Identity()
        self.v2s_2 = nn.Linear(vec_hidden_channels, scalar_hidden_channels, bias=False) if self.scalar_hidden_channels != self.vec_hidden_channels else nn.Identity()
        
        self.s2v_1 = nn.Linear(scalar_hidden_channels, vec_hidden_channels, bias=False) if self.scalar_hidden_channels != self.vec_hidden_channels else nn.Identity()

        
    
    def reset_parameters(self):
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.s_proj.weight)
        self.s_proj.bias.data.fill_(0)
        
        if not self.last_layer:
            nn.init.xavier_uniform_(self.f_proj.weight)
            self.f_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.dk_proj.weight)
        self.dk_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dv_proj.weight)
        self.dv_proj.bias.data.fill_(0)

        
    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij, edge_mask, node_mask):
        # x: node type embedding after message passing (n_nodes, hidden)
        # vec: node vector (n_nodes, 2**(l_max+1), hidden)
        # r_ij: rbf kernel
        # f_ij: edge embedding after message passing
        # d_ij: spherical harmonics of normalized edge vector
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)
        
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.s_act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim) # edge key
        dv = self.s_act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim) # edge value
        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.vec_hidden_channels, dim=1)
        vec_dot = self.algebra.geometric_product(vec1, vec2)[..., 0].squeeze()
        vec_dot = self.v2s_1(vec_dot)
        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, dk: Tensor, dv: Tensor, vec: Tensor, r_ij: Tensor, d_ij: Tensor, edge_mask: Tensor, node_mask: Tensor)
        
        output = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            dk=dk,
            dv=dv,
            vec=vec,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
            edge_mask=edge_mask,
            node_mask=node_mask,
        )
        if not self.last_layer:
            x, vec_out, edge_vec = output
            if node_mask is not None:
                x = x * node_mask
                vec_out = cl_split(cl_flatten(vec_out) * node_mask)
                edge_vec = cl_split(cl_flatten(edge_vec) * node_mask)
        else:
            x, vec_out = output
            if node_mask is not None:
                x = x * node_mask
                vec_out = cl_split(cl_flatten(vec_out) * node_mask)
        # scalar-vector interactions
        
        o1, o2, o3 = torch.split(self.o_proj(x), self.scalar_hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * self.s2v_1(o1).unsqueeze(-1) + vec_out
        if not self.last_layer:
            # edge_updater_type: (edge_vec: Tensor, d_ij: Tensor, f_ij: Tensor)
            df_ij, d_ij = self.edge_updater(edge_index, edge_vec=edge_vec, d_ij=d_ij, f_ij=f_ij)
            return dx, dvec, df_ij, d_ij
        else:
            return dx, dvec, None, None

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij, edge_mask=None):
        # use attention to calculate the node embeddings, using derived node embeddings to weighted sum with spherical edges
        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)
        
        v_j = v_j * dv
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.scalar_hidden_channels)

        s1, s2 = torch.split(self.s_act(self.s_proj(v_j)), self.vec_hidden_channels, dim=1)
        # use edge vector to update node vector
        vec_j = vec_j * s1.unsqueeze(-1) + s2.unsqueeze(-1) * d_ij
        if edge_mask is not None:
            v_j = v_j * edge_mask
            vec_j = cl_split(cl_flatten(vec_j) * edge_mask)
            d_ij = cl_split(cl_flatten(d_ij) * edge_mask)
        return v_j, vec_j, d_ij
    
    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec, d_ij = features
        x = global_mean_pool(x, index)
        vec = cl_split(global_mean_pool(cl_flatten(vec), index))
        edge_vec = cl_split(global_mean_pool(cl_flatten(d_ij), index))

        return x, vec, edge_vec

    def update(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec, edge_vec = inputs
        
        if not self.last_layer:
            edge_vec = self.edge_gp_layer(edge_vec)
            return x, vec, edge_vec
        else:
            return x, vec

    def edge_update(self, edge_vec_i, edge_vec_j, f_ij, d_ij):
        # # dihedral angles
        vec_features = self.node_gp_layer(edge_vec_i, edge_vec_j)
        w_dot = vec_features[..., 0].squeeze()
        w_dot = self.v2s_3(w_dot)
        # use angles as weights to update edge embedding
        df_ij = self.s_act(self.f_proj(f_ij)) * w_dot 
        return df_ij, vec_features
