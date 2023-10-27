from collections import defaultdict, namedtuple
from copy import deepcopy
from importlib import import_module
from math import log10

import torch
from FrEIA.utils import sum_except_batch
from lightning_trainable.trainable import SkipBatch
from torch.nn import Sequential

from fff.loss import nll_surrogate
from fff.model.base import BaseModelHParams, BaseModel
from fff.other_losses.exact_jac_det import log_det_exact
from fff.other_losses.mmd import maximum_mean_discrepancy

LogProbResult = namedtuple("LogProbResult", ["z", "x1", "log_prob", "regularizations"])
ConditionedBatch = namedtuple("ConditionedBatch", [
    "x0", "x_noisy", "loss_weights", "condition", "dequantization_jac"
])


class FreeFormFlowHParams(BaseModelHParams):
    models: list

    loss_weights: dict
    log_det_estimator: dict = dict(
        name="surrogate",
        hutchinson_samples=1
    )
    skip_val_nll: bool | int = False
    exact_train_nll_every: int | None = None

    warm_up_epochs: int | list = 0

    exact_chunk_size: None | int = None

    def __init__(self, **hparams):
        if "models" not in hparams:
            if "latent_inn_spec" in hparams:
                model_class = "fff.model.LatentFlow"
                copy_keys = ["layer_spec", "latent_inn_spec", "detached_inn"]
            elif "layers_spec" in hparams:
                model_class = "fff.model.ResNet"
                copy_keys = ["layers_spec", "latent_spec", "activation"]
            elif "zero_init" in hparams:
                model_class = "fff.model.SurFlow"
                copy_keys = ["inn_spec", "zero_init"]
            elif "latent_layer_spec" in hparams:
                model_class = "fff.model.AutoEncoder"
                copy_keys = [
                    "layer_spec",
                    "latent_layer_spec",
                    "skip_connection",
                    "detached_latent"
                ]
            elif "ch_factor" in hparams:
                model_class = "fff.model.ConvAutoEncoder"
                copy_keys = [
                    "skip_connection",
                    "ch_factor",
                    "encoder_spec",
                    "encoder_fc_spec",
                    "decoder_fc_spec",
                    "decoder_spec",
                    "batch_norm",
                ]
            else:
                model_class = copy_keys = None
            if model_class is not None:
                hparams["models"] = [{
                    key: hparams.pop(key)
                    for key in copy_keys + ["latent_dim"]
                    if key in hparams
                }]
                hparams["models"][0]["name"] = model_class

        if "log_det_estimator" in hparams and isinstance(hparams["log_det_estimator"], str):
            hparams["log_det_estimator"] = dict(
                log_det_estimator=hparams["log_det_estimator"],
                trace_space=hparams.pop("trace_space", "latent"),
                grad_to_enc_or_dec=hparams.pop("grad_to_enc_or_dec"),
                grad_type=hparams.pop("grad_type"),
                hutchinson_samples=hparams.pop("hutchinson_samples", 1),
            )
            if "detach_non_grad" in hparams:
                hparams["log_det_estimator"]["detach_non_grad"] = hparams.pop("detach_non_grad")
        if "detached_decoder" in hparams and not hparams["detached_decoder"]:
            del hparams["detached_decoder"]

        super().__init__(**hparams)


class FreeFormFlow(BaseModel):
    """
    A FreeFormFlow is a normalizing flow consisting of a pair of free-form
    encoder and decoder.
    """
    hparams: FreeFormFlowHParams

    def __init__(self, hparams: FreeFormFlowHParams | dict):
        if not isinstance(hparams, FreeFormFlowHParams):
            hparams = FreeFormFlowHParams(**hparams)

        super().__init__(hparams)
        self.models = build_model(self.hparams.models, self.data_dim, self.cond_dim)

    def encode(self, x, c):
        for model in self.models:
            x = model.encode(x, c)
        return x

    def decode(self, z, c):
        for model in self.models[::-1]:
            z = model.decode(z, c)
        return z

    def forward(self, x, c):
        return self.decode(self.encode(x, c), c)

    def log_prob(self, x, c, estimate=False, **kwargs) -> LogProbResult:
        # Then compute JtJ
        config = deepcopy(self.hparams.log_det_estimator)
        estimator_name = config.pop("name")

        if estimate:
            kwargs.update(config)

        if estimator_name == "exact" or not estimate:
            out = log_det_exact(
                x, self.encode, self.decode, c,
                chunk_size=self.hparams.exact_chunk_size,
                **kwargs,
            )
            volume_change = out.log_det
        elif estimator_name == "surrogate":
            out = nll_surrogate(
                x,
                lambda _x: self.encode(_x, c),
                lambda z: self.decode(z, c),
                **kwargs
            )
            volume_change = out.surrogate
        else:
            raise ValueError(f"Cannot understand log_det_estimator.name={estimator_name!r}")

        # Compute log-likelihood
        try:
            loss_gauss = self.get_latent(x.device).log_prob(out.z, c)
        except TypeError:
            loss_gauss = self.get_latent(x.device).log_prob(out.z)

        # Add additional nll terms if available
        for key, value in list(out.regularizations.items()):
            if key.startswith("vol_change_"):
                out.regularizations[key.replace("vol_change_", "nll_")] = -(loss_gauss + value)

        return LogProbResult(
            out.z, out.x1, loss_gauss + volume_change, out.regularizations
        )

    def compute_metrics(self, batch, batch_idx) -> dict:
        """
        Computes the metrics for the given batch.

        Rationale:
        - In training, we only compute the terms that are actually used in the loss function.
        - During validation, all possible terms and metrics are computed.

        :param batch:
        :param batch_idx:
        :return:
        """
        conditioned = self.apply_conditions(batch)
        loss_weights = conditioned.loss_weights
        x = conditioned.x_noisy
        c = conditioned.condition
        x0 = conditioned.x0
        deq_vol_change = conditioned.dequantization_jac

        loss_values = {}
        metrics = {}

        def check_keys(*keys):
            return any(
                (loss_key in loss_weights)
                and
                (
                    torch.any(loss_weights[loss_key] > 0)
                    if torch.is_tensor(loss_weights[loss_key]) else
                    loss_weights[loss_key] > 0
                )
                for loss_key in keys
            )

        # Empty until computed
        x1 = z = None

        # Negative log-likelihood
        if not self.training or (
                self.hparams.exact_train_nll_every is not None
                and batch_idx % self.hparams.exact_train_nll_every == 0
        ):
            key = "nll_exact" if self.training else "nll"
            # todo unreadable
            if self.training or (self.hparams.skip_val_nll is not True and (self.hparams.skip_val_nll is False or (
                    isinstance(self.hparams.skip_val_nll, int)
                    and batch_idx < self.hparams.skip_val_nll
            ))):
                with torch.no_grad():
                    log_prob_result = self.log_prob(x=x, c=c, jacobian_target="both")
                z = log_prob_result.z
                x1 = log_prob_result.x1
                loss_values[key] = -log_prob_result.log_prob - deq_vol_change
                loss_values.update(log_prob_result.regularizations)
            else:
                loss_weights["nll"] = 0
        if self.training and check_keys("nll"):
            warm_up = self.hparams.warm_up_epochs
            if isinstance(warm_up, int):
                warm_up = warm_up, warm_up + 1
            nll_start, warm_up_end = warm_up
            if nll_start == 0:
                nll_warmup = 1
            else:
                nll_warmup = soft_heaviside(
                    self.current_epoch + batch_idx / len(
                        self.trainer.train_dataloader
                        if self.training else
                        self.trainer.val_dataloaders
                    ),
                    nll_start, warm_up_end
                )
            loss_weights["nll"] *= nll_warmup
            if check_keys("nll"):
                log_prob_result = self.log_prob(x=x, c=c, estimate=True)
                z = log_prob_result.z
                x1 = log_prob_result.x1
                loss_values["nll"] = -log_prob_result.log_prob - deq_vol_change
                loss_values.update(log_prob_result.regularizations)

        # In case they were skipped above
        if z is None:
            z = self.encode(x, c)
        if x1 is None:
            x1 = self.decode(z, c)

        if not self.training or check_keys("mmd"):
            try:
                latent_samples = self.get_latent(z.device).sample(z.shape[:1])
                loss_values["mmd"] = maximum_mean_discrepancy(latent_samples, z)
            except (RuntimeError, TypeError):
                # Probably incorrect shape
                pass

        # Wasserstein distance of marginal to Gaussian
        with torch.no_grad():
            z_marginal = z.reshape(-1)
            z_gauss = torch.randn_like(z_marginal)

            z_marginal_sorted = z_marginal.sort().values
            z_gauss_sorted = z_gauss.sort().values

            metrics["z 1D-Wasserstein-1"] = (z_marginal_sorted - z_gauss_sorted).abs().mean()
            metrics["z std"] = torch.std(z_marginal)

        # Reconstruction
        if not self.training or check_keys("reconstruction", "noisy_reconstruction"):
            loss_values["reconstruction"] = reconstruction_loss(x0, x1)
            loss_values["noisy_reconstruction"] = reconstruction_loss(x, x1)

        # Cyclic consistency of latent code
        if not self.training or check_keys("z_reconstruction"):
            # Not reusing x1 from above, as it does not detach z
            z1 = self.encode(x1, c)
            loss_values["z_reconstruction"] = reconstruction_loss(z, z1)

        # Cyclic consistency of latent code -- gradient only to encoder
        if not self.training or check_keys("z_reconstruction_encoder"):
            # Not reusing x1 from above, as it does not detach z
            x1_detached = x1.detach()
            z1 = self.encode(x1_detached, c)
            loss_values["z_reconstruction_encoder"] = reconstruction_loss(z, z1)

        # Cyclic consistency of latent code sampled from Gauss
        if not self.training or check_keys("gauss_z_reconstruction"):
            z_gauss = torch.randn_like(z)
            # We re-use the data noise distribution here, reconstruction should
            # work for all data noises
            x_sample = self.decode(z_gauss, c)
            z_gauss1 = self.encode(x_sample, c)
            loss_values["gauss_z_reconstruction"] = reconstruction_loss(z_gauss, z_gauss1)

        # Reconstruction of Gauss with double std -- for invertibility
        if not self.training or check_keys("gauss_reconstruction"):
            # As we only care about the reconstruction, can ignore noise scale
            x_gauss = 2 * torch.randn_like(x)
            try:
                z_gauss = self.encode(x_gauss, c)
                x_gauss1 = self.decode(z_gauss, c)
                loss_values["gauss_reconstruction"] = reconstruction_loss(x_gauss, x_gauss1)
            except AssertionError:
                loss_values["gauss_reconstruction"] = float("nan") * torch.ones(x.shape[0])

        # Reconstruction of Gauss with double std -- for invertibility
        if not self.training or check_keys("shuffled_reconstruction"):
            # Make noise scale independent of applied noise, reconstruction should still be fine
            x_shuffled = x[torch.randperm(x.shape[0])]
            z_shuffled = self.encode(x_shuffled, c)
            x_shuffled1 = self.decode(z_shuffled, c)
            loss_values["shuffled_reconstruction"] = reconstruction_loss(x_shuffled, x_shuffled1)

        # Compute loss as weighted loss
        metrics["loss"] = sum(
            (weight * loss_values[key]).mean(-1)
            for key, weight in loss_weights.items()
            if check_keys(key) and (self.training or key in loss_values)
        )

        # Metrics are averaged, non-weighted loss_values
        invalid_losses = []
        for key, weight in loss_values.items():
            # One value per key
            if loss_values[key].shape != (x.shape[0],):
                invalid_losses.append(key)
            else:
                metrics[key] = loss_values[key].mean(-1)
        if len(invalid_losses) > 0:
            raise ValueError(f"Invalid loss shapes for {invalid_losses}")

        # Store loss weights
        if self.training:
            for key, weight in loss_weights.items():
                if not torch.is_tensor(weight):
                    weight = torch.tensor(weight)
                self.log(f"weights/{key}", weight.float().mean())

        # $Je Jd = I$ assumption on manifold
        if not self.training and batch_idx == 0 and False:
            # This should be a vmapped jacfwd, but it is not implemented for vmap for our model
            Jd = torch.stack([
                torch.func.jacfwd(self.decode)(zi, ci)
                for zi, ci in zip(z, c)
            ])

            x1 = self.decode(z, c)
            Je = torch.func.vmap(torch.func.jacrev(self.encode))(x1, c)
            JJt = torch.bmm(Je, Jd)

            orthogonality = reconstruction_loss(
                torch.eye(JJt.shape[-1], device=x.device).reshape(-1).unsqueeze(0),
                JJt.reshape(x.shape[0], -1)
            )
            metrics["orthogonality"] = orthogonality.mean()
            metrics["orthogonality-std"] = orthogonality.std()

        # Check finite loss
        if not torch.isfinite(metrics["loss"]) and self.training:
            self.trainer.save_checkpoint("erroneous.ckpt")
            print(f"Encountered nan loss from: {metrics}!")
            raise SkipBatch

        return metrics

    def on_train_epoch_end(self) -> None:
        try:
            for key, value in self.val_data.compute_metrics(self).items():
                self.log(f"validation/{key}", value)
        except AttributeError:
            pass

    def apply_conditions(self, batch) -> ConditionedBatch:
        x0 = batch[0]
        base_cond_shape = (x0.shape[0], 1)
        device = x0.device
        dtype = x0.dtype

        conds = []

        # Dataset condition
        if len(batch) != (2 if self.is_conditional() else 1):
            raise ValueError("You must pass a batch including conditions for each dataset condition")
        if len(batch) > 1:
            conds.append(batch[1])

        # SoftFlow
        noise_conds, x, dequantization_jac = self.dequantize(batch)
        conds.extend(noise_conds)

        # Loss weight aware
        loss_weights = defaultdict(float, self.hparams.loss_weights)
        for loss_key, loss_weight in self.hparams.loss_weights.items():
            if isinstance(loss_weight, list):
                min_weight, max_weight = loss_weight
                if not self.training:
                    # Per default, select the first value in the list
                    max_weight = min_weight
                weight_scale = rand_log_uniform(
                    min_weight, max_weight,
                    shape=base_cond_shape, device=device, dtype=dtype
                )
                loss_weights[loss_key] = (10 ** weight_scale).squeeze(1)
                conds.append(weight_scale)

        if len(conds) == 0:
            c = torch.empty((x.shape[0], 0), device=x.device, dtype=x.dtype)
        elif len(conds) == 1:
            # This is a hack to pass through the info dict from QM9
            c, = conds
        else:
            c = torch.cat(conds, -1)
        return ConditionedBatch(x0, x, loss_weights, c, dequantization_jac)

    def dequantize(self, batch):
        x0 = batch[0]
        base_cond_shape = (x0.shape[0], 1)
        device = x0.device
        dtype = x0.dtype

        noise = self.hparams.noise
        if isinstance(noise, list):
            min_noise, max_noise = noise
            if not self.training:
                max_noise = min_noise
            noise_scale = rand_log_uniform(
                max_noise, min_noise,
                shape=base_cond_shape, device=device, dtype=dtype
            )
            x = x0 + torch.randn_like(x0) * (10 ** noise_scale)
            noise_conds = [noise_scale]
        else:
            if noise > 0:
                x = x0 + torch.randn_like(x0) * noise
            else:
                x = x0
            noise_conds = []
        return noise_conds, x, torch.zeros(x0.shape[0], device=device, dtype=dtype)


def build_model(models, data_dim: int, cond_dim: int):
    if not isinstance(models[0], dict):
        return Sequential(*models)
    models = deepcopy(models)
    model = Sequential()
    for model_spec in models:
        module_name, class_name = model_spec.pop("name").rsplit(".", 1)
        model_spec["data_dim"] = data_dim
        model_spec["cond_dim"] = cond_dim
        if model_spec["latent_dim"] == "data":
            model_spec["latent_dim"] = data_dim
        model.append(
            getattr(import_module(module_name), class_name)(model_spec)
        )
        data_dim = model_spec["latent_dim"]
    return model


def soft_heaviside(pos, start, stop):
    return max(0., min(
        1.,
        (pos - start)
        /
        (stop - start)
    ))


def reconstruction_loss(a, b):
    return sum_except_batch((a - b) ** 2)


def rand_log_uniform(vmin, vmax, shape, device, dtype):
    vmin, vmax = map(log10, [vmin, vmax])
    return torch.rand(
        shape, device=device, dtype=dtype
    ) * (vmin - vmax) + vmax


def wasserstein2_distance_gaussian_approximation(x1, x2):
    # Returns the squared 2-Wasserstein distance between the Gaussian approximation of two datasets x1 and x2
    # 1. Calculate mean and covariance of x1 and x2
    # 2. Use fact that tr( ( cov1^(1/2) cov2 cov1^(1/2) )^(1/2) ) = sum(eigvals( ( cov1^(1/2) cov2 cov1^(1/2) )^(1/2) )) 
    # = sum(eigvals( cov1 cov2 )^(1/2))
    # 3. Return ||m1 - m2||^2 + tr( cov1 + cov2 - 2 ( cov1^(1/2) cov2 cov1^(1/2) )^(1/2) )
    m1 = x1.mean(0)
    m2 = x2.mean(0)
    cov1 = (x1 - m1[None]).T @ (x1 - m1[None]) / x1.shape[0]
    cov2 = (x2 - m2[None]).T @ (x2 - m2[None]) / x2.shape[0]
    cov_product = cov1 @ cov2
    eigenvalues_prod = torch.relu(torch.linalg.eigvals(cov_product).real)
    m_part = torch.sum((m1 - m2) ** 2)
    cov_part = torch.trace(cov1) + torch.trace(cov2) - 2 * torch.sum(torch.sqrt(eigenvalues_prod))
    return m_part + cov_part
