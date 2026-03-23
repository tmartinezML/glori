import pickle
from types import SimpleNamespace

import torch
import numpy as np
from scipy.special import boxcox, inv_boxcox

import utils.paths as paths


class scalerBase:
    def __init__(self, **parameters):
        # Set parameters as attributes
        self.pms = SimpleNamespace(**parameters)

    @classmethod
    def load(cls, name, **kwargs):
        path = paths.LOFAR_DATA_PARENT / "scalers" / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Scaler file {path} does not exist.")
        pms = torch.load(path)
        scaler = cls(**pms, **kwargs)
        return scaler

    def save(self, name):
        path = paths.LOFAR_DATA_PARENT / "scalers" / f"{name}.pt"
        torch.save(vars(self.pms), path)

    def scale(self, x):
        raise NotImplementedError("This method should be overridden by subclasses")

    def inverse_scale(self, x):
        raise NotImplementedError("This method should be overridden by subclasses")


class LOFARScaler(scalerBase):

    def calculate_params(self, px_sample):
        # Get stats from reference pixels
        log_pos = np.log(px_sample[px_sample > 0])
        log_neg = -np.log(-px_sample[px_sample < 0])

        # Set values for scaling
        a_pos = 1 / np.std(log_pos)
        a_neg = 1 / np.std(log_neg)

        # Set values for standardizing
        px_log_scaled = self.sym_log_scale(self.px_sample, var_scale=False)
        stdize_mean = np.mean(px_log_scaled)
        stdize_std = np.std(px_log_scaled)

        pms_dict = vars(self.pms)
        pms_dict.update(
            a_pos=a_pos,
            a_neg=a_neg,
            stdize_mean=stdize_mean,
            stdize_std=stdize_std,
        )
        self.pms = SimpleNamespace(**pms_dict)

        # Print all values:
        print(self.pms)

    def scale(self, px):
        px = px.copy() if isinstance(px, np.ndarray) else px.clone()
        return self.sym_log_scale(px)

    def inverse_scale(self, px, **kwargs):
        px = px.copy() if isinstance(px, np.ndarray) else px.clone()
        return self.sym_log_scale_inverse(px, **kwargs)

    def scaler_fn(self, px, a):
        return a * np.log(px * self.pms.slope / a + 1)

    def inverse_scaler_fn(self, px, a, use_torch=False):
        if use_torch:
            return a / self.pms.slope * (torch.exp(px / a) - 1)
        return a / self.pms.slope * (np.exp(px / a) - 1)

    def sym_log_scale(self, px, var_scale=True):
        pos_flag = px > 0

        # Positive pixels
        px[pos_flag] = self.scaler_fn(px[pos_flag], self.pms.a_pos)

        # Negative pixels
        px[~pos_flag] = -self.scaler_fn(-px[~pos_flag], self.pms.a_neg)

        if var_scale:
            # By choice, we're not subtracting the mean here
            px = px / self.pms.stdize_std

        return px

    def sym_log_scale_inverse(self, px, var_scale=True, use_torch=False):
        if var_scale:
            # By choice, we're not adding the mean here
            px = px * self.pms.stdize_std
        pos_flag = px > 0

        # Positive pixels
        px[pos_flag] = self.inverse_scaler_fn(
            px[pos_flag], self.pms.a_pos, use_torch=use_torch
        )

        # Negative pixels
        px[~pos_flag] = -self.inverse_scaler_fn(
            -px[~pos_flag], self.pms.a_neg, use_torch=use_torch
        )

        return px


class ContextScaler(scalerBase):
    def __init__(self, presc_and_inv=(np.log1p, np.expm1), **params):
        self.pre_scale, self.pre_scale_inv = presc_and_inv
        super().__init__(**params)

    def scale(self, x):
        x = self.pre_scale(x)
        x = boxcox(x, self.pms.lambda_)
        x = (x - self.pms.mean) / self.pms.std
        return x

    def inverse_scale(self, x):
        x = x * self.pms.std + self.pms.mean
        x = inv_boxcox(x, self.pms.lambda_)
        x = self.pre_scale_inv(x)
        return x
