import contextlib
from types import SimpleNamespace
from pathlib import Path
from datetime import datetime

import torch
import lightning as L
from torchvision.transforms.v2 import RandomCrop
from torchvision.transforms.v2.functional import crop

import glori.models.networks.dit as dit
from glori.models.networks.modules import configModuleBaseLightning, NaNDetectedError
from glori.models.networks.unet import EDMPrecond, UnetTimeEmb
from glori.config.model_config import modelConfig
import glori.models.diffusion.loss
import glori.models.diffusion.context_masks as tutils

from pytorch_lightning.utilities.rank_zero import rank_zero_only


class NaNDebugCallback(L.Callback):
    """
    Callback that saves a single .pt debug dump when a NaNDetectedError is reported.
    File is written into trainer.default_root_dir as ckpt_nan_debug_{train_step}.pt
    """

    def __init__(self, filename_template="ckpt_nan_debug_{train_step:08.0f}.pt"):
        super().__init__()
        self.filename_template = filename_template

    def on_exception(self, trainer, pl_module, exception):
        if not isinstance(exception, NaNDetectedError):
            print(
                f"Exception {type(exception)} is not NaNDetectedError, skipping NaN debug dump."
            )
            return
        print("NaNDetectedError caught, saving NaN debug dump...")
        out_dir = Path(trainer.default_root_dir or Path.cwd())
        out_dir.mkdir(parents=True, exist_ok=True)
        step = getattr(trainer, "global_step", 0)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        fname = out_dir / self.filename_template.format(train_step=step)
        payload = {
            **pl_module._debug_payload,
            "model_state_dict": pl_module.model.state_dict(),
            "global_step": step,
            "error": repr(exception),
            "timestamp": ts,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "optimizer_state_dict": trainer.optimizers[0].state_dict(),
        }
        try:
            torch.save(payload, fname)
            print(f"Saved NaN debug dump to {fname}")
        except Exception as e:
            print(f"Failed to save NaN debug dump to {fname}: {e}")


class EMACallback(L.Callback):
    def __init__(self, ema_model, warmup_steps=2500):
        super().__init__()
        self.ema_model = ema_model
        self.warmup_steps = warmup_steps
        self.ema_initialized = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Skip EMA during warmup period
        if trainer.global_step < self.warmup_steps:
            return

        if not self.ema_initialized:
            self.ema_initialized = True

        # Update EMA model after optimizer step
        self.ema_model.update_parameters(pl_module.model)


class Denoiser(configModuleBaseLightning):
    """
    Denoiser class for training and inference with a UNet-based model.

    Attributes:
        model (EDMPrecond): The UNet-based model for denoising.
        config (configModuleBaseLightning): Configuration parameters for the model.
    """

    def __init__(
        self,
        backbone,
        model_config,
        train_config,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_config = modelConfig(**train_config)

        # Load denoiser backbone
        if backbone == "Unet":
            self.model = EDMPrecond.from_config(modelConfig(**model_config))

        elif backbone == "DiT":
            self.model = dit.DiTWrapper(dit.DiT(**model_config))

        elif backbone in dit.DiT_models:
            self.model = dit.DiTWrapper(dit.DiT_models[backbone](**model_config))

        else:
            raise ValueError(
                f"Unsupported model backbone: {backbone}. "
                "Supported backbones are: 'Unet', 'DiT', or any key from `dit.DiT_models`."
            )

        # Initialize EMA model
        self.use_ema = self.train_config.get("ema_rate") is not None
        if self.use_ema:
            self.ema_model = torch.optim.swa_utils.AveragedModel(
                self.model,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                    self.train_config.ema_rate
                ),
            )
            self.ema_callback = EMACallback(
                self.ema_model,
                warmup_steps=(
                    self.train_config.ema_warmup_steps
                    if hasattr(self.train_config, "ema_warmup_steps")
                    else 2500
                ),
            )

        self.sigma_min = self.model.sigma_min
        self.sigma_max = self.model.sigma_max
        self.sigma_data = self.model.sigma_data
        self.in_channels = self.model.model.input_channels

        # Commented out on Oct 17th 25: This was moved to the dataset class
        # self.random_crop = getattr(self.train_config, "random_crop", None)
        # self.crop_fctxt = getattr(self.train_config, "crop_fctxt", 1)
        self.random_crop = None
        self.crop_fctxt = None

        # Added on Feb 10th 2026: For resuming partial checkpoints.
        self.strict_loading = False

    @classmethod
    def from_config(cls, config):
        """
        Create a Denoiser instance from a configuration dictionary.

        Parameters:
            config (dict): Configuration dictionary containing model and training parameters.

        Returns:
            Denoiser: An instance of the Denoiser class.
        """
        model_config = config.model_config
        train_config = config.train_config
        return cls(config.backbone, model_config, train_config)

    @classmethod
    def from_preset(cls, preset_name):
        """
        Create a Denoiser instance from a preset configuration.

        Parameters:
            preset_name (str): Name of the preset configuration.
        Returns:
            Denoiser: An instance of the Denoiser class.
        """
        conf = modelConfig.from_preset(preset_name)
        # Some global config settings should propagate into the training and
        # model configs for convenience
        if conf.get("img_context") is not None:
            conf.train_config["context"].append("img_context")
        if conf.get("catalog_context") is not None:
            conf.train_config["context"].append("catalog_context")
        if (topk := conf.get("catalog_topk")) is not None and conf.get(
            "catalog_context"
        ) is not None:
            conf.model_config["catalog_context_topk"] = topk
            conf.train_config["catalog_context_topk"] = topk
        return cls.from_config(conf)

    def forward(
        self,
        x,
        sigmas,
        img_context=None,
        catalog_context=None,
        context=None,
        class_labels=None,
    ):

        if img_context is None and x.shape[1] != self.model.model.input_channels:
            img_context = torch.zeros(
                (
                    x.shape[0],
                    x.shape[1] + 1,  # Inpainting context + mask
                    *x.shape[2:],
                ),
                device=x.device,
            )
        if catalog_context is None and self.model.model.use_catalog_ctxt:
            catalog_context = torch.zeros(
                (
                    x.shape[0],
                    self.model.model.catalog_emb[0].input_channels,
                    x.shape[-2] * 2,
                    x.shape[-1] * 2,
                ),
                device=x.device,
            )

        return self.model(
            x,
            sigmas,
            context=context,
            class_labels=class_labels,
            img_context=img_context,
            catalog_context=catalog_context,
        )

    def read_batch(self, batch):

        ctxt_keys = getattr(self.train_config, "context", [])
        assert len(ctxt_keys) == len(batch) - 1, (
            f"Expected {len(ctxt_keys) + 1} elements in batch, "
            f"given context keys {ctxt_keys}, "
            f"but got {len(batch)}."
        )
        ctxt_dict = {
            "img_context": None,
            "catalog_context": None,
            "context": None,
            "class_labels": None,
        }
        assert all(k in ctxt_dict for k in ctxt_keys), (
            f"Context keys {ctxt_keys} contain unsupported keys. "
            f"Supported keys are: {list(ctxt_dict.keys())}."
        )
        for i, k in enumerate(ctxt_keys):
            ctxt_dict[k] = batch[i + 1]

        img = batch[0]

        return img, ctxt_dict

    def loss(self, batch, apply_crop=True):
        """
        Compute the loss for a given batch.

        Parameters:
            batch (dict): A batch of data containing 'img_batch', 'noise', and 'sigmas'.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        x, ctxt_dict = self.read_batch(batch)

        if self.train_config.use_mask:
            use_ext = self.train_config.get("use_ext_ctxt_mask", False)
            mask_coverage = self.train_config.get("mask_coverage", 0.5)
            mask = tutils.weighted_random_quadrant_mask(
                x.shape,
                weights=self.train_config.get(
                    "training_mask_weights",
                    [0.25] * 4,
                ),
                make_ext_ctxt_mask=use_ext,
                ext_ctxt_p=self.train_config.get("ext_mask_p", [0.9, 0.1, 0.1]),
                mask_coverage=mask_coverage,
            )
            if use_ext:
                mask, ext_ctxt_mask = mask
                ctxt_dict["catalog_context"] *= ext_ctxt_mask.to(
                    ctxt_dict["catalog_context"].device
                )

        sigmas = glori.models.diffusion.loss.sample_sigmas(
            x, self.train_config.P_mean, self.train_config.P_std
        )
        n = torch.randn_like(x) * sigmas
        try:
            loss = glori.models.diffusion.loss.edm_loss(
                model=self.model,
                img_batch=x,
                **ctxt_dict,
                sigma_data=self.model.sigma_data,
                noise=n,
                sigmas=sigmas,
                mask=mask if self.train_config.use_mask else None,
                return_output=False,
            )
            self._debug_payload = {
                "batch": batch,
                "noise": n,
                "sigmas": sigmas,
            }
            if self.train_config.use_mask:
                self._debug_payload["mask"] = mask
        except NaNDetectedError as exc:
            self._debug_payload = {
                "batch": batch,
                "noise": n,
                "sigmas": sigmas,
            }
            if self.train_config.use_mask:
                self._debug_payload["mask"] = mask
            raise exc

        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step for the denoiser model.

        Parameters:
            batch (dict): A batch of data containing 'img_batch', 'noise', and 'sigmas'.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        loss = self.loss(batch)

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_step", self.trainer.global_step, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the denoiser model.

        Parameters:
            batch (dict): A batch of data containing 'img_batch', 'noise', and 'sigmas'.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        loss = self.loss(batch)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if (
            self.use_ema
            and self.train_config.validate_ema
            and self.ema_callback.ema_initialized
        ):
            with self.use_ema_weights():
                ema_loss = self.loss(batch)
        else:
            ema_loss = 10
        self.log(
            "val/ema_loss",
            ema_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_ema_loss",
            ema_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        # Log loss for checkpointing
        # log_metric = ema_loss if self.train_config.validate_ema else loss
        log_metric = loss
        self.log(
            "val_loss",
            log_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for the denoiser model.

        Returns:
            torch.optim.Optimizer: The optimizer for the model.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
        )
        if (
            hasattr(self.train_config, "scheduler")
            and self.train_config.scheduler is not None
        ):
            # Use the specified scheduler if it exists
            scheduler = getattr(torch.optim.lr_scheduler, self.train_config.scheduler)(
                optimizer,
                **self.train_config.scheduler_kwargs,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": dict(
                    scheduler=scheduler, **self.train_config.scheduler_settings
                ),
            }

        # Otherwise, no scheduler
        return optimizer

    def configure_callbacks(self):
        callbacks = [
            NaNDebugCallback(),
        ]
        if self.use_ema:
            callbacks.append(self.ema_callback)
        return callbacks

    def on_save_checkpoint(self, checkpoint):
        # Save EMA weights
        if self.use_ema:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint):

        # Load EMA weights if present
        if ("ema_state_dict" in checkpoint) and self.use_ema:
            self.ema_model.load_state_dict(
                self._rename_state_dict_keys(checkpoint["ema_state_dict"])
            )
        checkpoint["state_dict"] = self._rename_state_dict_keys(
            checkpoint["state_dict"]
        )

    def _rename_state_dict_keys(self, state_dict):
        """Helper method to rename keys for backwards compatibility."""
        keys_to_rename = []
        for old_key in list(state_dict.keys()):
            new_key = old_key

            if "middle_block" in old_key and not "catalog_emb" in old_key:
                new_key = (
                    old_key.replace("middle_block.0", "middle_blocks.0.resBlock")
                    .replace("middle_block.1", "middle_blocks.0.attnBlock")
                    .replace("middle_block.2", "middle_blocks.1")
                )

            if new_key != old_key:
                keys_to_rename.append((old_key, new_key))

        # Apply renamings
        renamed_state_dict = state_dict.copy()
        for old_key, new_key in keys_to_rename:
            renamed_state_dict[new_key] = renamed_state_dict.pop(old_key)

        if keys_to_rename:
            print(f"Renamed {len(keys_to_rename)} keys for backwards compatibility")

        return renamed_state_dict

    # def load_state_dict(self, state_dict, strict=True):
    #     """Override to rename keys before loading for backwards compatibility."""
    #     # Rename keys using shared helper method
    #     renamed_state_dict = self._rename_state_dict_keys(state_dict)

    #     if len(renamed_state_dict) != len(state_dict):
    #         print(f"[Model] Renamed keys for backwards compatibility")

    #     # Call parent's load_state_dict with renamed keys
    #     return super().load_state_dict(renamed_state_dict, strict=strict)

    @contextlib.contextmanager
    def use_ema_weights(self):
        """Context manager to temporarily use EMA weights."""
        # Backup current weights
        device = next(self.parameters()).device
        backup = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(
            {k: v.to(device) for k, v in self.ema_model.module.state_dict().items()}
        )
        try:
            yield
        finally:

            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in backup.items()}
            )
