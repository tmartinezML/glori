import itertools
from types import SimpleNamespace
from contextlib import nullcontext

import torch
import numpy as np

import glori.settings.paths as paths
import glori.infra.logging as my_logging
import glori.models.utils as mutil
from glori.models.vae.vqvae import VQVAE
from glori.models.diffusion.denoiser import Denoiser
from glori.inference.dm_sampler import DMSampler
from glori.inference.context.context_maps_utils import dice_pos_mask
from glori.data.trf.scalers import LOFARScaler, ContextScaler


class LDMSampler:
    def __init__(
        self,
        **settings,
    ):
        self.logger = my_logging.get_logger(__name__)
        self.logger.divider()

        default_settings = dict(
            denoiser="LDM-Denoiser-CatCond",
            denoiser_ckpt="best",
            uncond_denoiser=None,
            uncond_denoiser_ckpt="best",
            vae="VQ-VAE-256",
            vae_ckpt="best",
            use_ema={
                "denoiser": False,
                "uncond_denoiser": False,
                "vae": False,
            },
            px_scaler="LOFAR_scaler_II",
            image_size=256,
            latent_size=64,
            latent_channels=3,
            device="cuda:0",
            progress_bar=True,
            ctxt_scalers=["ctxt_scaler_ftot", "ctxt_scaler_fpeak", "ctxt_scaler_maj"],
            timesteps=25,
        )
        # Update default settings with provided settings
        default_settings.update(settings)
        self.settings = SimpleNamespace(**default_settings)

        # Print settings
        self.logger.info("Initializing LDMSampler with settings:")
        # If initialized models are provided, avoid printing them fully
        my_logging.pretty_print_config(
            SimpleNamespace(
                **{
                    k: (
                        v
                        if not isinstance(v, (Denoiser, VQVAE))
                        else f"<Provided Model: {type(v).__name__}>"
                    )
                    for k, v in self.settings.__dict__.items()
                }
            )
        )

        # Load denoiser
        if isinstance(self.settings.denoiser, str):
            ckpt = mutil.parse_lightning_ckpt(
                self.settings.denoiser_ckpt, model_name=self.settings.denoiser
            )
            self.logger.info(
                f"Loading Denoiser:\t{ckpt.parent.parent.name}\nfrom:\t{ckpt.name}"
            )
            self.denoiser = Denoiser.load_from_checkpoint(ckpt, map_location="cpu")
        else:
            self.logger.info("Using provided denoiser model.")
            self.denoiser = self.settings.denoiser

        # Initialize DM sampler
        self.dm_sampler = DMSampler(
            image_size=self.settings.latent_size,
            image_channels=self.settings.latent_channels,
            return_steps=False,
        )

        # If provided, load unconditional denoiser
        if self.settings.uncond_denoiser is not None:
            if isinstance(self.settings.uncond_denoiser, str):
                uncond_ckpt = mutil.parse_lightning_ckpt(
                    self.settings.uncond_denoiser_ckpt,
                    model_name=self.settings.uncond_denoiser,
                )
                self.logger.info(
                    f"Loading Unconditional Denoiser:\t{uncond_ckpt.parent.parent.name}\nfrom:\t{uncond_ckpt.name}"
                )
                self.uncond_denoiser = Denoiser.load_from_checkpoint(
                    uncond_ckpt, map_location="cpu"
                )
            else:
                self.logger.info("Using provided unconditional denoiser model.")
                self.uncond_denoiser = self.settings.uncond_denoiser
        # If no unconditional denoiser is specified, set to None
        else:
            self.logger.info("No separate unconditional denoiser will be used.")
            self.uncond_denoiser = None

        # Load VAE
        if isinstance(self.settings.vae, str):
            ckpt = mutil.parse_lightning_ckpt(
                self.settings.vae_ckpt, model_name=self.settings.vae
            )
            self.logger.info(
                f"Loading VAE:\t{ckpt.parent.parent.name}\nfrom:\t{ckpt.name}"
            )
            self.vae = VQVAE.load_from_checkpoint(ckpt, map_location="cpu")
        else:
            self.logger.info("Using provided VAE model.")
            self.vae = self.settings.vae
        if self.settings.use_ema["vae"]:
            self.logger.warning(
                "EMA for VAE is not implemented. Will ignore the setting."
            )

        # Load scalers
        self.logger.info(f"Loading scalers...")
        if isinstance(self.settings.px_scaler, str):
            self.px_scaler = LOFARScaler.load(self.settings.px_scaler)
        else:
            self.px_scaler = self.settings.px_scaler
        self.scaler = self.px_scaler  # Alias for compatibility
        self.ctxt_scalers = [
            ContextScaler.load(scaler) for scaler in self.settings.ctxt_scalers
        ]

        self.logger.info("LDMSampler initialized successfully.")
        self.logger.divider()

    def sample_latents(
        self,
        inpainting_context=None,
        img_context=None,
        catalog_context=None,
        **diffusion_kw,
    ):

        # If only context is passed, we have to create a dummy inpainting context
        if img_context is not None and inpainting_context is None:
            inpainting_mask = torch.tensor([1, 1, 1, 1]).reshape(1, 1, 2, 2).float()
            inpainting_mask = torch.nn.functional.interpolate(
                inpainting_mask,
                scale_factor=self.settings.latent_size // 2,
                mode="nearest",
            )
            inpainting_mask = inpainting_mask.repeat(img_context.shape[0], 1, 1, 1)
            inpainting_context = (
                inpainting_mask,
                inpainting_mask.repeat(1, self.settings.latent_channels, 1, 1),
            )

        # Set diffusion sampling kwargs
        kw = {
            k: v
            for k, v in vars(self.settings).items()
            if k in self.dm_sampler.settings and k != "image_size"
        }
        kw.update(diffusion_kw)

        # Prepare denoiser model
        self.denoiser.eval()
        self.denoiser = self.denoiser.to(self.settings.device)
        if self.uncond_denoiser is not None:
            self.uncond_denoiser = self.uncond_denoiser.to(self.settings.device)
            self.uncond_denoiser.eval()

        # Sample latents
        with (
            (
                self.denoiser.use_ema_weights()
                if self.settings.use_ema["denoiser"]
                else nullcontext()
            ),
            (
                self.uncond_denoiser.use_ema_weights()
                if self.settings.use_ema["uncond_denoiser"]
                and self.uncond_denoiser is not None
                else nullcontext()
            ),
        ):
            sampled_enc = self.dm_sampler.quick_sample(
                model=self.denoiser,
                uncond_model=self.uncond_denoiser,
                inpainting_context=(
                    (m.to(self.settings.device) for m in inpainting_context)
                    if inpainting_context is not None
                    else None
                ),
                img_context=(
                    img_context.to(self.settings.device)
                    if img_context is not None
                    else None
                ),
                catalog_context=(
                    catalog_context.to(self.settings.device)
                    if catalog_context is not None
                    else None
                ),
                distribute_model=False,
                rescale_images=False,
                **kw,
            )

        # Release GPU memory
        self.denoiser = self.denoiser.cpu()
        if self.uncond_denoiser is not None:
            self.uncond_denoiser = self.uncond_denoiser.cpu()

        return sampled_enc

    def decode_latents(self, latents):
        """
        Decode the latent representation into an image.
        """
        self.vae.eval()
        self.vae = self.vae.to(self.settings.device)

        if isinstance(latents, np.ndarray):
            self.logger.warning(
                "Latents are in numpy format, converting to torch tensor."
            )
            latents = torch.from_numpy(latents).to(self.settings.device)

        with torch.no_grad():
            decoded_image = self.vae.decode(latents)

        self.vae = self.vae.cpu()

        return decoded_image.cpu().numpy()

    def sample(
        self,
        inpainting_context=None,
        catalog_context=None,
        img_context=None,
        rescale=True,
        return_latents=False,
        **diffusion_kw,
    ):
        """
        Sample an image from the model.
        """
        # Check for shapes
        if inpainting_context is not None:
            assert (
                inpainting_context[0].shape[-2:] == inpainting_context[1].shape[-2:]
            ), f"Mask input must have consistent shape, got {inpainting_context[0].shape[-2:]} and {inpainting_context[1].shape[-2:]}"
            assert (
                len(set(inpainting_context[0].shape[-2:])) == 1
            ), f"Mask input must be square-shaped, got {inpainting_context[0].shape[-2:]}"
        if img_context is not None:
            assert (
                len(set(img_context.shape[-2:])) == 1
            ), f"Context must be square-shaped, got {img_context.shape[-2:]}"
        if img_context is not None and inpainting_context is not None:
            assert (
                img_context.shape[-2:] == inpainting_context[0].shape[-2:]
            ), f"Context and mask input must have the same spatial dimensions, got {img_context.shape[-2:]} and {inpainting_context[0].shape[2:]}"
        if catalog_context is not None:
            assert (
                len(set(catalog_context.shape[-2:])) == 1
            ), f"Catalog context must be square-shaped, got {catalog_context.shape[-2:]}"
        if img_context is not None or inpainting_context is not None:
            inp_shape = (
                img_context.shape[-2:]
                if img_context is not None
                else inpainting_context[0].shape[-2:]
            )
            if inp_shape[0] != self.settings.latent_size:
                self.logger.warning(
                    f"Input shape {inp_shape} does not match latent size {self.settings.latent_size}. Changing sampler settings. Check image size settings as well!"
                )
                self.settings.latent_size = inp_shape[0]

        self.logger.info("Sampling images...")

        # Sample latents
        sampled_latents = self.sample_latents(
            inpainting_context=inpainting_context,
            img_context=img_context,
            catalog_context=catalog_context,
            **diffusion_kw,
        )

        # Decode latents to image
        image = self.decode_latents(
            torch.from_numpy(sampled_latents).to(self.settings.device)
        )

        # Rescale image
        if rescale:
            image = self.px_scaler.inverse_scale(image)

        self.logger.info("Image sampling completed.")

        if return_latents:
            return image, sampled_latents
        return image

    def _context_scaler_pass(self, values, is_scaled):
        """
        Pass values through the scalers if they are not already scaled.
        """
        match is_scaled:
            case bool():
                is_scaled = (is_scaled,) * len(values)
            case tuple():
                assert len(is_scaled) == len(
                    values
                ), "is_scaled must be a boolean or a tuple of booleans with the same length as values."

        for i, (value, scaler) in enumerate(zip(values, self.scalers)):
            if not is_scaled[i]:
                values[i] = scaler.scale(value)

        return values

    def explorer_context(self, ftot, fpeak, maj, n_pts=3, is_scaled=True, img_size=64):

        # Scale values if necessary
        values = self._context_scaler_pass([ftot, fpeak, maj], is_scaled)

        ctxt = torch.zeros((img_size,) * 2)
        for i, j in itertools.product(range(1, n_pts + 1), repeat=2):
            ctxt[i * img_size // (n_pts + 1), j * img_size // (n_pts + 1)] = 1

        bsize = 1
        # Check if any of the values has len() function defined
        haslen = tuple(hasattr(value, "__len__") for value in values)
        if any(haslen):
            # Assert all have same length
            ll = [len(v) for h, v in zip(haslen, values) if h]
            assert (
                len(set(ll)) == 1
            ), f"All values must have the same length or be float, got {ll}."
            bsize = ll[0]

        ctxt = ctxt.reshape(1, 1, img_size, img_size).repeat(bsize, 4, 1, 1)
        for i in range(len(values)):
            ctxt[:, i + 1] *= values[i].reshape(-1, 1, 1) if haslen[i] else values[i]

        ctxt[:, 0] = ctxt[:, 0] * 2 - 1  # Scale position context

        return ctxt

    def dice_context(self, ftot, fpeak, maj, is_scaled=True):
        # Scale values if necessary
        values = self._context_scaler_pass([ftot, fpeak, maj], is_scaled)

        ctxt = (
            dice_pos_mask(
                enc_size=self.settings.latent_size,
                spacing="quarters",
            )
            .unsqueeze(1)
            .repeat(1, 4, 1, 1)
        )

        ctxt[:, 0] = ctxt[:, 0] * 2 - 1  # Scale position context
        ctxt[:, 1:] *= torch.tensor(values).reshape(-1, 1, 1)

        return ctxt
