import copy
from typing import Unpack
from functools import wraps
from itertools import product
from dataclasses import replace

import torch

import glori.infra.logging as my_logging
from glori.infra.devices import is_jupyter
from glori.models.load import parse_lightning_ckpt
from glori.models.vae.vqvae import VQVAE
from glori.models.diffusion.denoiser import Denoiser
from glori.inference.dm_sampler import DMSampler
from glori.models.networks.modules import configModuleBaseLightning
from glori.config.swiit_config import SWIITSamplerConfig

# Set up tqdm progress bar depending on whether we're in a Jupyter environment or not
if is_jupyter:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm


class SWIITSampler:
    def __init__(
        self,
        config: SWIITSamplerConfig | None = None,
        **kwargs,
    ):
        # Initialize logger
        self.logger = my_logging.get_logger("SWIITSampler")

        # Load config
        if config is None:
            self.logger.info("No config provided, initializing default...")
            config = SWIITSamplerConfig()
        self.logger.info("Overriding config with provided kwargs...")
        config = replace(config, **kwargs)
        self.config = config  # Save config as attribute for use in other methods

        # Save config
        self.config = config

        # Initialize denoiser
        self.denoiser = self._load_model(
            self.config.denoiser, self.config.denoiser_ckpt, Denoiser
        )
        if self.denoiser is None:
            self.logger.warning(
                "No denoiser model provided, sampling will not be possible."
            )
        else:
            self.denoiser.eval()

        # Initialize unconditional denoiser if specified, otherwise set to None
        self.uncond_denoiser = self._load_model(
            self.config.uncond_denoiser, self.config.uncond_denoiser_ckpt, Denoiser
        )

        # Initialize VQ-VAE
        self.vae = self._load_model(self.config.vae, self.config.vae_ckpt, VQVAE)

        if self.vae is None:
            self.logger.warning("No VAE model provided, decoding will not be possible.")
        else:
            self.vae.eval()

        # Load diffusion sampler
        # TODO: Make DM sampling config dataclass
        self.logger.info(f"Initializing diffusion sampler...")
        self.sampler = DMSampler(
            image_size=self.config.latent_size,
            return_steps=False,
        )
        self.sampler.logger.disabled = True  # Disable sampler logging

        self.logger.info("ICM Sampler initialized.")

    def with_temporary_settings(func):
        @wraps(func)
        def wrapper(self, *args, config_override=None, **kwargs):

            if not config_override:
                return func(self, *args, **kwargs)

            original = copy.deepcopy(self.config)

            try:
                self.config = replace(self.config, **config_override)
                return func(self, *args, **kwargs)
            finally:
                self.config = original

        return wrapper

    def _load_model(
        self, model_name: str | None, ckpt: str, model_cls: configModuleBaseLightning
    ) -> configModuleBaseLightning | None:
        if model_name is None:
            return None

        # Get checkpoint path and load model
        ckpt = parse_lightning_ckpt(ckpt, model_name=model_name)
        self.logger.info(
            f"Loading model <{model_name}> from checkpoint:\n\t{ckpt.name}..."
        )
        model_instance = model_cls.load_from_checkpoint(ckpt, map_location="cpu")
        return model_instance

    def inpainting_mask(
        self, *flg: Unpack[tuple[int, int, int, int]], size: str = "latent"
    ) -> torch.Tensor:

        # Ensure correct number of flags are provided
        assert len(flg) == 4, f"Four flags required for quadrant mask, got input: {flg}"

        # Get the appropriate mask size based on the specified size parameter
        match size:
            case "image":
                # Create a mask for the image size
                mask_size = self.config.image_size
            case "latent":
                # Create a mask for the latent size
                mask_size = self.config.latent_size
            case _:
                raise ValueError(
                    f"Invalid size '{size}' specified. Use 'image' or 'latent'."
                )

        # Get the flags for each quadrant and reshape to create the mask
        m1, m2, m3, m4 = torch.tensor(flg).int().reshape(4, 1, 1)

        # Initialize empty mask, fill with values
        mask = torch.zeros(mask_size, mask_size).int()
        pixels_covered = int(mask_size * self.config.mask_coverage)
        mask[:pixels_covered, :pixels_covered] = m1  # Top left
        mask[:pixels_covered, pixels_covered:] = m2  # Top right
        mask[pixels_covered:, :pixels_covered] = m3  # Bottom left
        mask[pixels_covered:, pixels_covered:] = m4  # Bottom right

        # Unsqueeze to add batch and channel dimensions
        mask = mask.reshape(1, 1, mask_size, mask_size)

        return mask

    def inpainting_mask_for_iteration(self, i: int, j: int) -> torch.Tensor:
        return self.inpainting_mask(
            int(i == j == 0),  # Top left masked only for very first quadrant
            int(i == 0),  # Top right if and only if first row
            int(j == 0),  # Bottom left if and only if first column
            1,  # Bottom right always masked
            size="latent",
        )

    def _make_slice_1d(
        self, i_step: int, size: int, stride: int | None = None
    ) -> slice:
        start = i_step * (stride or self.config.stride)
        return slice(start, start + size)

    def _make_slices_2d(
        self, row: int, col: int, size: int, stride: int | None = None
    ) -> tuple[slice, slice]:
        return (
            self._make_slice_1d(row, size, stride=stride),
            self._make_slice_1d(col, size, stride=stride),
        )

    def slices_for_iteration(
        self,
        i: int,
        j: int,
        size: int | None = None,
        resample_first: bool | None = None,
        stride: int | None = None,
    ) -> tuple[slice, slice]:

        # Set default paramerters
        size = size or self.config.latent_size
        resample_first = resample_first or self.config.resample_first

        # Ensure we don't go out of bounds
        # If not resampling first row/column:
        # If idxs are >0, we add 1 to skip the already sampled first row/column
        row = max(i + int(not resample_first and i > 0) - 1, 0)
        col = max(j + int(not resample_first and j > 0) - 1, 0)

        return self._make_slices_2d(row, col, size, stride=stride)

    def slices_for_decoding_iteration(self, i: int, j: int, stride: int | None = None):
        """
        Extracts the latent map based on the indices i and j,
        based on what has been sampled so far.
        """
        # Ensure we don't go out of bounds
        row = max(i - 1, 0)
        col = max(j - 1, 0)

        # We don't need over-sampling for decoding
        stride = stride or self.config.decoding_stride

        return self._make_slices_2d(row, col, self.config.latent_size, stride=stride)

    def blending_patch_left(
        self, overlap: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Set default value for overlap
        overlap = overlap or self.config.decode_overlap

        # Initialize blending patch and its inverse as tensors with same shape
        # as the image size, filled with ones as default value
        # (Non-overlaping image parts will never be masked)
        blending_patch, inv_blending_patch = torch.ones(
            (2, self.config.image_size, self.config.image_size)
        )

        # Construct blend as 1d tensor with smooth transitiion from 0 to 1,
        # Then repeat over the entire image to make it 2d
        blending_pattern = (
            torch.linspace(0, 1, overlap).unsqueeze(0).repeat(self.config.image_size, 1)
        )

        # Fill the patches with the (inverse) blending pattern at the overlap region
        blending_patch[:, :overlap] = blending_pattern
        inv_blending_patch[:, :overlap] = torch.flip(blending_pattern, (1,))
        return blending_patch, inv_blending_patch

    def blending_patch_top(
        self, overlap: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Get left blending patch, then transpose it to get top blending patch
        return tuple(
            torch.transpose(p, 0, 1) for p in self.blending_patch_left(overlap=overlap)
        )

    def blending_patch_corner(
        self, overlap: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Set default value for overlap
        overlap = overlap or self.config.decode_overlap

        # Get left and top blending patches, which will be combined to
        # make the corner blending patch
        patch_left, _ = self.blending_patch_left(overlap=overlap)
        patch_top, _ = self.blending_patch_top(overlap=overlap)

        # Stack them and take the minimum value across the stack to get
        # the corner blending patch
        patch_corner = torch.min(torch.stack((patch_top, patch_left)), axis=0)[0]

        # Get the inverse by applying boolean NOT, then setting the non-overlapping
        # region to 1 (since they will never be masked)
        inv_patch_corner = 1 - patch_corner
        inv_patch_corner[overlap:, overlap:] = 1
        return patch_corner, inv_patch_corner

    def blend_for_decoding_iteration(
        self, i: int, j: int, overlap: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the crossfade patch for the given quadrant indices (i, j).
        """
        match (i, j):

            # Top left quadrant is never blended
            case (1, 1):
                return (
                    torch.ones((self.config.image_size,) * 2),
                    torch.zeros((self.config.image_size,) * 2),
                )

            # Upper row is left-blended
            case (1, j) if j > 1:
                return self.blending_patch_left(overlap=overlap)

            # Left column is top-blended
            case (i, 1) if i > 1:
                return self.blending_patch_top(overlap=overlap)

            # Everything else is corner-blended
            case (i, j) if i > 1 and j > 1:
                return self.blending_patch_corner(overlap=overlap)

            # Anything else is invalid
            case _:
                raise ValueError(
                    f"Invalid quadrant indices ({i}, {j}) for blending patch. "
                )

    def _initialize_intermediates(
        self,
        total_steps_sample: int,
        batch_size: int,
        ctxt_channels: int,
        timesteps: int,
    ) -> dict[str, torch.Tensor | None]:
        return {
            "latents": torch.zeros(
                total_steps_sample,
                batch_size,
                self.config.latent_dim,
                self.config.latent_size,
                self.config.latent_size,
            ),
            "latents_time_steps": torch.zeros(
                total_steps_sample,
                batch_size,
                timesteps + 1,
                self.config.latent_dim,
                self.config.latent_size,
                self.config.latent_size,
            ),
            "inpainting_contexts": torch.zeros(
                total_steps_sample,
                batch_size,
                self.config.latent_dim,
                self.config.latent_size,
                self.config.latent_size,
            ),
            "context": torch.zeros(
                total_steps_sample,
                batch_size,
                ctxt_channels,
                self.config.latent_size,
                self.config.latent_size,
            ),
            "ext_context": (
                torch.zeros(
                    total_steps_sample,
                    batch_size,
                    ctxt_channels,
                    self.config.ext_ctxt_size,
                    self.config.ext_ctxt_size,
                )
                if self.config.f_ext_ctxt > 1
                else None
            ),
            "sampling_masks": torch.zeros(
                total_steps_sample,
                batch_size,
                1,
                self.config.latent_size,
                self.config.latent_size,
            ),
            "sampling_indices": torch.zeros(total_steps_sample, 2, dtype=torch.int),
            "seed_noise": torch.zeros(
                total_steps_sample,
                batch_size,
                self.config.latent_dim,
                self.config.latent_size,
                self.config.latent_size,
            ),
        }

    def _set_map_size(self, sampling_steps: tuple[int, int]) -> tuple[int, int]:
        height_steps, width_steps = sampling_steps
        latent_map_height = (
            self.config.latent_size + (height_steps - 1) * self.config.stride
        )
        latent_map_width = (
            self.config.latent_size + (width_steps - 1) * self.config.stride
        )
        return (latent_map_height, latent_map_width)

    def _check_stride_compatibility(self, h: int, w: int, stride: int) -> bool:
        return (h - self.config.latent_size) % stride == 0 and (
            w - self.config.latent_size
        ) % stride == 0

    def _infer_map_size(
        self, ctxt_map: torch.Tensor, current_map_size: tuple[int, int]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        # Map size:
        h, w = ctxt_map.shape[-2:]
        latent_map_height, latent_map_width = current_map_size

        # If the size of the provided context map does not match the input
        # sampling steps, we try to infer the correct values from the map.
        if (h, w) != (latent_map_height, latent_map_width):
            self.logger.info(
                f"Inferring sampling steps from context map size {ctxt_map.shape[-2:]}."
            )
            # Verify that the shape of provided context map is divisible by stride
            if not self._check_stride_compatibility(h, w, self.config.stride):
                raise ValueError(
                    f"Incompatible context map size {ctxt_map.shape[-2:]} for\n"
                    f"latent size {self.config.latent_size}.\n"
                    f"Context map size must be of the form (latent_size + n*stride, latent_size + m*stride)\n"
                    f"for some integers n, m >= 0., with stride={self.config.stride} derived from\n"
                    f"the requested sampling steps and latent size."
                )

            # Define new latent map dimensions
            latent_map_height, latent_map_width = h, w
        height_steps = (h - self.config.latent_size) // self.config.stride + 1
        width_steps = (w - self.config.latent_size) // self.config.stride + 1
        self.logger.info(
            f"Adjusting sampling steps to ({height_steps}, {width_steps}) to match context map size."
        )
        return (height_steps, width_steps), (latent_map_height, latent_map_width)

    def _prepare_batch_dimension(
        self, x: torch.Tensor, batch_size: int, infer_batch_size: bool = True
    ) -> tuple[torch.Tensor, int]:
        # Handle batch dimension of context map:
        match x.shape:
            # If it has no batch dimension, add one and repeat to match batch size
            case (_, _, _):
                self.logger.info("\tAdding batch dimension.")
                x = x.unsqueeze(0).repeat(batch_size, 1, 1, 1)

            # If batch dimension is 1, repeat to match batch size
            case (1, _, _, _):
                self.logger.info("\tExpanding batch dimension.")
                x = x.repeat(batch_size, 1, 1, 1)

            # If batch dimension matches batch size, great!
            case (batch_size, _, _, _):
                pass

            # If batch dimension is larger than 1 and not batch_size,
            # infer batch_size from context map.
            case (b, _, _, _) if infer_batch_size:
                self.logger.info(f"\tInferring batch size from shape {x.shape}.")
                batch_size = b

            case _:
                raise ValueError(
                    f"Invalid shape {x.shape}. Expected shape (C, H, W), (1, C, H, W), or (batch_size, C, H, W) with batch_size={batch_size}."
                )
        return x, batch_size

    def _make_extended_ctxt(
        self,
        ctxt_map: torch.Tensor,
        ext_ctxt_map: torch.Tensor | None,
    ):
        # Size for zero-padding the context map to make the extended context
        # map.
        padding_width = (self.config.ext_ctxt_size - self.config.latent_size) // 2

        # If not provided, construct from context map by padding
        if ext_ctxt_map is None:
            self.logger.info("Making extended context map from context map by padding")
            ext_ctxt_map = ctxt_map
            ext_ctxt_map = torch.nn.functional.pad(
                ext_ctxt_map,
                (padding_width,) * 4,
                mode="constant",
                value=0,
            )

        # Assert that the sizes are compatible
        assert all(
            [
                ext_ctxt_map.shape[i] == ctxt_map.shape[i] + 2 * padding_width
                for i in [-1, -2]
            ]
        ), f"Incompatible shapes between context map ({ctxt_map.shape}) and extended context map ({ext_ctxt_map.shape})"

        # Set zero-values in location flag to -1
        ext_ctxt_map[:, 0][ext_ctxt_map[:, 0] == 0] = -1

        # Handle batch dimension
        ext_ctxt_map, _ = self._prepare_batch_dimension(
            ext_ctxt_map, batch_size=ctxt_map.shape[0], infer_batch_size=False
        )
        return ext_ctxt_map

    def sampling_inputs_for_iteration(
        self, row, col, ctxt_map, ext_ctxt_map, latent_map, seed_noise_map, catalog_mode
    ):
        # Create mask for the current sample
        # By default, mask has shape (1, 1, latent_q_size, latent_q_size),
        # but we repeat it to match the batch size
        batch_size = ctxt_map.shape[0]
        inpainting_mask = (
            self.inpainting_mask_for_iteration(
                *((row, col) if self.config.do_inpainting else (0, 0))
            )
            .to(self.config.device)
            .repeat(batch_size, 1, 1, 1)
        )

        # Get the latent map slice for the current sample
        slc_row, slc_col = self.slices_for_iteration(row, col)

        # Grab the inpainting context for the current sample, which might
        # contain parts of previously sampled regions.
        inpainting_latent = latent_map[:, :, slc_row, slc_col].clone()
        assert inpainting_latent.shape[-2:] == inpainting_mask.shape[-2:], (
            f"Mask shape {inpainting_mask.shape} does not match latent map slice shape "
            f"{inpainting_latent.shape} for sampling step ({row}, {col})."
        )

        # Grab seed noise for the current sample
        seed_noise = seed_noise_map[:, :, slc_row, slc_col]

        # Grab the context and extend context if present
        ctxt_vector, ext_ctxt_vector = None, None
        if ctxt_map is not None:

            # Grab regular context map only in separate mode
            if catalog_mode == "separate":
                ctxt_vector = ctxt_map[:, :, slc_row, slc_col]

            # Grab extended context map
            if self.config.f_ext_ctxt > 1 or catalog_mode == "combined":
                ext_slc_row, ext_slc_col = self.slices_for_iteration(
                    row,
                    col,
                    size=self.config.ext_ctxt_size,
                )
                ext_ctxt_vector = ext_ctxt_map[:, :, ext_slc_row, ext_slc_col].clone()

                # Zero-out center only in separate mode
                if catalog_mode == "separate":
                    zero_padw = self.config.ext_ctxt_size // 4
                    ext_ctxt_vector[
                        :, :, zero_padw:-zero_padw, zero_padw:-zero_padw
                    ] *= 0

        return (
            (slc_row, slc_col),
            inpainting_mask,
            inpainting_latent,
            seed_noise,
            ctxt_vector,
            ext_ctxt_vector,
        )

    @torch.no_grad()
    @with_temporary_settings
    def sample(
        self,
        ctxt_map: torch.Tensor | None = None,
        seed_noise_map: torch.Tensor | None = None,
        sampling_steps: tuple[int, int] = (2, 2),
        ext_ctxt_map: torch.Tensor | None = None,
        batch_size: int = 1,
        save_intermediates: bool = False,
        # TODO: These should be part of another config for sampler/denoiser
        catalog_mode="combined",
        drop_inpainting_ctxt_at=-1,
        use_inpainting_replacement=True,
        timesteps=25,
        **dm_sampling_kw,
    ):

        # At least the denoiser and VAE need to be available for sampling.
        if any([model is None for model in [self.denoiser, self.vae]]):
            raise RuntimeError(
                f"Denoiser or VAE model is None: Denoiser={self.denoiser is None}, VAE={self.vae is None}. "
            )

        # Check validity of catalog mode
        valid_catalog_modes = ["combined", "separate"]
        assert (
            catalog_mode in valid_catalog_modes
        ), f"Invalid catalog_mode <{catalog_mode}>, must be one of: {valid_catalog_modes}."

        # Set latent map dimensions based on sampling steps and latent size
        height_steps, width_steps = sampling_steps
        latent_map_height, latent_map_width = self._set_map_size(sampling_steps)

        # Prepare context map
        self.logger.info("Preparig context map...")
        if ctxt_map is not None:
            # Try to infer sampling steps and latent map size from context map
            (height_steps, width_steps), (latent_map_height, latent_map_width) = (
                self._infer_map_size(ctxt_map, (latent_map_height, latent_map_width))
            )

            # Handle batch dimension of context map:
            ctxt_map, batch_size = self._prepare_batch_dimension(
                ctxt_map, batch_size, infer_batch_size=True
            )
        # If no context map is provided, set to zeros
        else:
            ctxt_map = torch.zeros(
                (
                    batch_size,
                    4,
                    latent_map_height,
                    latent_map_width,
                ),
                device=self.config.device,
            )

        # Check that sampling steps are at least 2
        assert min(sampling_steps) >= 2, "Sampling steps must be at least 2."

        # Make sure that the decoding_stride is compatible with the sampling steps and latent size
        if not self._check_stride_compatibility(
            latent_map_height, latent_map_width, self.config.decoding_stride
        ):
            raise ValueError(
                f"Decoding stride {self.config.decoding_stride} is not compatible with latent map size ({latent_map_height}, {latent_map_width}) and latent size {self.config.latent_size}.\n"
                f"Ensure that (map_size - latent_size) is divisible by decoding_stride for both dimensions."
            )

        # Make extended context map if required
        if self.config.f_ext_ctxt > 1:
            ext_ctxt_map = self._make_extended_ctxt(ctxt_map, ext_ctxt_map)
        # If we use combined catalog encoding with no extended context, we just
        # pass the regular context map to the catalog context encoder
        elif catalog_mode == "combined":
            ext_ctxt_map = ctxt_map

        # Info message
        self.logger.info(
            f"Sampling {sampling_steps[0]}x{sampling_steps[1]} steps "
            f"with latent size {self.config.latent_size} and image size {self.config.image_size}"
        )

        # Define total sampling steps
        total_steps = height_steps * width_steps

        # If we sample with smaller stride (latent_size - mask_overlap) / div_stride,
        # there are more in-between steps that are sampled.
        height_steps_sample = (height_steps - 2) * self.config.div_stride + 2
        width_steps_sample = (width_steps - 2) * self.config.div_stride + 2
        # We need to add an additional step if we want to double-sample the
        # first row/column
        if self.config.resample_first:
            height_steps_sample += 1
            width_steps_sample += 1

        # These is the total number of actual sampling steps
        total_steps_sample = height_steps_sample * width_steps_sample

        self.logger.divider()
        self.logger.info("Sample latent map...")

        # Initialize empty tensor for latents map
        latent_map = torch.zeros(
            batch_size,
            self.config.latent_dim,
            latent_map_height,
            latent_map_width,
        ).to(self.config.device)

        # Initialize seed noise for consistent sampling
        if seed_noise_map is None:
            seed_noise_map = torch.randn_like(latent_map)
        assert (
            seed_noise_map.shape == latent_map.shape
        ), f"Seed noise map has shape {seed_noise_map.shape}, but latent map has shape {latent_map.shape}."
        seed_noise_map = seed_noise_map.to(self.config.device)

        # Initialize dict for saving intermediate results
        if save_intermediates:
            intermediates = self._initialize_intermediates(
                total_steps_sample, batch_size, ctxt_map.shape[1], timesteps
            )

        # Move denoiser to device
        self.denoiser.to(self.config.device)
        if self.uncond_denoiser is not None:
            self.uncond_denoiser.to(self.config.device)

        # Initialize progress bar
        pbar = tqdm(
            total=total_steps_sample, desc="Sampling Latent Map", dynamic_ncols=True
        )

        # Start DM sampling loop
        for step, (row, col) in enumerate(
            product(range(height_steps_sample), range(width_steps_sample))
        ):

            # Set the description of the progress bar to include the current sampling step
            pbar.set_description(f"Sampling iteration ({row}, {col})")

            # Get sampling inputs
            (
                (slc_row, slc_col),
                inpainting_mask,
                inpainting_latent,
                seed_noise,
                ctxt_vector,
                ext_ctxt_vector,
            ) = self.sampling_inputs_for_iteration(
                row,
                col,
                ctxt_map,
                ext_ctxt_map,
                latent_map,
                seed_noise_map,
                catalog_mode,
            )

            # Sample latent image
            sampled_latent = self.sampler.quick_sample(
                model=self.denoiser,
                uncond_model=self.uncond_denoiser,
                inpainting_context=(inpainting_mask, inpainting_latent),
                img_context=ctxt_vector,
                catalog_context=ext_ctxt_vector,
                distribute_model=False,
                rescale_images=False,
                quiet=True,
                image_channels=self.config.latent_dim,
                seed_noise=seed_noise,
                return_steps=save_intermediates,
                use_inpainting_replacement=use_inpainting_replacement,
                drop_inpainting_ctxt_at=drop_inpainting_ctxt_at,
                timesteps=timesteps,
                **dm_sampling_kw,
            )
            sampled_latent = torch.from_numpy(sampled_latent)

            # Save to intermediates if required
            if save_intermediates:
                intermediates["latents_time_steps"][step] = sampled_latent
                # If we save intermediates, timesteps are returned, so we remove those
                sampled_latent = sampled_latent[:, -1]

            # Move sampled latent to same device as latent map
            sampled_latent = sampled_latent.to(self.config.device)

            # Clear the space for the new sample
            latent_map[:, :, slc_row, slc_col] *= 1 - inpainting_mask
            # Place the new sample in the latent map
            latent_map[:, :, slc_row, slc_col] += (
                sampled_latent * inpainting_mask
            ).squeeze()

            # Save intermediates if required
            if save_intermediates:
                intermediates["latents"][step] = sampled_latent.cpu()
                intermediates["inpainting_contexts"][step] = inpainting_latent.cpu()
                if ctxt_vector is not None:
                    intermediates["context"][step] = ctxt_vector.cpu()
                if ext_ctxt_vector is not None:
                    intermediates["ext_context"][step] = ext_ctxt_vector.cpu()
                intermediates["sampling_masks"][step] = inpainting_mask.cpu()
                intermediates["sampling_indices"][step] = torch.tensor([row, col])
                intermediates["seed_noise"][step] = seed_noise.cpu()

            # Update the progress bar
            pbar.update(1)

        self.logger.info("Sampling latents completed.")
        self.logger.divider()
        self.logger.info("Decoding latents...")

        # Allocate models
        self.denoiser.to("cpu")
        self.vae.to(self.config.device)

        # Decoding might have different number of steps, if decoding_stride is
        # set to a different value
        if self.config.decoding_stride != self.config.stride:
            height_steps = (
                latent_map_height - self.config.latent_size
            ) // self.config.decoding_stride + 1
            width_steps = (
                latent_map_width - self.config.latent_size
            ) // self.config.decoding_stride + 1
        total_steps = height_steps * width_steps

        # Initialize progress bar
        pbar = tqdm(total=total_steps, desc="Decoding latents", unit="step")

        # Initialize empty tensor for the final decoded image,
        # which will be assembled through crossfading inside the loop
        map_image = torch.zeros(
            batch_size,
            latent_map_height * self.config.f_vae,
            latent_map_width * self.config.f_vae,
        )

        # Iterate over steps again, now to decode the latent images.
        # We can start at 1 because no double-sampling is needed for decoding.
        for step, (row, col) in enumerate(
            product(range(1, height_steps + 1), range(1, width_steps + 1))
        ):
            # Get the latent map slice for the current sample
            slc_row, slc_col = self.slices_for_decoding_iteration(row, col)

            # Get the sampled latent image for the current sample
            latent = latent_map[:, :, slc_row, slc_col]

            # Decode the sampled latent image to pixel space
            with torch.no_grad():
                sampled_image = self.vae.decode_code(latent).cpu().squeeze()

            # Get slices of image location on the final map product
            slc_row_img, slc_col_img = self._make_slices_2d(
                row - 1, col - 1, self.config.image_size, stride=self.config.img_stride
            )

            # Make a temporary empty image for blended adding
            sampled_image_step = torch.zeros(
                batch_size,
                latent_map_height * self.config.f_vae,
                latent_map_width * self.config.f_vae,
            )
            # Place the sampled image in the corresponding location
            sampled_image_step[:, slc_row_img, slc_col_img] = sampled_image

            # Add the sampled image to the map image thorugh blending
            blend_patch, blend_inv_patch = self.blend_for_decoding_iteration(row, col)

            # Blend patch has size of image, blend has size of map
            blend_sampled_step = torch.ones_like(map_image)
            blend_sampled_step[:, slc_row_img, slc_col_img] = blend_patch.repeat(
                batch_size, 1, 1
            )
            blend_previous_image = torch.ones_like(map_image)
            blend_previous_image[:, slc_row_img, slc_col_img] = blend_inv_patch.repeat(
                batch_size, 1, 1
            )

            # Blend-add the new sampled image to the map
            map_image = (
                map_image * blend_previous_image
                + sampled_image_step * blend_sampled_step
            )

            # Update progress bar and sampling step index
            pbar.update(1)

        pbar.close()

        # If desired, decode and save intermediates
        if save_intermediates:
            self.logger.info("Decoding intermediates...")
            intermediates["decoded_images"] = torch.zeros(
                total_steps_sample,
                batch_size,
                1,
                self.config.image_size,
                self.config.image_size,
            )
            for idx in tqdm(
                range(intermediates["latents"].shape[0]),
                desc="Decoding intermediates",
                dynamic_ncols=True,
            ):
                latent = intermediates["latents"][idx].to(self.config.device)
                decoded_img = self.vae.decode_code(latent).cpu()
                intermediates["decoded_images"][idx] = decoded_img

        # Move everything back to CPU
        self.vae.cpu()

        if save_intermediates:
            return (
                map_image.numpy(),
                latent_map.cpu().numpy(),
                intermediates,
            )

        return map_image.numpy(), latent_map.cpu().numpy()
