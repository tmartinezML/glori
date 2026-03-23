import sys

import numpy as np
from tqdm import tqdm
import torch
import numpy as np

import utils.my_logging
from utils.devices import is_jupyter

if is_jupyter():
    # print(__name__, " - check")
    from tqdm.auto import tqdm
else:
    # print(__name__, " - check false")
    from tqdm import tqdm

logger = utils.my_logging.get_logger(__name__)


@torch.no_grad()
def edm_sampling(
    model,
    img_context_batch=None,
    context_batch=None,
    catalog_context_batch=None,
    label_batch=None,
    seed_noise=None,
    inpainting_context=None,
    *,
    image_size=80,
    image_channels=None,
    batch_size=16,
    timesteps=25,
    guidance_strength=0.1,
    use_inpainting_replacement=True,
    drop_inpainting_ctxt_at=-1,
    sigma_min=2e-3,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=torch.inf,
    S_noise=1,
    uncond_model=None,
    quiet=False,
):
    """
    Perform deterministic or stochastic sampling from EDM paper (arXiv:2206.00364).
    Setting S_churn = 0 results in deterministic sampling.

    Parameters
    ----------
    model : torch.nn.Module
        The energy-based model.
    context_batch : array_like, optional
        The context batch. Defaults to None.
    label_batch : array_like, optional
        The label batch. Defaults to None.
    latents : array_like, optional
        The seed gaussian noise images. Defaults to None.
    image_size : int, optional
        The size of the image. Defaults to 80.
    batch_size : int, optional
        The batch size. Defaults to 16.
    timesteps : int, optional
        The number of sampling timesteps. Defaults to 25.
    guidance_strength : numeric, optional
        The guidance strength 'omega' parameter. Can be any numeric type. Defaults to 0.1.
    sigma_min : numeric, optional
        The minimum noise level. Can be any numeric type. Defaults to 2e-3.
    sigma_max : numeric, optional
        The maximum noise level. Can be any numeric type. Defaults to 80.
    rho : numeric, optional
        The value of rho. Defaults to 7.
    S_churn : numeric, optional
        The value of S_churn. Defaults to 0.
    S_min : numeric, optional
        The minimum value of S. Can be any numeric type. Defaults to 0.
    S_max : numeric, optional
        The maximum value of S. Can be any numeric type. Defaults to torch.inf.
    S_noise : numeric, optional
        The value of S_noise. Defaults to 1.

    Returns
    -------
    list
        A list of image batches, where every entry correponds to one time step. Batches have shape (batch_size, 1, image_size, image_size).

    """

    # Set device and log
    if quiet:
        logger.setLevel(30)
    device = next(model.parameters()).device
    do_parallel_sampling = isinstance(model, torch.nn.DataParallel)
    if do_parallel_sampling:
        logger.info(f"Parallel sampling on devices: {model.device_ids}")
    else:
        logger.info(f"Sampling on device: {device}")
    if uncond_model is not None:
        logger.info("Separate unconditioned model will be used for guided sampling.")
        if guidance_strength == 0:
            logger.warning(
                "Guidance strength is zero, unconditioned model will not be used."
            )
        elif guidance_strength == -1:
            logger.warning(
                "Guidance strength is -1, unconditioned model will be used with no guidance."
            )
        if (uncdv := next(uncond_model.parameters()).device) != device:
            logger.warning(
                f"Unconditioned model is not on the same device ({uncdv}) as the main model ({device}). "
                "Will move it to this device. This may lead to GPU memory issues."
            )
            uncond_model = uncond_model.to(device)

    # This function will be used on additional inputs like context, labels etc.
    def prepare_tensor(tensor, name="tensor", dtype=torch.float32):
        """
        Prepare a tensor for sampling by moving it to the device and ensuring it's a torch tensor.
        """
        assert tensor.shape[0] == batch_size, (
            f"{name} batch size ({tensor.shape[0]})"
            f"must match batch size ({batch_size})."
        )
        if type(tensor) != torch.Tensor:
            tensor = torch.tensor(tensor, device=device, dtype=dtype)
        return tensor.to(device)

    # If passed, prepare context and labels
    if img_context_batch is not None:
        img_context_batch = prepare_tensor(
            img_context_batch, name="Image context batch"
        )

    if catalog_context_batch is not None:
        catalog_context_batch = prepare_tensor(
            catalog_context_batch, name="Catalog context batch"
        )

    if context_batch is not None:
        context_batch = prepare_tensor(context_batch, name="Context batch")

    if label_batch is not None:
        label_batch = prepare_tensor(label_batch, name="Label batch", dtype=torch.long)

    # If passed, prepare mask input
    if inpainting_context is not None:
        inpainting_mask, inpainting_img = inpainting_context
        inpainting_mask = prepare_tensor(
            inpainting_mask, name="Inpainting mask", dtype=torch.float32
        )
        inpainting_img = prepare_tensor(
            inpainting_img, name="Inpainting image", dtype=torch.float32
        )
        masked_inpainting_img = inpainting_img * (
            1 - inpainting_mask
        )  # Masked image for the masked pixels

        if img_context_batch is not None:
            # Concatenate mask to image context
            img_context_batch = torch.cat(
                (img_context_batch, masked_inpainting_img, inpainting_mask), dim=1
            ).to(device)
        else:
            # Use mask as image context
            img_context_batch = torch.cat(
                (masked_inpainting_img, inpainting_mask), dim=1
            ).to(device)

        if drop_inpainting_ctxt_at > timesteps:
            logger.info(
                f"Value for drop_inpainting_ctxt {drop_inpainting_ctxt_at} is larger than timesteps ({timesteps}). Capping at 25, which means inpainting context will not be used."
            )
            drop_inpainting_ctxt_at = timesteps

    # If passed, prepare latents
    if seed_noise is not None:
        expected_channels = (
            image_channels
            if image_channels is not None
            else (
                int(model.in_channels - img_context_batch.shape[1])
                if img_context_batch is not None
                else model.in_channels
            )
        )
        assert (
            seed_noise.shape[1] == expected_channels
        ), f"Latents must have {expected_channels} channels."
        assert (
            seed_noise.shape[2] == image_size
        ), f"Latents must have size {image_size}x{image_size}."
        batch_size = seed_noise.shape[0]
        # Make sure it's torch tensor
        seed_noise = prepare_tensor(seed_noise, name="Seed noise")

    # If not passed, sample latents from normal distribution.
    # Scaling will happen before the first sampling loop.
    else:
        if image_channels is None:
            image_channels = (
                int(model.in_channels - img_context_batch.shape[1])
                if img_context_batch is not None
                else model.in_channels
            )
        seed_noise = torch.randn(
            [
                batch_size,
                image_channels,
                image_size,
                image_size,
            ],
            device=device,
        )

    # Get noise level limits from model, which may be more restrictive
    model_sigma_min = (
        model.module.sigma_min if do_parallel_sampling else model.sigma_min
    )
    model_sigma_max = (
        model.module.sigma_max if do_parallel_sampling else model.sigma_max
    )

    # Update noise level limits
    sigma_min = max(sigma_min, model_sigma_min)
    sigma_max = min(sigma_max, model_sigma_max)

    # Generate time steps (= noise levels).
    sigma_steps = get_sampling_noise_levels(
        timesteps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho
    )

    # Move sigmas to gpu
    sigma_steps = sigma_steps.to(device)

    # Prepare sampling loop.
    imgs = []
    denoiser_outputs = []
    x_next = seed_noise * sigma_steps[0]  # Generate initial sample at t_0
    inpainting_dropped = False
    if (
        inpainting_context is not None
        and use_inpainting_replacement
        and drop_inpainting_ctxt_at != timesteps
    ):
        # Apply mask to initial sample
        x_next = x_next * inpainting_mask + (
            inpainting_img + seed_noise * sigma_steps[0]
        ) * (1 - inpainting_mask)

    imgs.append(x_next.cpu())

    # Sampling loop:
    for i, (sigma_cur, sigma_next) in tqdm(
        enumerate(zip(sigma_steps[:-1], sigma_steps[1:])),
        desc="Sampling...",
        total=timesteps,
        leave=not quiet,
        file=sys.stdout if quiet else sys.stderr,
        dynamic_ncols=True,
        # position=int(quiet),
    ):
        # Update current image (= output from previous iteration)
        x_cur = x_next

        if drop_inpainting_ctxt_at == timesteps - i and inpainting_context is not None:
            logger.info(
                f"Dropping inpainting context at {i=} (timestep {timesteps - i})."
            )
            # Set mask to ones and ctxt to zero
            img_context_batch[:, -1] = 1
            img_context_batch[:, -(image_channels + 1) : -1] = 0
            inpainting_dropped = True

        # Stochastic sampling: Increase noise temporarily
        if S_churn > 0:
            sigma_cur, x_cur = stochastic_churn(
                timesteps, S_churn, S_min, S_max, S_noise, sigma_cur, x_cur
            )

        # Calculate denoised image with forward model pass
        denoised = denoised_guided(
            model,
            x_cur,
            sigma_cur,
            img_context=img_context_batch,
            catalog_context=catalog_context_batch,
            context=context_batch,
            class_labels=label_batch,
            guidance_strength=guidance_strength,
            uncond_model=uncond_model,
        )
        denoiser_outputs.append(denoised.cpu().detach())

        # Score estimate
        d_cur = (x_cur - denoised) / sigma_cur

        # Euler step
        x_next = x_cur + d_cur * (sigma_next - sigma_cur)

        # Apply inpainting replacement to current image
        if (
            inpainting_context is not None
            and use_inpainting_replacement
            and not inpainting_dropped
        ):
            x_next = x_next * inpainting_mask + (
                inpainting_img + seed_noise * sigma_next
            ) * (1 - inpainting_mask)

        # Apply 2nd order correction
        if i < timesteps - 1:

            # Denoised image for next step
            denoised = denoised_guided(
                model,
                x_next,
                sigma_next,
                img_context=img_context_batch,
                catalog_context=catalog_context_batch,
                context=context_batch,
                class_labels=label_batch,
                guidance_strength=guidance_strength,
                uncond_model=uncond_model,
            )

            # Score estimate for next step
            d_next = (x_next - denoised) / sigma_next

            # 2nd order correction by applying trapezoidal rule
            x_next = x_cur + (sigma_next - sigma_cur) * (0.5 * d_cur + 0.5 * d_next)

            # Apply inpainting replacement to current image
            if (
                inpainting_context is not None
                and use_inpainting_replacement
                and not inpainting_dropped
            ):
                x_next = x_next * inpainting_mask + (
                    inpainting_img + seed_noise * sigma_next
                ) * (1 - inpainting_mask)

        # Append to list
        imgs.append(x_next.cpu())

    return imgs, denoiser_outputs


def get_sampling_noise_levels(timesteps, sigma_min=2e-3, sigma_max=80, rho=7):
    """
    Generate noise levels for each sampling step, according to the scheme proposed
    in the EDM paper (arXiv:2206.00364).

    Parameters
    ----------
    timesteps : int
        The number of sampling steps.
    sigma_min : numeric, optional
        The minimum noise level. Defaults to 2e-3.
    sigma_max : numeric, optional
        The maximum noise level. Defaults to 80.
    rho : numeric, optional
        The value of rho. Defaults to 7.

    Returns
    -------
    tuple
        A tuple containing the time step indices and the noise levels.
    """
    # Time step indices
    step_inds = torch.arange(timesteps)

    # Noise level for each time step
    rho_inv = 1 / rho
    sigma_steps = (
        sigma_max**rho_inv
        + step_inds / (timesteps - 1) * (sigma_min**rho_inv - sigma_max**rho_inv)
    ) ** rho

    # Add t_N=0 at the end
    sigma_steps = torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])])

    return sigma_steps


@torch.no_grad()
def denoised_guided(
    model,
    img,
    sigma,
    img_context=None,
    catalog_context=None,
    context=None,
    class_labels=None,
    guidance_strength=0.1,
    uncond_model=None,
):
    """
    Calculate the denoised image. Guidance is applied if context or class labels are passed.

    Parameters
    ----------
    model : torch.nn.Module
        The energy-based model.
    img : torch.Tensor
        The input image.
    sigma : torch.Tensor
        The noise level.
    context : torch.Tensor, optional
        The context tensor. Defaults to None.
    class_labels : torch.Tensor, optional
        The class labels. Defaults to None.
    guidance_strength : numeric, optional
        The guidance strength 'omega' parameter. Can be any numeric type. Defaults to 0.1.

    Returns
    -------
    torch.Tensor
        The denoised image.
    """
    # Set batch size
    batch_size = img.shape[0]
    sigma = sigma.expand(batch_size)

    uncond_model = uncond_model if uncond_model is not None else model

    # Calculate unconditioned denoised image with forward model pass
    if guidance_strength != 0:
        denoised = uncond_model(
            img, sigma, img_context=None, context=None, catalog_context=None
        )
    else:
        denoised = torch.zeros_like(img).to(img.device)

    # Calculate denoised image with conditioning if any context is provided
    if (
        any(
            [
                c is not None
                for c in [img_context, context, catalog_context, class_labels]
            ]
        )
        and guidance_strength != -1
    ):

        # Denoised image with conditioning
        denoised_cond = model(
            img,
            sigma,
            img_context=img_context,
            catalog_context=catalog_context,
            class_labels=class_labels.long() if class_labels is not None else None,
            context=context,
        )

        # Apply guidance
        denoised = (
            1 + guidance_strength
        ) * denoised_cond - guidance_strength * denoised

    return denoised


def stochastic_churn(timesteps, S_churn, S_min, S_max, S_noise, sigma_cur, x_cur):
    """
    Add stochastic churn for stochastic sampling, i.e. temporarily increase noise level.

    Parameters
    ----------
    timesteps : int
        The number of timesteps.
    S_churn : int
        The value of S_churn.
    S_min : int
        The minimum value of S.
    S_max : torch.Tensor
        The maximum value of S.
    S_noise : int
        The value of S_noise.
    sigma_cur : torch.Tensor
        The current noise level.
    x_cur : torch.Tensor
        The current image.

    Returns
    -------
    tuple
        A tuple containing the updated noise level and image.
    """
    # Factor by which noise level is increased (gamma = 0 if S_churn = 0)
    gamma = (
        min(S_churn / timesteps, np.sqrt(2) - 1)
        if S_min <= sigma_cur.item() <= S_max
        else 0
    )

    # Increase noise level (sigma_hat = sigma_cur if gamma = 0).
    sigma_hat = (1 + gamma) * sigma_cur

    return sigma_hat, x_cur
