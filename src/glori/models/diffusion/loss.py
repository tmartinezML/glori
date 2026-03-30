import torch


def sample_sigmas(
    img_batch,
    P_mean=-1.2,
    P_std=1.2,
):
    """
    Sample noise levels from a log-normal distribution. used during training for
    adding noise to the input images.

    Parameters
    ----------
    img_batch : torch.Tensor
        Input image batch, used to infer shape.
    P_mean : float, optional
        log(mean) parameter for the log-normal distribution, by default -1.2
    P_std : float, optional
        log(std) parameter for the log-normal distribution, by default 1.2

    Returns
    -------
    _type_
        _description_
    """
    rnd_normal = torch.randn([img_batch.shape[0], 1, 1, 1], device=img_batch.device)
    sigmas = (rnd_normal * P_std + P_mean).exp()
    return sigmas


def edm_loss(
    model,
    img_batch,
    sigma_data=0.5,
    P_mean=-1.2,
    P_std=1.2,
    mask=None,
    sigmas=None,
    noise=None,
    context=None,
    img_context=None,
    catalog_context=None,
    class_labels=None,
    return_output=False,
    mean=True,
):
    """
    Calculates the EDM (Expected Denoising MSE) loss between the denoised image and the original image.

    Parameters
    ----------
    model : nn.Module
        The denoising model used for denoising the image.
    img_batch : torch.Tensor
        The batch of input images to be denoised.
    sigma_data : float, optional
        The assumed standard deviation of the noise in the training data, by default 0.5.
    P_mean : float, optional
        The log-mean of the log-normal distribution used for sampling sigmas, by default -1.2.
    P_std : float, optional
        The log-standard deviation of the log-normal distribution used for sampling sigmas, by default 1.2.
    rec_reference : torch.Tensor, optional
        The reference image to compute the loss against, by default None.
        If None, the input image batch is used as the reference.
    sigmas : torch.Tensor, optional
        The noise levels for each image in the batch, by default None.
        If None, they are sampled from a log-normal distribution.
    noise : torch.Tensor, optional
        The noise vector to be added to the input images, by default None.
        If None, it is sampled from a normal distribution with noise levels given by 'sigmas'.
    context : object, optional
        The context information for the denoising model, by default None.
    class_labels : object, optional
        The class labels for the input images, by default None.
    return_output : bool, optional
        Whether to return the denoised image along with the loss, by default False.
    mean : bool, optional
        Whether to compute the mean loss across the batch, by default True.

    Returns
    -------
    torch.Tensor or tuple
        If `return_output` is True, returns a tuple containing the loss and the denoised image.
        If `return_output` is False, returns only the loss.

    Raises
    ------
    AssertionError
        If `noise` is provided but `sigmas` is not provided.

    Notes
    -----
    The EDM loss is calculated as the weighted mean squared error between the denoised image and the original image.
    The weight coefficient for the loss is computed based on the noise levels and the standard deviation of the noise in the input images.
    The denoised image is obtained by adding the noise vector to the input images and passing them through the denoising model.
    """

    # Assert mask has adequate shape
    if mask is not None:
        assert (
            mask.shape[0] == img_batch.shape[0]
            and (mask.shape[1] in [img_batch.shape[1], 1])
            and mask.shape[2:] == img_batch.shape[2:]
        ), f"Mask shape {mask.shape} incompatible with image batch shape {img_batch.shape}."
        mask = mask.to(img_batch.device)

    # Set noise vector
    if noise is not None:
        assert sigmas is not None, "If noise is provided, sigmas must be provided."
        n = noise
    else:
        sigmas = sigmas or sample_sigmas(img_batch, P_mean, P_std)
        n = torch.randn_like(img_batch) * sigmas

    # Apply mask to the context vector
    if mask is not None:
        # mask == 1 means that particular part of the image will be sampled.
        # The part that should be sampled
        # is therefore removed from the context, hence using the inverted mask.
        masked_img = img_batch * (1 - mask)

        # Mask should also be passed as image context:
        if img_context is not None:
            img_context = torch.cat((img_context, masked_img, mask), dim=1).to(
                img_batch.device
            )
        else:
            img_context = torch.cat((masked_img, mask), dim=1).to(img_batch.device)

    # Compute denoised image with forward model pass
    D_yn = model(
        img_batch + n,
        sigmas,
        context=context,
        catalog_context=catalog_context,
        class_labels=class_labels,
        img_context=img_context,
    )

    # Compute loss
    if mask is not None:
        # Apply mask to the reference image and denoised image,
        # so that only the masked pixels contribute to the loss
        # TODO: Implement switch
        # img_batch = img_batch * mask
        # D_yn = D_yn * mask
        pass

    # Weight coefficient for loss, as introduced in EDM paper
    weight = (sigmas**2 + sigma_data**2) / (sigmas * sigma_data) ** 2
    loss = weight * (D_yn - img_batch) ** 2

    if mean:
        if mask is not None:
            # If mask is provided, compute mean only over active pixels
            loss = loss.sum() / mask.sum()
            if mask.shape[1] < img_batch.shape[1]:
                # If mask has fewer channels than the image batch, average over channels
                loss /= img_batch.shape[1]
        else:
            # If no mask is provided, compute mean over all pixels
            loss = loss.mean()

    return (loss, D_yn) if return_output else loss
