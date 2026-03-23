import torch

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF

from data.trf.functional import *


def ExtendedCatalogContextTransform(
    zero_center_frac=2, random_crop=False, scale_fn=None
):
    transform = T.Compose(
        [
            T.Lambda(torch.from_numpy),  # Get the first element from tuple
            T.Lambda(catalog_context_pos_rescale),  # Rescale pos. encoding to [-1, 1]
            T.Lambda(scale_fn) if scale_fn is not None else T.Identity(),
            T.RandomCrop(random_crop) if random_crop else T.Identity(),
            (
                T.Lambda(lambda img: zero_center(img, zero_center_frac))
                if zero_center is not None
                else T.Identity()
            ),
        ]
    )
    return transform


def ModelContextTransform(scale_fn=None):
    transform = T.Compose(
        [
            T.Lambda(torch.from_numpy),  # Get the first element from tuple
            T.Lambda(single_channel),  # Exactly one channel
            T.Lambda(lambda t: torch.clamp(t, 0, None) * 1e3),
            T.Lambda(scale_fn) if scale_fn is not None else T.Identity(),
        ]
    )
    return transform


def CatalogContextTransform(random_crop=False, scale_fn=None):
    transform = T.Compose(
        [
            T.Lambda(torch.from_numpy),  # Get the first element from tuple
            T.Lambda(catalog_context_pos_rescale),  # Rescale pos. encoding to [-1, 1]
            T.Lambda(scale_fn) if scale_fn is not None else T.Identity(),
            T.RandomCrop(random_crop) if random_crop else T.Identity(),
        ]
    )
    return transform


def EncodingTransformLDM(scale_fn=None):
    transform = T.Compose(
        [
            T.Lambda(torch.from_numpy),
            T.Lambda(half_size_random_crop),
            T.Lambda(scale_fn) if scale_fn is not None else T.Identity(),
        ]
    )
    return transform


def WebdatasetTransformRaw(scale_fn=None):
    transform = T.Compose(
        [
            T.Lambda(torch.from_numpy),
            T.Lambda(scale_fn) if scale_fn is not None else T.Identity(),
        ]
    )
    return transform


def WebdatasetTransformCrop(scale_fn=None, crop_size=512, random_crop=False):
    transform = T.Compose(
        [
            T.Lambda(torch.from_numpy),
            (T.RandomCrop if random_crop else T.CenterCrop)(crop_size),
            T.Lambda(scale_fn) if scale_fn is not None else T.Identity(),
            T.Lambda(add_channel_dim),
        ]
    )
    return transform


def MicromapTransformTest(scale_fn=None):
    transform = T.Compose(
        [
            T.Lambda(torch.from_numpy),
            T.Lambda(single_channel),  # Exactly one channel
            T.CenterCrop(80),
            T.Lambda(scale_fn) if scale_fn is not None else T.Identity(),
        ]
    )
    return transform


def MicroMapTransformVAE(scale_fn=None):
    transform = T.Compose(
        [
            T.Lambda(torch.from_numpy),
            T.Lambda(single_channel),  # Exactly one channel
            T.Lambda(half_size_random_crop),
            T.Lambda(scale_fn) if scale_fn is not None else T.Identity(),
        ]
    )
    return transform


def MicroMapTransformLDM(scale_fn=None):
    transform = T.Compose(
        [
            T.Lambda(torch.from_numpy),  # Get the first element from tuple
            T.Lambda(single_channel),  # Exactly one channel
            T.Lambda(scale_fn) if scale_fn is not None else T.Identity(),
        ]
    )
    return transform


def ToTensor():
    transform = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32),
        ]
    )
    return transform


def TrainTransform(image_size):
    transform = T.Compose(
        [
            T.CenterCrop(image_size),
            T.Lambda(single_channel),  # Exactly one channel
            T.Lambda(minmax_scale),  # Scale to [0, 1]
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Lambda(random_rotate_90),
            T.Lambda(train_scale),  # Scale to [-1, 1]
        ]
    )
    return transform


def TrainTransformPrototypesWDS(image_size):
    transform = T.Compose(
        [
            T.Lambda(torch.from_numpy),
            T.Lambda(minmax_scale),  # Scale to [0, 1]
            T.CenterCrop(image_size),
            T.Lambda(single_channel),  # Exactly one channel
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(180, interpolation=TF.InterpolationMode.BILINEAR),
            T.Lambda(train_scale),  # Scale to [-1, 1]
        ]
    )
    return transform


def EvalTransformPrototypesWDS(image_size):
    transform = T.Compose(
        [
            T.Lambda(torch.from_numpy),
            T.Lambda(minmax_scale),  # Scale to [0, 1]
            T.CenterCrop(image_size),
            T.Lambda(single_channel),  # Exactly one channel
        ]
    )
    return transform


def TrainTransformPrototypes(image_size):
    transform = T.Compose(
        [
            T.CenterCrop(image_size),
            T.Lambda(single_channel),  # Exactly one channel
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(180, interpolation=TF.InterpolationMode.BILINEAR),
            T.Lambda(train_scale),  # Scale to [-1, 1]
        ]
    )
    return transform


def EvalTransform(image_size):
    transform = T.Compose(
        [
            T.Lambda(single_channel),  # Only one channel
            T.Lambda(minmax_scale),  # Scale to [0, 1]
            T.CenterCrop(image_size),
        ]
    )
    return transform
