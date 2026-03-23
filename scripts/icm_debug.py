from deprecated.datasets import MicromapsDatasetLDM
from glori.inference.swiit_sampler import SWIITSampler


from data.sets.datasets import MicromapsDatasetVAE

import data.trf.transforms as T

from data.trf.post import make_dependent_center_crop


out = ("npy", "context_downscaled.npy", "__url__", "__key__")
kw = dict(
    shuffle=False,
    shardshuffle=False,
    output_tuple=out,
    split="test",
    resampled=False,
    ctxt_transform={
        "context_downscaled.npy": T.CatalogContextTransform(),
    },
    post_transform=make_dependent_center_crop(128, f=1),
)
ldm_dset = MicromapsDatasetLDM("micromap-encodings-DR3-1024", **kw)


from webdataset import WebLoader

loader = WebLoader(ldm_dset, batch_size=16, num_workers=16)  # .shuffle(1000)
loader_it = iter(loader)

import tarfile
import io
import numpy as np


def get_micromap(urls, keys):
    data = []
    for url, key in zip(urls, keys):
        url_map = url.replace("micromap_encodings", "micromaps")
        with tarfile.open(url_map, "r") as tar:
            member = tar.getmember(f"{key}.npy")
            f = tar.extractfile(member)
            data.append(np.load(io.BytesIO(f.read())))
            f.close()
    return np.array(data)


icm_sampler = SWIITSampler(
    denoiser="LDM-Denoiser-CC-128",
    denoiser_ckpt="best-last-10%",
    uncond_denoiser="LDM-Denoiser-Uncond-128",
    uncond_denoiser_ckpt="best-last-10%",
    vae="VQ-VAE-256",
    device="cuda:1",
    latent_size=128,
    image_size=512,
    div_stride=2,
)

loader_it = iter(loader)
i = 5
for _ in range(i):
    batch = next(loader_it)
encodings, enc_ctxt, urls, keys = batch
batch_maps = get_micromap(urls, keys)

map_img, latent_map = icm_sampler.sample(
    ctxt_map=enc_ctxt,
    guidance_strength=0.0,
)
