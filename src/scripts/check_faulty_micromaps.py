import numpy as np
from tqdm import tqdm

from data.sets.micromaps import MicromapDatasetHF
import data.trf.post as post
import data.trf.transforms as T


output_tuple = (
    "__key__",
    "npy",
    "context_downscaled.npy",
    "context_downscaled_d=2_f=2.npy",
)

ctxt_transform = (
    {
        "context_downscaled.npy": T.CatalogContextTransform(),
        "context_downscaled_d=2_f=2.npy": T.CatalogContextTransform(),
    },
)

split = "train"

dset = MicromapDatasetHF(
    "micromap-encodings-DR3-512",
    split="train",
    # post_transform=post.make_ctxt_to_topk(key="context_downscaled_d=2_f=2.npy", k=5),
    output_tuple=output_tuple,
    # ctxt_transform=ctxt_transform,
    mode="raw",
)

loader = dset.get_dataloader(batch_size=16, num_workers=32)

outfile = dset.arrow_path / f"metadata/nan_check_{split}.txt"

for batch in tqdm(loader):
    # Skip key for nan check
    for k, v in zip(output_tuple[1:], batch[1:]):
        is_nan = np.isnan(v)
        if is_nan.any():
            print(f"NaN values found in {batch[0]}: {k}!")
            for i in np.argwhere(is_nan.any(axis=tuple(range(1, is_nan.ndim)))):
                with open(outfile, "a") as f:
                    f.write(f"{batch[0][i]};{k}\n")
