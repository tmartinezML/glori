from data.utils import load_lotss_catalog
import utils.paths as paths

import json
import traceback
from tqdm import tqdm
import pandas as pd
import pyarrow as pa, pyarrow.dataset as ds
from astropy.wcs import WCS
from data.obs.micromaps.utils import filter_catalog_by_wcs
from concurrent.futures import ThreadPoolExecutor, as_completed


fpath = paths.MICROMAP_SUBSETS["micromaps-DR3-512"]
split = "val"

cat = load_lotss_catalog()

json_files = sorted(list(fpath.glob(f"metadata/{split}/*.json")))

(fpath / f"catalogs/{split}").mkdir(exist_ok=True, parents=True)


def process_file(f):
    with open(f, "r") as file:
        metadata = json.load(file)
        headers = metadata["wcss"]
        names = metadata["names"]
        pointing = metadata["pointing"]

    cats = []
    for name, header in zip(names, headers):
        wcs = WCS(header=header)
        # print(wcs.pixel_shape)

        sub_cat = filter_catalog_by_wcs(cat, wcs, axes=(512,) * 2).copy()
        sub_cat["cutout_key"] = [
            name,
        ] * len(sub_cat)
        cats.append((name, sub_cat))

    if len(cats) > 0:
        sub_cat = pd.concat([c[1] for c in cats], ignore_index=True)
    else:
        # Make empty df with same colums as cat
        sub_cat = pd.DataFrame(
            columns=cat.columns.tolist()
            + [
                "cutout_key",
            ]
        )

    ds.write_dataset(
        pa.Table.from_pandas(sub_cat),
        base_dir=fpath / f"catalogs/{split}",
        basename_template=pointing + ".{i}.parquet",
        format="parquet",
        existing_data_behavior="overwrite_or_ignore",
    )


with ThreadPoolExecutor(max_workers=32) as executor:
    futures = [executor.submit(process_file, f) for f in json_files]
    for future in tqdm(as_completed(futures), total=len(json_files)):
        try:
            future.result()
        except Exception as e:
            traceback.print_exc()

print("Combining catalogs into one...")
import pandas as pd

df = pd.read_parquet(fpath / f"catalogs/{split}")
df.to_parquet(fpath / f"catalogs/{split}.parquet", index=False)