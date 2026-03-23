from pathlib import Path
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm

train_dir = Path(
    "/hs/fs08/data/group-brueggen/tmartinez/diffusion/image_data/LOFAR/micromaps-arrow/micromap_encodings-DR3-opt-1024px-spacing=1/train.arrow"
)

# 1) Collect cache shards
cache_shards = sorted(train_dir.glob("cache-*.arrow"))
print("cache shards:", len(cache_shards))

# 2) Load each cache shard
cache_dsets = []
for shard in tqdm(cache_shards, desc="Loading cache shards"):
    try:
        cache_dsets.append(Dataset.from_file(str(shard)))
    except Exception as e:
        print(f"Failed to load {shard}: {e}")

print("loaded shards:", len(cache_dsets))

# 3) Concatenate and save recovered dataset
if len(cache_dsets) > 0:
    print("Concatenating datasets...")
    recovered = concatenate_datasets(cache_dsets)
    print("Filtering out None samples...")
    print("total recovered samples:", len(recovered))
    recovered = recovered.filter(lambda x: x is not None, input_columns=["__key__"])
    print("total valid samples:", len(recovered))
    print("Saving recovered dataset...")
    out_dir = train_dir.with_name("train_recovered.arrow")
    recovered.save_to_disk(str(out_dir), num_shards=256, num_proc=8)

    print("saved:", out_dir)
