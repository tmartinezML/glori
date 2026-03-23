import os
import shutil
import statistics

import torch.distributed
import wandb
import torch
import lightning as L
import lightning.pytorch.callbacks as LCallbacks
from torch.utils.data import DataLoader
from webdataset import WebLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler

import utils.paths as paths
import data.sets.datasets as datasets
from models.vae.vae import VAE
from models.config import modelConfig
from utils.devices import visible_gpus_by_space
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class Benchmark(L.Callback):
    """A callback that measures the median execution time between the start and end of a batch."""

    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.times = []

    def median_time(self):
        return statistics.median(self.times)

    def on_train_batch_start(self, trainer, *args, **kwargs):
        self.start.record()

    def on_train_batch_end(self, trainer, *args, **kwargs):
        # Exclude the first iteration to let the model warm up
        if trainer.global_step > 1:
            self.end.record()
            torch.cuda.synchronize()
            self.times.append(self.start.elapsed_time(self.end) / 1000)


# Optimize for tensor cores
torch.set_float32_matmul_precision("high")

# When using DDP, some things should just run in the main process
# i.e. the first time the script is run
is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0

# Hyperparameters
conf = modelConfig.from_preset("VAE")
# conf.model_name = f"VAE"
if is_main_process:
    conf.pretty_print()

# Set callback
benchmark = Benchmark()

# Load Datasets
n_devices = len(conf.devices)
train_set, valid_set = [
    datasets.MicromapsDatasetVAE(
        dset="micromaps-256", split=split, n_gpu=n_devices
    ).batched(conf.batch_size)
    for split in ["train", "val"]
]
dl_kw = dict(  # Dataloader kwargs
    batch_size=None,
    num_workers=32,
    pin_memory=True,
)
# This entire party is necessary for DDP
train_dataloader, valid_dataloader = [
    WebLoader(dset, **dl_kw)
    .unbatched()
    .shuffle(1000)
    .with_epoch(nsamples=dset._len // n_devices)
    .batched(conf.batch_size)
    for dset in [train_set, valid_set]
]

# Create model
model = VAE.from_config(conf)
comp_model = torch.compile(model)

# Initialize trainer
devices = conf.devices
print(f"Using devices: {devices}")

# Measure the median iteration time with uncompiled model
trainer = L.Trainer(
    max_steps=10,
    devices=devices,
    strategy="ddp" if len(devices) > 1 else "auto",
    precision="bf16-mixed",
    callbacks=[
        benchmark,
    ],
    accumulate_grad_batches=conf.accumulate_grad_batches,
)
trainer.fit(model, train_dataloaders=train_dataloader)
eager_time = benchmark.median_time()

# Measure the median iteration time with compiled model
benchmark = Benchmark()
trainer = L.Trainer(
    max_steps=10,
    devices=devices,
    strategy="ddp" if len(devices) > 1 else "auto",
    precision="bf16-mixed",
    callbacks=[
        benchmark,
    ],
    accumulate_grad_batches=conf.accumulate_grad_batches,
)
trainer.fit(comp_model, train_dataloaders=train_dataloader)
compile_time = benchmark.median_time()

# Compare the speedup for the compiled execution
speedup = eager_time / compile_time
print(f"Eager median time: {eager_time:.4f} seconds")
print(f"Compile median time: {compile_time:.4f} seconds")
print(f"Speedup: {speedup:.1f}x")
