import os
import shutil
import warnings
import inspect
from functools import partial
import argparse

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

from models.utils import parse_lightning_ckpt
import utils.paths as paths
import data.sets.datasets as datasets
import data.trf.transforms as T
from models.vae.vae import VAE
from models.vae.vqvae import VQVAE
from models.config import modelConfig
from utils.devices import visible_gpus_by_space
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# Optimize for tensor cores
torch.set_float32_matmul_precision("high")


class ResetLRSchedulerCallback(LCallbacks.Callback):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    def on_train_start(self, trainer, pl_module):
        """
        This method is called when training starts.
        It removes the lr scheduler and resets the learning rate while keeping
        the optimizer state.
        """
        trainer.schedulers = []
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.lr


# Add argument for limiting cpus
parser = argparse.ArgumentParser(description="Limit CPU usage")
parser.add_argument(
    "config", type=str, help="Preset config name for the model to train"
)
parser.add_argument("--num_cpus", type=int, default=16, help="Number of CPUs to use")
args = parser.parse_args()

# Limit Python to use only tNcpu = args.num_cpusN CPUs
Ncpu = args.num_cpus
num_workers = 16

if Ncpu != -1:
    os.sched_setaffinity(0, set(range(Ncpu, 2 * Ncpu)))


if __name__ == "__main__":

    # When using DDP, some things should just run in the main process
    # i.e. the first time the script is run
    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0

    # Hyperparameters
    # Get preset name from user input. Use default if no input is passed.
    preset_name = args.config
    conf = modelConfig.from_preset(preset_name)
    if is_main_process:
        if Ncpu != -1:
            print(f"Limiting CPU usage to {Ncpu} CPUs.")
            torch.set_num_threads(Ncpu)
        conf.pretty_print()

    # Set visible devices
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in conf.devices])

    # Set output directory
    output_dir = paths.MODEL_PARENT / conf.model_name

    # Sometimes interrupting leaves a broken symlink
    ckpt_path = output_dir / "lightning" / "last.ckpt"
    if (not os.path.exists(ckpt_path)) and os.path.islink(ckpt_path):
        print(f"Removing broken symlink {ckpt_path}")
        os.unlink(str(ckpt_path))
        assert not os.path.islink(
            ckpt_path
        ), f"Failed to remove broken symlink {ckpt_path}"

    # Initial setup only in main process
    if is_main_process:
        if output_dir.exists() and not conf.pickup:
            if conf.override_files:
                print(f"Overriding files in {output_dir}")
                shutil.rmtree(output_dir)
            else:
                # Rename by appending a number, increase number until
                # a non-existing name is found
                i = 1
                while output_dir.exists():
                    model_name = f"{model_name if i==1 else model_name[:-2]}_{i}"
                    output_dir = output_dir.parent / model_name
                    i += 1
                print(
                    f"Model name {model_name} already exists."
                    f" Renaming to {model_name}."
                )
                conf.model_name = model_name
        # Create output directory
        output_dir.mkdir(exist_ok=conf.pickup)

        # Save config as json file in output directory
        conf.save_to_json(output_dir / f"config.json")

        # Initialize Wandb logger
        # (only in main process, otherwise we get weird logging behavior)
        wandb_logger = WandbLogger(
            name=conf.model_name,
            save_dir=str(output_dir),
            project="LDM-VAE",
            log_model=False,
        )

    # Set callbacks and trainer args:
    # - Learning rate monitor
    lr_monitor = LCallbacks.LearningRateMonitor(logging_interval="step")
    # - Model checkpoint
    checkpoint_callback = LCallbacks.ModelCheckpoint(
        every_n_train_steps=conf.snapshot_interval
        * 2,  # factor 2 bc. of two optimizers
        dirpath=str(output_dir / "lightning"),
        filename="{train_step:08.0f}-{val_loss:.5e}",
        save_top_k=10,
        monitor="train_step",
        mode="max",
        save_last="link",
        enable_version_counter=False,
    )
    # Best checkpoint based on a specific metric (e.g., validation loss)
    best_checkpoint_callback = LCallbacks.ModelCheckpoint(
        monitor="val_loss",  # Metric to monitor
        mode="min",  # Save the checkpoint with the minimum `val_loss`
        dirpath=str(output_dir / "lightning"),
        filename="best-{train_step:08.0f}-{val_loss:.5e}",  # Filename format for the best checkpoint
        save_top_k=5,  # Save only the best checkpoint
    )
    # - Profiler (prints performance stats at the end of training)
    profiler = SimpleProfiler(
        dirpath=str(output_dir / "lightning"),
        filename="performance_logs",
    )
    callbacks = [
        checkpoint_callback,
        best_checkpoint_callback,
        lr_monitor,
    ]
    if conf.pickup:
        pass
        callbacks.append(ResetLRSchedulerCallback(conf.lr))

    # Load datasets
    dset_kwargs = dict(
        dset=conf.dataset,
        output_tuple=("npy",),
        missing_is_error=conf.get("missing_is_error", True),
        max_beam_arcsec=conf.get("max_beam_arcsec", 6),
        dset_lookup=getattr(
            paths, conf.get("dataset_lookup", "MICROMAP_SUBSETS_ARROW_HOPPER")
        ),
        scaler=conf.get("scaler"),
    )
    crop_trf = partial(
        T.WebdatasetTransformCrop,
        crop_size=conf.get("image_size", 256),
    )
    num_workers = conf.get("num_workers", num_workers)
    dl_kwargs = dict(
        batch_size=conf.batch_size,
        num_workers=conf.get("num_workers", num_workers),
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True if num_workers > 0 else False,
    )
    # Train set
    train_set = datasets.MicromapDatasetHF(
        **dset_kwargs,
        split="train",
        weights_file=conf.get("weights_file"),
        custom_transform=partial(
            crop_trf,
            random_crop=True,
        ),
    )
    train_dataloader = train_set.get_dataloader(shuffle=True, **dl_kwargs)
    # Val set
    valid_set = datasets.MicromapDatasetHF(
        **dset_kwargs,
        split="val",
        custom_transform=crop_trf,
    )
    ii = torch.randint(0, len(valid_set), (conf.batch_size * conf.val_batches,))
    valid_set.dataset.select(ii)
    valid_dataloader = valid_set.get_dataloader(shuffle=False, **dl_kwargs)

    # Parse checkpoint path
    ckpt_path = (
        parse_lightning_ckpt(conf.checkpoint, model_name=conf.model_name)
        if (
            conf.get("pickup", False)
            or conf.get("weights_only", False)
            or conf.get("tune_lr", False)
        )
        else None
    )

    # Create model
    match conf.model_class:
        case "VAE":
            model_class = VAE
        case "VQVAE":
            model_class = VQVAE
        case _:
            raise ValueError(f"Unknown model class: {conf.model_class}")

    if conf.get("tune_lr", False):
        model = model_class.load_from_checkpoint(ckpt_path)

    else:
        model = model_class.from_config(conf)

    if conf.get("weights_only", False):
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    if is_main_process:
        wandb_logger.log_hyperparams(conf.param_dict)

    # Initialize trainer
    # devices = visible_gpus_by_space()[: conf.n_devices]
    # devices = [3, 1]
    devices = conf.devices
    print(f"Using devices: {devices}")
    trainer = L.Trainer(
        accumulate_grad_batches=conf.get("accumulate_grad_batches", 1),
        detect_anomaly=False,
        overfit_batches=1 if conf.overfit_batch else 0.0,
        check_val_every_n_epoch=None,
        # gradient_clip_val=1.0,
        max_steps=conf.iterations * 2,  # factor 2 bc. of two optimizers
        devices=devices,
        strategy="ddp_find_unused_parameters_true" if len(devices) > 1 else "auto",
        precision="32",
        logger=wandb_logger if is_main_process else None,
        default_root_dir=str(output_dir / "lightning"),
        log_every_n_steps=conf.log_interval,
        callbacks=callbacks,
        val_check_interval=conf.val_every,
        limit_val_batches=conf.val_batches,
        profiler=profiler,
        num_sanity_val_steps=2,
    )

    if conf.get("tune_lr", False):
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
            min_lr=1e-6,
            max_lr=1e-4,
            num_training=100,
            early_stop_threshold=None,
        )
        # Log and print the suggested learning rate
        print(f"Suggested learning rate: {lr_finder.suggestion()}")
        if is_main_process:
            # Log the suggested learning rate as a scalar
            wandb_logger.experiment.log(
                {"lr_finder/suggested_lr": lr_finder.suggestion()}
            )

            # Log the learning rate vs. loss as a table
            lr_results = lr_finder.results
            lr_table = wandb.Table(columns=["lr", "loss"])
            for lr, loss in zip(lr_results["lr"], lr_results["loss"]):
                lr_table.add_data(lr, loss)
            wandb_logger.experiment.log({"lr_finder/lr_vs_loss": lr_table})

        # Exit after tuning
        exit(0)

    # Run training
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=(
            ckpt_path
            if (conf.get("pickup", False) and not conf.get("weights_only", False))
            else None
        ),
    )
