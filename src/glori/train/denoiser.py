import argparse
import os
import shutil
import warnings
import inspect
from types import SimpleNamespace
from collections import namedtuple
from pathlib import Path
from functools import partial

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

import glori.settings.paths as paths
import glori.data.sets.micromaps as micromaps
import glori.data.trf.transforms as transforms
from glori.data.trf.functional import zero_center
from glori.models.vae.vae import VAE
from glori.models.vae.vqvae import VQVAE
from glori.models.diffusion.denoiser import Denoiser
from glori.config.model_config import modelConfig
from glori.models.load import parse_lightning_ckpt
from glori.infra.devices import visible_gpus_by_space
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import glori.data.trf.post as post
from glori.data.trf.post import make_dependent_random_crop

# Optimize for tensor cores
torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    # Add argument for limiting cpus
    parser = argparse.ArgumentParser(description="Limit CPU usage")
    parser.add_argument(
        "config", type=str, help="Preset config name for the model to train"
    )
    parser.add_argument(
        "--num_cpus", type=int, default=16, help="Number of CPUs to use"
    )
    args = parser.parse_args()

    # Limit Python to use only tNcpu = args.num_cpusN CPUs
    Ncpu = args.num_cpus
    num_workers = 16

    if Ncpu != -1:
        os.sched_setaffinity(0, set(range(Ncpu, 2 * Ncpu)))
    # Check the CPUs available to the process
    # print("CPUs available:", os.sched_getaffinity(0))
    # When using DDP, some things should just run in the main process
    # i.e. the first time the script is run
    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0

    # Hyperparameters
    # Get preset name from user input. Use default if no input is passed.
    preset_name = args.config  # if len(os.sys.argv) > 1 else "VAE"
    conf = modelConfig.from_preset(preset_name)

    # Some global config settings should propagate into the training and
    # model configs for convenience
    conf.train_config["context"] = conf.train_config.get("context", [])
    if conf.get("img_context") is not None:
        conf.train_config["context"].append("img_context")
    if conf.get("catalog_context") is not None:
        conf.train_config["context"].append("catalog_context")
    if (topk := conf.get("catalog_topk")) is not None and conf.get(
        "catalog_context"
    ) is not None:
        conf.model_config["catalog_context_topk"] = topk
        conf.train_config["catalog_context_topk"] = topk

    train_conf = modelConfig(**conf.train_config)
    if is_main_process:
        if Ncpu != -1:
            print(f"Limiting CPU usage to {Ncpu} CPUs.")
            torch.set_num_threads(Ncpu)
        conf.pretty_print()

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
        if output_dir.exists():
            if conf.override_files:
                if conf.pickup:
                    assert "/" in conf.checkpoint and not Path(
                        conf.checkpoint
                    ).is_relative_to(
                        output_dir
                    ), "Cannot override files when picking up from existing output directory."
                print(f"Overriding files in {output_dir}")
                shutil.rmtree(output_dir)
            elif not conf.pickup:
                # Rename by appending a number, increase number until
                # a non-existing name is found
                i = 1
                model_name = conf.model_name
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
            project="LDM-Denoiser",
            log_model=False,
        )

    # Set callbacks and trainer args:
    # - Learning rate monitor
    lr_monitor = LCallbacks.LearningRateMonitor(logging_interval="step")
    # - Model checkpoint
    checkpoint_callback = LCallbacks.ModelCheckpoint(
        every_n_train_steps=train_conf.snapshot_interval,
        dirpath=str(output_dir / "lightning"),
        filename="{train_step:08.0f}-{val_loss:.5e}",
        save_top_k=10,
        monitor="train_step",
        mode="max",
        save_last="link",
        enable_version_counter=False,
    )
    # Best checkpoint based on validation loss
    best_checkpoint_callback = LCallbacks.ModelCheckpoint(
        monitor="val_loss",  # Metric to monitor
        mode="min",  # Save the checkpoint with the minimum `val_loss`
        dirpath=str(output_dir / "lightning"),
        filename="best-{train_step:08.0f}-{val_loss:.5e}",  # Filename format for the best checkpoint
        save_top_k=5,  # Save the best k checkpoints
    )
    # - Profiler (prints performance stats at the end of training)
    profiler = SimpleProfiler(
        dirpath=str(output_dir / "lightning"),
        filename="performance_logs",
    )
    callbacks = [
        lr_monitor,
        checkpoint_callback,
        best_checkpoint_callback,
    ]
    if train_conf.get("ema_rate", None) is not None:
        # Best checkpoint based on EMA validation loss
        best_ema_checkpoint_callback = LCallbacks.ModelCheckpoint(
            monitor="val_ema_loss",  # Metric to monitor
            mode="min",  # Save the checkpoint with the minimum `val_ema_loss`
            dirpath=str(output_dir / "lightning"),
            filename="best_ema-{train_step:08.0f}-{val_ema_loss:.5e}",  # Filename format for the best EMA checkpoint
            save_top_k=1,  # Save only the best EMA checkpoint
        )
        callbacks.append(best_ema_checkpoint_callback)

    # Prepare configuration for Datasets
    output_tuple = ("npy",)
    if conf.get("img_context") is not None:
        # If the model uses image context, add it to the output tuple
        output_tuple += (conf.img_context,)
    if conf.get("catalog_context") is not None:
        # If the model uses catalog context, add it to the output tuple
        output_tuple += (conf.catalog_context,)

    ctxt_transform = {}
    if hasattr(conf, "ctxt_transform"):
        if hasattr(conf, "ctxt_scalers"):
            scale_fns = {
                k: transforms.make_catalog_context_value_scale(v)
                for k, v in conf.ctxt_scalers.items()
            }
        ctxt_transform = {
            k: getattr(transforms, v)(
                **(dict(scale_fn=scale_fns[k])) if k in scale_fns else {}
            )
            for k, v in conf.ctxt_transform.items()
        }
    post_transforms = []
    if train_conf.get("random_crop") is not None:
        post_transforms.append(
            post.make_dependent_random_crop(
                crop_size=train_conf.random_crop,
                f=train_conf.get("crop_fctxt", 1),
                s=train_conf.get("crop_sctxt", 1),
                keys=("npy", *list(ctxt_transform.keys())),
            )
        )

    if train_conf.get("blank_center") is not None:
        post_transforms.append(
            post.make_post(
                {
                    k: partial(zero_center, f_center=v)
                    for k, v in train_conf.blank_center.items()
                },
            )
        )

    if train_conf.get("artifact_maker") is not None:
        post_transforms.append(
            post.make_post(
                {k: post.ctxt_to_artifact for k in train_conf.artifact_maker},
            )
        )

    if (topk := train_conf.get("catalog_context_topk")) is not None:
        key = conf.get("catalog_context", None)
        assert (
            key is not None
        ), "catalog_context_topk is set but catalog_context is not defined in the config."
        if key in scale_fns:
            print(
                f"WARNING: Using scale fns for topk selection on key {key}."
                f" Make sure topk_min_val (set to {conf.get('topk_min_val', 0)}) is in the right scale."
            )
        post_transforms.append(
            post.make_ctxt_to_topk(
                key=key,
                k=topk,
                min_val=conf.get("topk_min_val", 0),
                copy=False,
            )
        )

    # Load datasets
    dset_kwargs = dict(
        output_tuple=output_tuple,
        ctxt_transform=ctxt_transform,
        post_transform=post.compose(*post_transforms),
        missing_is_error=train_conf.get("missing_is_error", True),
        max_beam_arcsec=conf.get("max_beam_arcsec", None),
    )
    if conf.get("weights_fn") is not None:
        dset_kwargs["weights_fn"] = eval(conf.weights_fn)
    num_workers = train_conf.get("num_workers", num_workers)
    dl_kwargs = dict(
        batch_size=train_conf.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True if num_workers > 0 else False,
    )
    # Train set
    dset_lookup = paths.MICROMAP_SUBSETS_ARROW
    if conf.get("dataset_lookup") is not None:
        dset_lookup = getattr(paths, conf.dataset_lookup)
    train_set = micromaps.MicromapDatasetHF(
        dset=conf.dataset,
        dset_lookup=dset_lookup,
        split="train",
        weights_file=conf.get("weights_file"),
        **dset_kwargs,
    )
    if conf.get("selection_file") is not None:
        # Expected to be a pytorch file with the indices to select
        selection_indices = torch.load(
            train_set.path / "metadata" / conf.selection_file
        )
        assert (
            selection_indices.ndim == 1
        ), f"Selection indices should be a 1D array of indices, got {selection_indices.ndim}."
        train_set.select(selection_indices)
    train_dataloader = train_set.get_dataloader(shuffle=True, **dl_kwargs)
    # Val set
    valid_set = micromaps.MicromapDatasetHF(
        dset=conf.dataset, split="val", dset_lookup=dset_lookup, **dset_kwargs
    )
    ii = torch.randint(
        0, len(valid_set), (train_conf.batch_size * train_conf.val_batches,)
    )
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
        case "Denoiser":
            model_class = Denoiser
        case _:
            raise ValueError(f"Unknown model class: {conf.model_class}")
    model = model_class.from_config(conf)
    if conf.get("weights_only", False):
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    elif conf.get("tune_lr", False):
        model = model_class.load_from_checkpoint(ckpt_path)
    if is_main_process:
        wandb_logger.log_hyperparams(conf.param_dict)

    # Initialize trainer
    devices = train_conf.devices
    print(f"Using devices: {devices}")
    trainer = L.Trainer(
        accumulate_grad_batches=train_conf.get("accumulate_grad_batches", 1),
        detect_anomaly=False,
        overfit_batches=1 if conf.get("overfit_batch", False) else 0.0,
        check_val_every_n_epoch=None,
        # gradient_clip_val=1.0,
        max_steps=train_conf.iterations,
        devices=devices,
        # strategy="ddp_find_unused_parameters_true" if len(devices) > 1 else "auto",
        strategy="auto",
        precision="32",
        logger=wandb_logger if is_main_process else None,
        default_root_dir=str(output_dir / "lightning"),
        log_every_n_steps=train_conf.log_interval,
        callbacks=callbacks,
        val_check_interval=train_conf.val_every,
        limit_val_batches=train_conf.val_batches,
        profiler=profiler,
        num_sanity_val_steps=0,  # 2,
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
