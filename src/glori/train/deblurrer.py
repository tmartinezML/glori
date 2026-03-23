import os
import shutil
import warnings

import wandb
import torch
import lightning as L
import lightning.pytorch.callbacks as LCallbacks
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler

import glori.settings.paths as paths
import glori.data.sets.micromaps as micromaps
from glori.models.deblur.deblurrer import Deblurrer, ResidualDeblurrer
from glori.config.model_config import modelConfig

# Optimize for tensor cores
torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    # When using DDP, some things should just run in the main process
    # i.e. the first time the script is run
    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0

    # Hyperparameters
    conf = modelConfig.from_preset("Deblurrer")
    if is_main_process:
        conf.pretty_print()

    # Set output directory
    output_dir = paths.MODEL_PARENT / conf.model_name
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

        # Initialize Wandb logger
        # (only in main process, otherwise we get weird logging behavior)
        wandb_logger = WandbLogger(
            name=conf.model_name,
            save_dir=str(output_dir),
            project="Deblurrer",
            log_model=False,
        )

    # Set callbacks and trainer args:
    # - Learning rate monitor
    lr_monitor = LCallbacks.LearningRateMonitor(logging_interval="step")
    # - Model checkpoint
    checkpoint_callback = LCallbacks.ModelCheckpoint(
        every_n_epochs=conf.snapshot_interval,
        dirpath=str(output_dir / "lightning"),
        filename="{epoch:02d}-{val_loss:.2e}",
        save_top_k=-1,
        save_last="link",
        enable_version_counter=False,
        save_on_train_epoch_end=True,
    )
    # Best checkpoint based on a specific metric (e.g., validation loss)
    best_checkpoint_callback = LCallbacks.ModelCheckpoint(
        monitor="val_loss",  # Metric to monitor
        mode="min",  # Save the checkpoint with the minimum `val_loss`
        dirpath=str(output_dir / "lightning"),
        filename="best-{step:08.0f}-{val_loss:.2e}",  # Filename format for the best checkpoint
        save_top_k=1,  # Save only the best checkpoint
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

    # Load Datasets
    n_devices = len(conf.devices)
    # train_set = micromaps.LOFARPrototypesDataset("prototypes", img_size=conf.image_size)
    # TODO: Replace with new dataset implementation
    train_set = None
    train_set, valid_set = random_split(
        train_set,
        [0.9, 0.1],
        generator=torch.Generator().manual_seed(42),
    )
    dl_kw = dict(  # Dataloader kwargs
        batch_size=conf.batch_size,
        num_workers=32,
        pin_memory=True,
        shuffle=True,
    )
    train_dataloader, valid_dataloader = [
        DataLoader(dset, **dl_kw) for dset in [train_set, valid_set]
    ]

    # Set checkpoint path
    if conf.checkpoint == "best":
        # Look for best checkpoint
        try:
            ckpt_path = sorted(
                (output_dir / "lightning").glob("best-*.ckpt"),
                key=os.path.getmtime,
            )[-1]
        except IndexError as e:
            if conf.pickup:
                raise RuntimeError(
                    "No checkpoint found. "
                    "Please set `--checkpoint` to a valid checkpoint path."
                ) from e
            else:
                warnings.warn(
                    "No checkpoint found. "
                    "Training from scratch. "
                    "Please set `checkpoint` to a valid checkpoint path."
                )

    else:
        ckpt_path = (
            conf.checkpoint
            if "/" in conf.checkpoint
            else output_dir / "lightning" / conf.checkpoint
        )

    # Create model
    match conf.model_type:
        case "Deblurrer":
            model = Deblurrer.from_config(conf)
        case "ResidualDeblurrer":
            model = ResidualDeblurrer.from_config(conf)
        case _:
            raise ValueError(
                f"Unknown model type {conf.model_type}. "
                "Please set `model_type` to either `Deblurrer` or `ResidualDeblurrer`."
            )

    if conf.tune_lr:
        model = Deblurrer.load_from_checkpoint(ckpt_path)

    if is_main_process:
        wandb_logger.log_hyperparams(conf.param_dict)

    # Initialize trainer
    # devices = visible_gpus_by_space()[: conf.n_devices]
    # devices = [3, 1]
    devices = conf.devices
    print(f"Using devices: {devices}")
    trainer = L.Trainer(
        # detect_anomaly=True,
        overfit_batches=1 if conf.overfit_batch else 0.0,
        # gradient_clip_val=1.0,
        max_epochs=conf.epochs,
        devices=devices,
        strategy="auto",
        precision="32",
        logger=wandb_logger if is_main_process else None,
        default_root_dir=str(output_dir / "lightning"),
        log_every_n_steps=conf.log_interval,
        callbacks=callbacks,
        val_check_interval=conf.val_every,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=conf.accumulate_grad_batches,
        limit_val_batches=conf.val_batches,
        profiler=profiler,
    )

    if conf.tune_lr:
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
        ckpt_path=ckpt_path if conf.pickup else None,
    )
