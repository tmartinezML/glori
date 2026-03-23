import wandb

import utils.paths as paths
import data.sets.datasets as datasets
from deprecated.micromap_webdataset import MicromapDataset
from models.config import modelConfig
from deprecated.trainer import DiffusionTrainer
from utils.devices import set_visible_devices


if __name__ == "__main__":
    # Limit visible GPUs
    set_visible_devices(2)

    # Hyperparameters
    conf = modelConfig.from_preset("WebDset_Test")
    # conf.model_name = f"Prototypes_Model_SizeCond"

    dataset = datasets.MicromapsDatasetTest(dset="micromaps-256")

    trainer = DiffusionTrainer(
        config=conf,
        dataset=dataset,
        # pickup=True,
    )

    wandb.init(
        project="Diffusion",
        config=conf.param_dict,
        dir=paths.ANALYSIS_PARENT / "wandb",
        # Use this for pickup:
        # id="mm5hsmh2",
        # resume="must",
    )

    trainer.training_loop()
