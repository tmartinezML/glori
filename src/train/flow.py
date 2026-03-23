import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import wandb

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import CompositeTransform
import nflows.transforms.coupling as nfcoupling
from nflows.transforms.permutations import RandomPermutation
from nflows.nn.nets import ResidualNet, MLP
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
import itertools

import utils.paths as paths
import utils.my_logging as my_logging
from models.config import modelConfig
from data.utils import load_fits_catalog
from data.trf.scalers import ContextScaler

logger = my_logging.get_logger(__name__)


# -----------------------------
# Create RealNVP flow
# -----------------------------
def create_realnvp_flow(
    input_dim=3,
    context_dim=6,
    hidden_features=64,
    num_layers=6,
    num_blocks=2,
    num_splits=2,
    coupling_transform="PiecewiseRationalQuadraticCouplingTransform",
    coupling_transform_kwargs={},
):

    def create_net(in_features, out_features):
        return ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            context_features=context_dim,
            num_blocks=num_blocks,
            activation=nn.ReLU(),
            dropout_probability=0.1,
        )

    transforms = []
    for i in range(num_layers):
        mask = torch.tensor(
            [(i + j) % num_splits for j in range(input_dim)], dtype=torch.bool
        )
        transform = getattr(nfcoupling, coupling_transform)(
            mask=mask, transform_net_create_fn=create_net, **coupling_transform_kwargs
        )
        transforms.append(transform)
        transforms.append(RandomPermutation(features=input_dim))

    flow = Flow(
        transform=CompositeTransform(transforms),
        distribution=StandardNormal(shape=[input_dim]),
    )
    return flow


# -----------------------------
# Loss function
# -----------------------------
def compute_loss(flow, x_batch, mask_batch):
    context = torch.cat([x_batch * mask_batch, mask_batch], dim=-1)
    log_prob = flow.log_prob(x_batch, context=context)
    return -log_prob.mean()


# -----------------------------
# Conditional sampling
# -----------------------------
def conditional_sample(flow, fixed_values, fixed_mask, num_samples):
    context = torch.cat([fixed_values * fixed_mask, fixed_mask], dim=-1)
    z = torch.randn(num_samples, 3).to(fixed_values.device)
    x_sampled = flow._transform.inverse(z, context=context)
    x_sampled = x_sampled * (1 - fixed_mask) + fixed_values * fixed_mask
    return x_sampled


# -----------------------------
# Training
# -----------------------------
def train(flow, train_loader, val_loader, network_config, **train_kwargs):
    # Optimizer setup
    optimizer = torch.optim.Adam(flow.parameters(), lr=train_kwargs["lr"])
    if use_scheduler := (
        ("scheduler" in train_kwargs) and (train_kwargs["scheduler"] is not None)
    ):
        scheduler_cls = getattr(torch.optim.lr_scheduler, train_kwargs["scheduler"])
        scheduler = scheduler_cls(optimizer, **train_kwargs["scheduler_kwargs"])

    # Training params setup
    model_name = train_kwargs["model_name"]
    model_path = paths.MODEL_PARENT / model_name
    resume = train_kwargs.get("pickup", False)
    if resume:
        assert (
            model_path.exists()
        ), f"Cannot resume - model path {model_path} does not exist."
        last_ckpt = sorted(
            model_path.glob("params_epoch=*.pt"),
            key=lambda x: x.name.split("epoch=")[1],
        )[0]
        logger.info(f"Resuming from {last_ckpt}")
        checkpoint = torch.load(last_ckpt, map_location="cpu")
        flow.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if use_scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
        last_epoch = checkpoint["epoch"]
        last_i_step = checkpoint["i_step"]

    elif model_path.exists():
        if not train_kwargs.get("override", False):
            raise FileExistsError(f"Model path {model_path} already exists.")
        logger.warning(f"Model path {model_path} already exists. Overriding.")
        shutil.rmtree(model_path)
    model_path.mkdir(exist_ok=True)

    log_interval = train_kwargs.get("log_interval", 100)
    epochs = train_kwargs.get("epochs", 10)
    device = train_kwargs.get("device", "cuda:0")
    flow = flow.to(device)

    # This is required for validation plot
    all_xval = torch.cat([x[0] for x in val_loader], dim=0).to(device)

    # Training Loop
    flow.train()
    i_step = last_i_step if resume else 0
    for epoch in range(last_epoch if resume else 0, epochs):
        total_loss = 0
        for i, x_batch in enumerate(tqdm(train_loader)):

            # Training Step
            x_batch = x_batch[0].to(device)
            mask_batch = torch.randint(0, 2, x_batch.shape).float().to(device)
            loss = compute_loss(flow, x_batch, mask_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_scheduler:
                scheduler.step()
            total_loss += loss.item()

            # Logging
            if i % log_interval == 0:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_step": i_step + 1,
                        "loss": loss.item(),
                        "LR": (
                            scheduler.get_last_lr()[0]
                            if use_scheduler
                            else optimizer.param_groups[0]["lr"]
                        ),
                    }
                )
            i_step += 1

        # Validation
        total_val_loss = 0
        for x_val in val_loader:
            x_val = x_val[0].to(device)
            mask_val = torch.randint(0, 2, x_val.shape).float().to(device)
            val_loss = compute_loss(flow, x_val, mask_val)
            total_val_loss += val_loss.item()

        # Log metrics
        wandb.log(
            {
                "epoch": epoch + 1,
                "epoch_loss": total_loss / len(train_loader),
                "val_loss": total_val_loss / len(val_loader),
            }
        )
        logger.info(
            f"Epoch {epoch+1}, Average loss: {total_loss / len(train_loader):.4f}, Val loss: {total_val_loss / len(val_loader):.4f}"
        )

        # Make and log test plots
        fig, _ = flow_test_plot(flow, all_xval, device=device)
        wandb.log({"test_plot": wandb.Image(fig)})
        fig, _ = flow_latent_distribution_plot(flow, all_xval, device=device)
        wandb.log({"latent_distribution_plot": wandb.Image(fig)})

        # Save model
        torch.save(
            {
                "train_config": train_kwargs,
                "network_config": network_config.param_dict,
                "model": flow.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if use_scheduler else None,
                "epoch": epoch + 1,
                "i_step": i_step + 1,
            },
            model_path
            / f"params_epoch={epoch + 1}_val-loss={total_val_loss / len(val_loader):.4f}.pt",
        )


def flow_test_plot(flow, x_test, device="cuda:0"):
    npoints = 150
    x = torch.linspace(-5, 5, npoints)
    xgrid, ygrid, zgrid = torch.meshgrid(x, x, x, indexing="ij")
    xyz_input = torch.stack(
        [xgrid.flatten(), ygrid.flatten(), zgrid.flatten()], dim=-1
    ).to("cuda:0")
    context = torch.zeros_like(xyz_input).repeat(1, 2).to("cuda:0")

    flow = flow.to(device)
    with torch.no_grad():
        p_xyz = (
            flow.log_prob(xyz_input, context=context)
            .exp()
            .reshape(*(npoints,) * 3)
            .cpu()
            .numpy()
        )

    bin_width = x[1] - x[0]
    density = lambda x: x / x.sum() / bin_width

    px = density(p_xyz.sum(axis=(1, 2)))
    py = density(p_xyz.sum(axis=(0, 2)))
    pz = density(p_xyz.sum(axis=(0, 1)))
    marginals_1 = [px, py, pz]

    p_xy = density(p_xyz.sum(axis=2))
    p_xz = density(p_xyz.sum(axis=1))
    p_yz = density(p_xyz.sum(axis=0))
    marginals_2 = [p_xy, p_xz, p_yz]

    x_test_np = x_test.cpu().numpy()

    # Plot marginal distributions into 3x3 grid, with no titles
    flow_color = "#72109c"
    data_color = "#23840d"
    labels = ["Total Flux", "Peak Flux", "Major Axis"]
    # Example: Share x and y axis for custom selection of subplots
    # axs is a 3x3 array of axes

    # You can also use set_xlim/set_ylim to manually synchronize limits if needed
    fig, axs = plt.subplots(
        3,
        3,
        figsize=(14, 14),
        # tight_layout=True,
        gridspec_kw={"height_ratios": [1, 1, 1], "width_ratios": [1, 1, 1]},
        sharex="col",
    )
    hist_kw = dict(bins=x, density=True)
    contour_kw = dict(
        colors=data_color,
        levels=5,
        extent=(-5, 5, -5, 5),
        origin="lower",
    )
    imshow_kw = dict(
        extent=(-5, 5, -5, 5),
        origin="lower",
        aspect="auto",
    )

    for i, j in itertools.product(range(3), range(3)):
        ax = axs[i, j]
        if i == j:
            ax.plot(
                x,
                marginals_1[i],
                color=flow_color,
            )
            ax.hist(x_test_np[:, i], **hist_kw, color=data_color)

        elif i < j:
            # set invisible
            ax.axis("off")

        else:
            ax.imshow(
                # Transpose because we plot on bottom left half
                marginals_2[i + j - 1].T,
                **imshow_kw,
            )
            ax.contour(
                np.histogram2d(x_test_np[:, i], x_test_np[:, j], **hist_kw)[0],
                **contour_kw,
            )

        if j != 0 and j != i:
            ax.set_yticklabels([])

        elif i != j:
            ax.set_ylabel(labels[i])

        if i == j:
            ax.set_ylabel(f"{labels[i]} Marginal")
            # put everything on right hand side
            # ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

        # X tick labels are handled by sharex=col option
        if i == 2:
            ax.set_xlabel(labels[j])

    for ax in axs.flatten():
        ax.grid(alpha=0.3)

    # To share y axis for subplots in column 0 and 2:
    axs[2, 0].get_shared_y_axes().join(axs[2, 0], axs[2, 1])

    legend_elements = [
        Line2D([0], [0], color=flow_color, label="Flow"),
        Line2D([0], [0], color=data_color, label="Data"),
    ]
    fig.legend(
        handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2
    )
    return fig, axs


# -----------------------------
# Testing latent distribution
# -----------------------------
def flow_latent_distribution_plot(flow, x_test, device="cuda:0"):
    flow = flow.to(device)
    flow.eval()
    x_test = x_test.to(device)
    with torch.no_grad():
        B = x_test.shape[0]
        mask = torch.randint(0, 2, x_test.shape).float().to(device)
        context = torch.cat([x_test * mask, mask], dim=-1)
        z = flow.transform_to_noise(x_test, context=context)
        z = z.cpu().numpy()

        fig, axs = plt.subplots(3, 1, figsize=(10, 15), constrained_layout=True)
        for i in range(z.shape[1]):
            axs[i].hist(
                z[:, i],
                bins=100,
                density=True,
                alpha=0.6,
                log=False,
                label=f"Mean: {z[:, i].mean():.2f}, Std: {z[:, i].std():.2f}",
            )
            axs[i].set_title(f"Latent dim {i} distribution")
            # Plot gaussian for visual aid
            axs[i].plot(
                x := np.linspace(-5, 5, 100),
                np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi),
                color="red",
                label="Standard Normal",
                alpha=0.5,
                ls="--",
            )
            axs[i].legend()
            axs[i].grid(alpha=0.3)
    return fig, axs


def load_datasets(catalog=paths.LOTSS_DR3_CAT):
    logger.info("Reading catalog...")
    cat = load_fits_catalog(catalog)

    # Scale and bring into right shape
    logger.info("Loading scalers...")
    scalers = {
        "Total_flux": ContextScaler.load("ctxt_scaler_ftot"),
        "Peak_flux": ContextScaler.load("ctxt_scaler_fpeak"),
        "Maj": ContextScaler.load("ctxt_scaler_maj"),
    }
    logger.info("Shaping and scaling...")
    x_data = torch.stack(
        [torch.from_numpy(scaler.scale(cat[k].values)) for k, scaler in scalers.items()]
    ).T.to(torch.float32)

    # Split into 10% test set and 90% validation set
    logger.info("Splitting data...")
    x_train, x_test = train_test_split(x_data, test_size=0.1, random_state=42)
    x_test, x_valid = train_test_split(x_test, test_size=0.5, random_state=42)
    return x_train, x_valid, x_test


def load_model(model_name, ckpt="best"):

    model_path = paths.MODEL_PARENT / model_name

    # Check for issues with model path
    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    if not model_path.is_dir():
        raise NotADirectoryError(f"Model path {model_path} is not a directory.")
    if not any(model_path.glob("params_epoch=*.pt")):
        raise FileNotFoundError(
            f"No checkpoints found in {model_path}. Expected files matching 'params_epoch=*.pt'."
        )

    # Get best or last checkpoint file
    match ckpt:
        case "best":
            sort_key = lambda x: float(x.stem.split("val-loss=")[1])
            reverse = False
        case "last":
            sort_key = lambda x: int(x.stem.split("epoch=")[1].split("_")[0])
            reverse = True
        case _:
            raise ValueError(
                f"Unknown checkpoint type. Expected 'best' or 'last', but got '{ckpt}'."
            )
    ckpt = sorted(model_path.glob(f"params_epoch=*.pt"), key=sort_key, reverse=reverse)[
        0
    ]
    logger.info(f"Loading model from\n\t{ckpt}")

    checkpoint = torch.load(ckpt, map_location="cpu")
    flow = create_realnvp_flow(**checkpoint["network_config"])
    flow.load_state_dict(checkpoint["model"])
    return flow


if __name__ == "__main__":

    my_logging.logger.divider()

    train_config = modelConfig(
        **dict(
            model_name="LDM-RQ-NSF-II",
            device="cuda:0",
            batch_size=4096,
            epochs=60,
            lr=5e-3,
            scheduler="CyclicLR",
            scheduler_kwargs=dict(
                base_lr=1e-5,
                max_lr=5e-4,
                step_size_up=4000,
                mode="triangular2",
                scale_mode="cycle",
                cycle_momentum=False,
            ),
            pickup=False,
            override=True,
        )
    )
    network_config = modelConfig(
        **dict(
            hidden_features=128,
            num_blocks=4,
            num_layers=12,
            num_splits=3,
            context_dim=6,
            coupling_transform="PiecewiseRationalQuadraticCouplingTransform",
            coupling_transform_kwargs={
                "num_bins": 20,
                "tail_bound": 5.0,
                "tails": "linear",
            },
        )
    )
    train_config.pretty_print()
    network_config.pretty_print()

    my_logging.logger.divider()

    # Load data
    x_train, x_valid, x_test = load_datasets()
    train_set = TensorDataset(x_train)
    train_loader = DataLoader(
        train_set, batch_size=train_config.batch_size, shuffle=True
    )

    valid_set = TensorDataset(x_valid)
    valid_loader = DataLoader(
        valid_set, batch_size=train_config.batch_size, shuffle=False
    )

    with wandb.init(
        project="LDM-RealNVP",
        dir=paths.ANALYSIS_PARENT / "wandb",
        config={
            "train_config": train_config.param_dict,
            "network_config": network_config.param_dict,
        },
        name=train_config.model_name,
    ) as run:

        flow = create_realnvp_flow(**network_config.param_dict).to(train_config.device)
        train(
            flow, train_loader, valid_loader, network_config, **train_config.param_dict
        )
