import math
from fractions import Fraction

import torch
import numpy as np

import glori.inference.context.context_maps_utils as context_maps_utils
import glori.settings.paths as paths
from glori.infra.logging import get_logger

logger = get_logger("ctxt_maps")


def text_to_context_map(
    text,
    ranges=((None, None), (None, None), (None, None)),
    size_quadrants=(2, 2),
    font_size="auto",
    px_separation=8,
    latent_size=64,
    hist3d_file=paths.LOFAR_DATA_PARENT_HOPPER / "lotss_dr3_histogram3d_data.npz",
):

    if not hist3d_file.exists():
        raise FileNotFoundError(
            f"Histogram data file not found at {hist3d_file}. Please run the notebook 'icm_sampling_text.ipynb' to generate it."
        )

    # Create the text mask
    logger.info(
        f"Creating text mask for '{text}' with font size {font_size} and separation {px_separation}px."
    )
    size = tuple(latent_size // 2 * q for q in size_quadrants)
    text_mask = context_maps_utils.text_symbol_to_mask(
        text, size=size, font_size=font_size
    )

    # Multiply with dot pattern to ensure pixel separation
    logger.info(f"Applying dot pattern with separation {px_separation}px to text mask.")
    dots = np.zeros_like(text_mask)
    offset = px_separation // 2
    dots[offset:-offset:px_separation, offset:-offset:px_separation] = 1
    text_mask = text_mask * torch.from_numpy(dots)

    # Sample context values
    logger.info(f"Loading histogram data from \n\t{hist3d_file}...")
    counts3d = np.load(hist3d_file)["counts3d"]
    bins_list = list(np.load(hist3d_file)["bins_arr"])

    logger.info(
        f"Sampling context values from histogram for {text_mask.sum().item()} sources..."
    )
    samples = context_maps_utils.sample_histdd(
        counts3d,
        bins_list,
        n_samples=text_mask.sum().item(),
        ranges=ranges,
    )

    # Create the context map
    logger.info("Creating context map...")
    ctxt_map = text_mask.unsqueeze(0).repeat(4, 1, 1).to(torch.float32)
    ctxt_map[0] = ctxt_map[0] * 2 - 1
    ctxt_map[1:][:, text_mask.bool()] *= torch.from_numpy(samples.T)

    return ctxt_map


def explorer_context_map(
    ftot,
    fpeak,
    maj,
    size_quadrants=(2, 3),
    latent_size=64,
    i_product=2,
    logger=None,
):
    def info(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    values = {"Total Flux": ftot, "Peak Flux": fpeak, "Size": maj}

    map_size = tuple(latent_size // 2 * q for q in size_quadrants)
    keys_arr = np.array(list(values.keys()))
    haslen = np.array([hasattr(value, "__len__") for value in values.values()])
    match haslen.sum():
        case 0:
            info(
                "Only single values passed, will default to one source per quadrant with constant conditioning."
            )
            N = math.prod(size_quadrants)
            map_vals = torch.tensor(list(values.values()))

        case 1:
            info(f"Only {keys_arr[haslen]} is array, will vary across quadrants.")
            N = len(values[keys_arr[haslen][0]])
            map_vals = torch.stack(
                [
                    (
                        torch.tensor(values[k])
                        if haslen[i]
                        else torch.tensor([values[k]] * N)
                    )
                    for i, k in enumerate(keys_arr)
                ],
                dim=-1,
            )

        case 2:
            if i_product is not None and haslen[i_product]:
                info(
                    f"Will make product of {keys_arr[haslen]} with {keys_arr[~haslen]} constant."
                )
                prod_dim = tuple(len(values[k]) for k in keys_arr[haslen])
                if (fp := Fraction(*prod_dim)) != (fs := Fraction(*size_quadrants)):
                    (logger.warning if logger is not None else print)(
                        f"Product dimensions {fp} do not match quadrant sizes {fs}."
                        " This will lead to weird arrangements in the output images."
                    )
                N = math.prod(prod_dim)
                map_vals = torch.cat(
                    [
                        context_maps_utils.cartesian_outer_prod_nd(
                            *[
                                torch.tensor(values[k]).unsqueeze(1)
                                for k in keys_arr[haslen]
                            ]
                        ),
                        torch.tensor(
                            [values[k] for k in keys_arr[~haslen]] * N
                        ).unsqueeze(1),
                    ],
                    dim=-1,
                )
                # Map vals are now in order of (haslen..., not haslen...)
                # This is equivalent to a permutation by argsort(~haslen),
                # another argsort gives the inverse permutation
                map_vals = map_vals.T[np.argsort(np.argsort(~haslen))].T

            else:
                assert len(set(len(values[k]) for k in keys_arr[haslen])) == 1, (
                    "All values meant to be zipped must have the same length, got "
                    f"{[len(values[k]) for k in keys_arr[haslen]]}."
                )
                info(
                    f"Will zip {keys_arr[haslen]} across quadrants with {keys_arr[~haslen]} constant."
                )
                N = len(values[keys_arr[haslen][0]])
                map_vals = torch.stack(
                    [
                        (
                            torch.tensor(values[k])
                            if haslen[i]
                            else torch.tensor([values[k]] * N)
                        )
                        for i, k in enumerate(keys_arr)
                    ],
                    dim=-1,
                )

        case 3:
            i_product_flag = np.array([i == i_product for i in range(3)], dtype=bool)

            if i_product is not None:
                assert (
                    len(set(len(values[k]) for k in keys_arr[~i_product_flag])) == 1
                ), (
                    "All values meant to be zipped must have the same length, got "
                    f"{[len(values[k]) for k in keys_arr[~i_product_flag]]}."
                )
                info(
                    f"Will make product of {keys_arr[i_product_flag]} and zipped {keys_arr[~i_product_flag]}."
                )
                prod_dim = len(values[keys_arr[i_product]]), len(
                    values[keys_arr[~i_product_flag][0]]
                )
                if (fp := Fraction(*prod_dim)) != (fs := Fraction(*size_quadrants)):
                    (logger.warning if logger is not None else print)(
                        f"Product dimensions {fp} do not match quadrant sizes {fs}."
                        " This will lead to weird arrangements in the output images."
                    )
                N = math.prod(prod_dim)
                map_vals = context_maps_utils.cartesian_outer_prod_nd(
                    torch.stack(
                        [torch.tensor(values[k]) for k in keys_arr[~i_product_flag]],
                        dim=-1,
                    ),
                    torch.tensor(values[keys_arr[i_product]]).unsqueeze(1),
                )
                # See explanation in case 2
                map_vals = map_vals.T[np.argsort(np.argsort(i_product_flag))].T

            else:
                assert len(set(list(len(values[k]) for k in keys_arr))) == 1, (
                    "If all values should be zipped they must have same length, got "
                    f"{[len(values[k]) for k in keys_arr[haslen]]}."
                )
                info(f"Will zip all {keys_arr} across quadrants.")
                N = len(values[keys_arr[0]])
                map_vals = torch.stack(
                    [torch.tensor(v) for v in values.values()], dim=-1
                )

    grid = context_maps_utils.best_point_grid(N, *map_size)
    if (s := grid.sum(dim=1))[s != 0].min() != s.max():
        (logger.warning if logger is not None else print)(
            f"Number of points {N} cannot be perfectly distributed across map size ratio {Fraction(*size_quadrants)}.\n"
            f"Distribution is {Fraction(int(grid.sum(dim=0).max().item()), int(s.max().item()))}, but last row has only {int(s.min().item())} points."
        )
    map = grid.unsqueeze(-1).repeat(1, 1, 4).flatten(0, 1)
    map[map[:, 0].to(bool), 1:] *= map_vals
    map = map.reshape(*map_size, 4).permute(2, 0, 1)
    map[0] = map[0] * 2 - 1  # Normalize first channel to [-1, 1]

    return map


def dice_context_map(self, ftot, fpeak, fmax):
    ctxt = (
        # Make mask
        context_maps_utils.dice_pos_mask(self.config.latent_size, spacing="thirds")
        # Arrange into 2x3 latents map + add dimensions
        .view(2, 3, self.config.latent_size, self.config.latent_size)
        .permute(0, 2, 1, 3)
        .reshape(1, 1, 2 * self.config.latent_size, 3 * self.config.latent_size)
        .repeat(1, 4, 1, 1)  # Repeat for all four channels
    )

    # Scale context channels
    ctxt[:, 0] = ctxt[:, 0] * 2 - 1  # Scale position context
    ctxt[:, 1:] *= torch.tensor([ftot, fpeak, fmax]).reshape(-1, 1, 1)
    return ctxt
