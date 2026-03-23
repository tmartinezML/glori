import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
import torch
from tqdm import tqdm

from data.obs.micromaps.utils import expand_context_map
from data.trf.functional import make_catalog_context_value_scale

from utils.my_logging import get_logger

logger = get_logger("ldm-match")


def match_sources(
    resample_result,
    ctxt_scaled=False,
    match_radius_arcsec=10,
    use_original_bdsf_results=False,
):

    # If used, assert original BDSF results are present
    if use_original_bdsf_results:
        assert (
            len(resample_result["orig_bdsf_results"]) > 0
        ), "Original BDSF results are required but not available."

    # Extract data from resample result
    logger.info("Extracting data from resample result...")
    srls = (
        resample_result["srls"]
        if not use_original_bdsf_results
        else [r["catalogs"]["srl"] for r in resample_result["orig_bdsf_results"]]
    )
    wcs = resample_result["wcs"]
    pos_masks = resample_result.get("pos_masks", None)

    # Concatenate input catalogs if they exist
    input_catalogs = resample_result["input_catalogs"]

    # Expand context map to match image dimensions
    ctxt_map_exp = expand_context_map(resample_result["ctxt_map"], f_upscale=4)
    # If working with scaled context values, we need to invert the scaling
    if ctxt_scaled:
        ctxt_map_exp = make_catalog_context_value_scale(inverse=True)(
            torch.tensor(ctxt_map_exp)
        ).numpy()

    # prepare to collect all matched/unmatched sources
    all_inp = []
    all_outp = []
    all_d2d = []
    match_dicts = []
    input_cat = None  # Start value, will be filled in loop

    # Loop through output source lists
    logger.info("Matching sources...")
    for i in tqdm(range(len(srls)), total=len(srls)):

        # Get output catalog for this image
        srl = srls[i]

        # if we have no position masks, we get the input catalog from the context map
        if pos_masks is None:

            # Get context map for this image
            ctxt = ctxt_map_exp[i]

            # Extract input catalog from context map
            # These will all be None if no sources are present on input
            input_coords, inp_vals, input_subcat = input_cat_from_context_map(ctxt, wcs)

        # If otherwise we have position masks, that means we also have input catalogs
        else:
            pos_mask = pos_masks[i]
            input_coords = SkyCoord(
                wcs.array_index_to_world_values(
                    np.flip(np.argwhere(pos_mask == 1), axis=1)
                ),
                unit="deg",
            )
            nsrc = int(pos_mask.sum())
            input_subcat = (
                input_catalogs.iloc[nsrc * i : nsrc * (i + 1)]
                .copy()
                .reset_index(drop=True)
            )
            input_subcat["RA"] = input_coords.ra.deg
            input_subcat["DEC"] = input_coords.dec.deg
            inp_vals = np.array(
                [input_subcat[col].values for col in ["Total_flux", "Peak_flux", "Maj"]]
            )

        # Skip the rest if no sources are on input (this is faulty input image)
        if input_coords is None or len(input_coords) == 0:
            continue

        # Append to overall input catalog
        input_cat = (
            pd.concat([input_cat, input_subcat], ignore_index=True)
            if input_cat is not None
            else input_subcat
        )

        # If no sources are found in image, srl is None
        if srl is None:
            # All input is unmatched
            match_dict = {
                "matched_in": pd.DataFrame({}),
                "unmatched_in": input_cat,
                "matched_out": pd.DataFrame({}),
                "unmatched_out": pd.DataFrame({}),
            }
            match_dicts.append(match_dict)
            all_inp.append(inp_vals)
            all_outp.append(np.zeros((3, 0)))
            all_d2d.append(np.array([]))
            continue

        # Define coordinates for output sources
        srl_coords = SkyCoord(ra=srl["RA"], dec=srl["DEC"], unit="deg")

        # idx are indices into input_coords that correspond to the
        # nearest neighbor of srl_coords (i.e. len(idx) = len(srl_coords))
        idx, d2d, _ = srl_coords.match_to_catalog_sky(input_coords)

        # Apply distance threshold
        dist_mask = d2d.to(u.arcsec).value <= match_radius_arcsec  # arcsec

        # Count occurrences of each index to handle duplicates (or multiples)
        unique_idx, count = np.unique(idx[dist_mask], return_counts=True)

        # Make dictionary with matched and unmatched catalogs for both input and output
        match_dict = {
            "matched_in": input_subcat.iloc[unique_idx],
            "unmatched_in": input_subcat.drop(index=unique_idx),
            "matched_out": srl.iloc[dist_mask],
            "unmatched_out": srl.drop(index=np.where(dist_mask)[0]),
        }
        match_dicts.append(match_dict)

        # Output exists for all indices presend in idx
        out_vals = np.zeros((len(unique_idx), 3))

        # Sources with multiple counts are merged:
        # - Total_flux is summed
        # - Peak_flux uses the maximum
        # - Maj uses the maximum (this is crappy tho)
        for j, i in enumerate(unique_idx):
            sub_srl = srl[idx == i]
            out_vals[j] = [
                sub_srl["Total_flux"].sum(),
                sub_srl["Peak_flux"].max(),
                sub_srl["Maj"].max(),
            ]

        # Collect matched input and output values and distances
        all_inp.append(inp_vals[:, unique_idx])
        all_outp.append(out_vals.T)
        all_d2d.append(d2d.to(u.arcsec).value)

    # Concatenate all collected data, i.e. convert lists to arrays
    logger.info("Concatenating matched data...")
    combined_dict = dict(
        all_inp=np.concatenate(all_inp, axis=1),
        all_outp=np.concatenate(all_outp, axis=1),
        all_d2d=np.concatenate(all_d2d, axis=0),
    )

    logger.info("Done matching sources.\n\n" + match_summary(match_dicts))
    return combined_dict, match_dicts, input_cat


def match_summary(match_dicts):
    ljust_len = 35
    return (
        str.rjust("Total input sources:\t", ljust_len)
        + f"{sum([len(md['matched_in']) + len(md['unmatched_in']) for md in match_dicts])}\n"
        + str.rjust("Total output sources:\t", ljust_len)
        + f"{sum([len(md['matched_out']) + len(md['unmatched_out']) for md in match_dicts])}\n"
        + str.rjust("Total matched sources:\t", ljust_len)
        + f"{sum([len(md['matched_in']) for md in match_dicts])}\n"
        + str.rjust("Total unmatched input sources:\t", ljust_len)
        + f"{sum([len(md['unmatched_in']) for md in match_dicts])}\n"
        + str.rjust("Total unmatched output sources:\t", ljust_len)
        + f"{sum([len(md['unmatched_out']) for md in match_dicts])}\n"
    )


def input_cat_from_context_map(ctxt, wcs):

    # Return empty if no sources
    if (ctxt[0] == 1).sum() == 0:
        return None, None, None
        input_coords = SkyCoord([], unit="deg")
        inp_vals = np.zeros((3, 0))
        input_subcat = pd.DataFrame(
            {
                "RA": [],
                "DEC": [],
                "Total_flux": [],
                "Peak_flux": [],
                "Maj": [],
            }
        )
        return input_coords, inp_vals, input_subcat

    # Convert array indices to world coordinates
    input_coords = SkyCoord(
        wcs.array_index_to_world_values(np.flip(np.argwhere(ctxt[0] == 1), axis=1)),
        # wcs.array_index_to_world_values(np.argwhere(ctxt[0] == 1)),
        unit="deg",
    )

    # Extract input source values from context map
    input_ftot, input_fpeak, input_maj = inp_vals = np.array(
        [ctxt[j][ctxt[0] == 1] for j in range(1, 4)]
    )

    # Create input source sub-catalog
    input_subcat = pd.DataFrame(
        {
            "RA": input_coords.ra.deg,
            "DEC": input_coords.dec.deg,
            "Total_flux": input_ftot,
            "Peak_flux": input_fpeak,
            "Maj": input_maj,
        }
    )
    return input_coords, inp_vals, input_subcat
