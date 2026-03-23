import os
from shutil import copyfile

import bdsf
import numpy as np
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from photutils.aperture import SkyEllipticalAperture

import glori.settings.paths as paths
import glori.infra.logging as my_logging

logger = my_logging.get_logger(__name__)


def run_tiered_bdsf(
    imf,
    appf,
    thresh_pix=4.5,
    label="",
    catprefix="mosaic",
    quiet=False,
    bdsf_kw={},
    # folder=paths.ANALYSIS_PARENT / "bdsf",
):
    # Tiered source finding code adapted from Catherine Hale code.
    # For mosaics put imf and appf as the same image
    # For individual pointings imf and appf are int and app images.
    # Use thres_pix=4.0 for deep fields and 4.5 for LoTSS fields
    bdsf_kw["quiet"] = quiet
    if quiet:
        logger.setLevel(40)

    logger.separator("Create Mean 0 map")

    # folder = folder + "/"
    folder = ""

    intermediate_products = []

    imagef = folder + imf
    imageappf = folder + appf
    meanf = imf[:-5] + "_Mean0.fits"

    copyfile(imagef, meanf)

    d = fits.open(meanf, mode="update")
    d[0].data *= 0
    d.flush()
    d.close()
    intermediate_products.append(meanf)

    pbimagef = imf[:-5] + "_PB.fits"
    copyfile(imagef, pbimagef)
    d = fits.open(pbimagef, mode="update")
    openimagef = fits.open(imagef)
    openimageappf = fits.open(imageappf)
    d[0].data = openimageappf[0].data / openimagef[0].data
    d.flush()
    d.close()
    intermediate_products.append(pbimagef)

    # -----------------------------------------------------
    logger.separator("Run initial PyBDSF")
    # -----------------------------------------------------

    restfrq = 144000000.0

    img = bdsf.process_image(
        imf,
        detection_image=appf,
        thresh_isl=3.0,
        thresh_pix=thresh_pix,
        rms_box=(150, 15),
        rms_map=True,
        mean_map="zero",
        ini_method="intensity",
        adaptive_rms_box=True,
        adaptive_thresh=150,
        rms_box_bright=(60, 15),
        group_by_isl=False,
        group_tol=10.0,
        output_opts=True,
        output_all=False,
        atrous_do=True,
        atrous_jmax=4,
        flagging_opts=True,
        flag_maxsize_fwhm=0.5,
        advanced_opts=True,
        blank_limit=None,
        frequency=restfrq,
        **bdsf_kw,
    )
    img.export_image(
        outfile=imagef[:-5] + "-Default-" + label + ".rms.fits",
        img_format="fits",
        img_type="rms",
        clobber=True,
    )
    img.export_image(
        outfile=imagef[:-5] + "-Default-" + label + ".resid.fits",
        img_format="fits",
        img_type="gaus_resid",
        clobber=True,
    )
    img.write_catalog(
        outfile=imagef[:-5] + "-Default-" + label + ".srl.fits",
        format="fits",
        catalog_type="srl",
        clobber=True,
    )
    img.write_catalog(
        outfile=imagef[:-5] + "-Default-" + label + ".gaul.fits",
        format="fits",
        catalog_type="gaul",
        clobber=True,
    )
    img.export_image(
        outfile=imagef[:-5] + "-Default-" + label + ".model.fits",
        img_format="fits",
        img_type="gaus_model",
        clobber=True,
    )
    img.export_image(
        outfile=imagef[:-5] + "-Default-" + label + ".mask.fits",
        img_format="fits",
        img_type="island_mask",
        clobber=True,
    )

    intermediate_products.append(imagef[:-5] + "-Default-" + label + ".rms.fits")
    intermediate_products.append(imagef[:-5] + "-Default-" + label + ".resid.fits")
    intermediate_products.append(imagef[:-5] + "-Default-" + label + ".srl.fits")
    intermediate_products.append(imagef[:-5] + "-Default-" + label + ".gaul.fits")
    intermediate_products.append(imagef[:-5] + "-Default-" + label + ".model.fits")
    intermediate_products.append(imagef[:-5] + "-Default-" + label + ".mask.fits")

    # -----------------------------------------------------
    logger.separator("Add Bright Sources back into field")
    # -----------------------------------------------------

    snr_thresh = 150
    source_dat = Table.read(imagef[:-5] + "-Default-" + label + ".srl.fits")
    gaus_dat = Table.read(imagef[:-5] + "-Default-" + label + ".gaul.fits")

    bright_sources = Table(
        source_dat[source_dat["Peak_flux"] / source_dat["Isl_rms"] >= snr_thresh],
        copy=True,
    )
    bright_sources.write(
        imagef[:-5] + "-Bright_Sources_SNR_%i-%s.fits" % (snr_thresh, label),
        overwrite=True,
    )

    intermediate_products.append(
        imagef[:-5] + "-Bright_Sources_SNR_%i-%s.fits" % (snr_thresh, label)
    )

    print("Number of bright sources: ", len(bright_sources))

    bright_gaus = Table()
    for i in range(len(bright_sources)):
        bright_gaus = vstack(
            [
                bright_gaus,
                gaus_dat[gaus_dat["Source_id"] == bright_sources["Source_id"][i]],
            ]
        )

    bright_gaus.write(
        imagef[:-5] + "-Bright_Gaussians_SNR_%i-%s.fits" % (snr_thresh, label),
        overwrite=True,
    )

    intermediate_products.append(
        imagef[:-5] + "-Bright_Gaussians_SNR_%i-%s.fits" % (snr_thresh, label)
    )

    model_im_new_f = imagef[:-5] + "-Default-" + label + ".model-Mask-Bright.fits"

    wcs_im = WCS(imagef[:-5] + "-Default-" + label + ".model.fits", naxis=2)

    model_im_new = flatten(fits.open(imagef[:-5] + "-Default-" + label + ".model.fits"))

    # -----------------------------------------------------
    logger.separator("Model mask regions (i.e. regions where model is not zero)")
    # -----------------------------------------------------

    model_mask_final = np.zeros(
        (np.shape(model_im_new.data)[0], np.shape(model_im_new.data)[1])
    )
    for i in range(len(bright_gaus)):
        position = SkyCoord(
            ra=bright_gaus["RA"][i], dec=bright_gaus["DEC"][i], unit="deg"
        )
        aper = SkyEllipticalAperture(
            position,
            1.5 * bright_gaus["Maj"][i] * u.degree,
            1.5 * bright_gaus["Min"][i] * u.degree,
            bright_gaus["PA"][i] * u.degree,
        )
        pix_aper = aper.to_pixel(wcs_im)
        m = pix_aper.to_mask()
        out = m.to_image(
            (np.shape(model_im_new.data)[0], np.shape(model_im_new.data)[1])
        )
        model_mask_final += out

    model_mask_final[model_mask_final != 0] = 1

    model_im_new.data = 1 - model_mask_final
    intermediate_products.append(model_im_new_f)
    model_im_new.writeto(model_im_new_f, overwrite=True)

    # -----------------------------------------------------
    logger.separator("Make final residual map with bright sources in")
    # -----------------------------------------------------

    model_im_orig = fits.getdata(imagef[:-5] + "-Default-" + label + ".model.fits")
    model_im_no_brightf = (
        imagef[:-5] + "-Default-" + label + ".model-Bright-Removed.fits"
    )

    intermediate_products.append(imagef[:-5] + "-Default-" + label + ".model.fits")

    copyfile(imagef[:-5] + "-Default-" + label + ".model.fits", model_im_no_brightf)
    intermediate_products.append(model_im_no_brightf)

    model_im_no_bright = fits.open(model_im_no_brightf, mode="update")
    model_im_no_bright[0].data = model_im_new.data * model_im_orig
    model_im_no_bright.flush()
    model_im_no_bright.close()

    resid_im_no_brightf = imf[:-5] + "-Default-" + label + ".resid-Bright-Remain.fits"
    copyfile(imagef[:-5] + "-Default-" + label + ".resid.fits", resid_im_no_brightf)
    intermediate_products.append(resid_im_no_brightf)

    full_im_orig = fits.getdata(folder + imf)
    resid_im_no_bright = fits.open(resid_im_no_brightf, mode="update")
    resid_im_no_bright[0].data = full_im_orig - model_im_no_bright[0].data
    resid_im_no_bright.flush()
    resid_im_no_bright.close()

    # -----------------------------------------------------
    logger.separator("Run PyBDSF on Residual")
    # -----------------------------------------------------

    resid_im_no_bright_appf = (
        appf[:-5] + "-Default-" + label + ".resid-Bright-Remain.app.fits"
    )

    create_pb_image(resid_im_no_brightf, resid_im_no_bright_appf, pbimagef)
    intermediate_products.append(resid_im_no_bright_appf)

    img_resid = bdsf.process_image(
        resid_im_no_brightf,
        detection_image=resid_im_no_bright_appf,
        thresh_isl=3.0,
        thresh_pix=thresh_pix,
        rms_box=(150, 15),
        rms_map=True,
        mean_map="zero",
        ini_method="intensity",
        adaptive_rms_box=True,
        adaptive_thresh=150,
        rms_box_bright=(60, 15),
        group_by_isl=False,
        group_tol=10.0,
        output_opts=True,
        output_all=False,
        atrous_do=True,
        atrous_jmax=4,
        flagging_opts=True,
        flag_maxsize_fwhm=0.5,
        advanced_opts=True,
        blank_limit=None,
        frequency=restfrq,
        **bdsf_kw,
    )

    rms_image_withBright_f = resid_im_no_brightf[:-5] + ".rms.fits"
    img_resid.export_image(
        outfile=rms_image_withBright_f, img_format="fits", img_type="rms", clobber=True
    )
    intermediate_products.append(rms_image_withBright_f)

    # -----------------------------------------------------
    logger.separator("Run PyBDSF - supplying RMS image")
    # -----------------------------------------------------

    rms_image_withBright_appf = rms_image_withBright_f.replace(".fits", ".app.fits")
    intermediate_products.append(rms_image_withBright_appf)
    create_pb_image(rms_image_withBright_f, rms_image_withBright_appf, pbimagef)

    img_supply = bdsf.process_image(
        imf,
        detection_image=appf,
        thresh_isl=3.0,
        thresh_pix=thresh_pix,
        rms_box=(150, 15),
        rms_map=True,
        mean_map="zero",
        ini_method="intensity",
        adaptive_rms_box=True,
        adaptive_thresh=150,
        rms_box_bright=(60, 15),
        group_by_isl=False,
        group_tol=10.0,
        output_opts=True,
        output_all=False,
        atrous_do=True,
        atrous_jmax=4,
        flagging_opts=True,
        flag_maxsize_fwhm=0.5,
        advanced_opts=True,
        blank_limit=None,
        frequency=restfrq,
        rmsmean_map_filename=[
            os.path.basename(meanf),
            os.path.basename(rms_image_withBright_f),
        ],
        rmsmean_map_filename_det=[
            os.path.basename(meanf),
            os.path.basename(rms_image_withBright_appf),
        ],
        **bdsf_kw,
    )
    img_supply.export_image(
        outfile=imagef[:-5] + "-Default-" + label + "-SupplyMaps.rms.fits",
        img_format="fits",
        img_type="rms",
        clobber=True,
    )
    img_supply.export_image(
        outfile=imagef[:-5] + "-Default-" + label + "-SupplyMaps.resid.fits",
        img_format="fits",
        img_type="gaus_resid",
        clobber=True,
    )
    img_supply.write_catalog(
        outfile=imagef[:-5] + "-Default-" + label + "-SupplyMaps.srl.fits",
        format="fits",
        catalog_type="srl",
        clobber=True,
    )
    img_supply.write_catalog(
        outfile=imagef[:-5] + "-Default-" + label + "-SupplyMaps.gaul.fits",
        format="fits",
        catalog_type="gaul",
        clobber=True,
    )
    img_supply.export_image(
        outfile=imagef[:-5] + "-Default-" + label + "-SupplyMaps.model.fits",
        img_format="fits",
        img_type="gaus_model",
        clobber=True,
    )
    img_supply.export_image(
        outfile=imagef[:-5] + "-Default-" + label + "-SupplyMaps.mask.fits",
        img_format="fits",
        img_type="island_mask",
        clobber=True,
    )

    intermediate_products.append(
        imagef[:-5] + "-Default-" + label + "-SupplyMaps.rms.fits"
    )
    intermediate_products.append(
        imagef[:-5] + "-Default-" + label + "-SupplyMaps.resid.fits"
    )
    intermediate_products.append(
        imagef[:-5] + "-Default-" + label + "-SupplyMaps.srl.fits"
    )
    intermediate_products.append(
        imagef[:-5] + "-Default-" + label + "-SupplyMaps.gaul.fits"
    )
    intermediate_products.append(
        imagef[:-5] + "-Default-" + label + "-SupplyMaps.model.fits"
    )
    intermediate_products.append(
        imagef[:-5] + "-Default-" + label + "-SupplyMaps.mask.fits"
    )

    # -----------------------------------------------------
    logger.separator("Extract Bright sources from Residual")
    # -----------------------------------------------------

    int_residf = imf[:-5] + "-Default-" + label + "-SupplyMaps.resid.fits"

    app_residf = int_residf.replace(".fits", ".app.fits")
    intermediate_products.append(app_residf)
    create_pb_image(int_residf, app_residf, pbimagef)

    img_supply_large = bdsf.process_image(
        int_residf,
        detection_image=app_residf,
        thresh_isl=3.0,
        thresh_pix=10.0,
        rms_box=(150, 15),
        rms_map=True,
        mean_map="zero",
        ini_method="intensity",
        adaptive_rms_box=True,
        adaptive_thresh=150,
        rms_box_bright=(60, 15),
        group_by_isl=False,
        group_tol=10.0,
        output_opts=True,
        output_all=False,
        flagging_opts=True,
        flag_maxsize_fwhm=0.5,
        advanced_opts=True,
        blank_limit=None,
        frequency=restfrq,
        rmsmean_map_filename=[
            os.path.basename(meanf),
            os.path.basename(rms_image_withBright_f),
        ],
        rmsmean_map_filename_det=[
            os.path.basename(meanf),
            os.path.basename(rms_image_withBright_appf),
        ],
        flag_maxsize_bm=100,
        **bdsf_kw,
    )
    img_supply_large.write_catalog(
        outfile=int_residf + "-Default-" + label + "-SupplyMaps-FlagBeam.gaul.fits",
        format="fits",
        catalog_type="gaul",
        clobber=True,
    )
    img_supply_large.write_catalog(
        outfile=int_residf + "-Default-" + label + "-SupplyMaps-FlagBeam.srl.fits",
        format="fits",
        catalog_type="srl",
        clobber=True,
    )
    img_supply_large.export_image(
        outfile=int_residf + "-Default-" + label + "-SupplyMaps-FlagBeam.model.fits",
        img_format="fits",
        img_type="gaus_model",
        clobber=True,
    )
    img_supply_large.export_image(
        outfile=int_residf + "-Default-" + label + "-SupplyMaps-FlagBeam.mask.fits",
        img_format="fits",
        img_type="island_mask",
        clobber=True,
    )

    intermediate_products.append(
        int_residf + "-Default-" + label + "-SupplyMaps-FlagBeam.gaul.fits"
    )
    intermediate_products.append(
        int_residf + "-Default-" + label + "-SupplyMaps-FlagBeam.srl.fits"
    )
    intermediate_products.append(
        int_residf + "-Default-" + label + "-SupplyMaps-FlagBeam.model.fits"
    )
    intermediate_products.append(
        int_residf + "-Default-" + label + "-SupplyMaps-FlagBeam.mask.fits"
    )

    # -----------------------------------------------------
    logger.separator("Update residual with large sources")
    # -----------------------------------------------------

    resid_ext_emf = imagef + "-Default-" + label + "-SupplyMaps-withFlagBeam.resid.fits"
    model_final = imagef + "-Default-" + label + "-SupplyMaps-withFlagBeam.model.fits"

    model_flagged = int_residf + "-Default-" + label + "-SupplyMaps-FlagBeam.model.fits"
    model_original = imagef[:-5] + "-Default-" + label + "-SupplyMaps.model.fits"

    model_flag_dat = fits.open(model_flagged)
    model_orig_dat = fits.open(model_original)

    copyfile(model_original, model_final)
    copyfile(
        imagef[:-5] + "-Default-" + label + "-SupplyMaps.resid.fits", resid_ext_emf
    )
    intermediate_products.append(resid_ext_emf)

    model_withflag_dat = fits.open(model_final, mode="update")
    resid_withflag_dat = fits.open(resid_ext_emf, mode="update")

    model_withflag_dat[0].data += model_flag_dat[0].data
    model_withflag_dat.flush()

    resid_withflag_dat[0].data -= model_flag_dat[0].data
    resid_withflag_dat.flush()

    model_withflag_dat.close()
    resid_withflag_dat.close()

    # -----------------------------------------------------
    logger.separator("Update Catalogue")
    # -----------------------------------------------------

    dat_srl_orig = Table.read(
        imagef[:-5] + "-Default-" + label + "-SupplyMaps.srl.fits"
    )
    dat_gaul_orig = Table.read(
        imagef[:-5] + "-Default-" + label + "-SupplyMaps.gaul.fits"
    )
    # Bonny's modification to prevent a rare crash
    try:
        dat_srl_flagbeam = Table.read(
            int_residf + "-Default-" + label + "-SupplyMaps-FlagBeam.srl.fits"
        )
        dat_gaul_flagbeam = Table.read(
            int_residf + "-Default-" + label + "-SupplyMaps-FlagBeam.gaul.fits"
        )

    except FileNotFoundError:

        dat_srl_orig["Flag_beam"] = np.zeros(len(dat_srl_orig), dtype=int)
        dat_gaul_orig["Flag_beam"] = np.zeros(len(dat_gaul_orig), dtype=int)
        dat_srl_final = dat_srl_orig
        dat_gaul_final = dat_gaul_orig

    else:
        max_src_id = np.max(dat_gaul_orig["Source_id"]) + 1
        max_isl_id = np.max(dat_gaul_orig["Isl_id"]) + 1
        max_gaus_id = np.max(dat_gaul_orig["Gaus_id"]) + 1

        dat_srl_flagbeam["Source_id"] += max_src_id + 1
        dat_srl_flagbeam["Isl_id"] += max_isl_id + 1

        dat_gaul_flagbeam["Source_id"] += max_src_id + 1
        dat_gaul_flagbeam["Isl_id"] += max_isl_id + 1
        dat_gaul_flagbeam["Gaus_id"] += max_gaus_id + 1

        #

        dat_srl_orig["Flag_beam"] = np.zeros(len(dat_srl_orig), dtype=int)
        dat_gaul_orig["Flag_beam"] = np.zeros(len(dat_gaul_orig), dtype=int)

        dat_srl_flagbeam["Flag_beam"] = np.ones(len(dat_srl_flagbeam), dtype=int)
        dat_gaul_flagbeam["Flag_beam"] = np.ones(len(dat_gaul_flagbeam), dtype=int)

        #

        dat_srl_final = vstack([dat_srl_orig, dat_srl_flagbeam])
        dat_gaul_final = vstack([dat_gaul_orig, dat_gaul_flagbeam])

    # -----------------------------------------------------
    logger.separator("Write to disk")
    # -----------------------------------------------------

    dat_srl_final.write(
        imagef[:-5] + "-Default-" + label + "-SupplyMaps-withFlagBeam.srl.fits",
        overwrite=True,
    )
    dat_gaul_final.write(
        imagef[:-5] + "-Default-" + label + "-SupplyMaps-withFlagBeam.gaul.fits",
        overwrite=True,
    )

    mask1 = fits.open(imagef[:-5] + "-Default-" + label + "-SupplyMaps.mask.fits")
    mask2 = fits.open(
        int_residf + "-Default-" + label + "-SupplyMaps-FlagBeam.mask.fits"
    )

    mask1[0].data = mask1[0].data + mask2[0].data
    mask1[0].data[mask1[0].data != 0] = 1

    mask1.writeto(imagef[:-5] + "-combined-mask.fits", overwrite=True)

    # Rename other final proucts
    # Final products are:
    # mosaic-blanked-Default-P23Hetdex20-SupplyMaps.rms.fits - final rms map (this is the rms map that is supplied to the last source finding step and the rms is not recalcuated in the last step)
    # mosaic-blanked.fits-Default-P23Hetdex20-SupplyMaps-withFlagBeam.resid.fits - final resid map
    # mosaic-blanked.fits-Default-P23Hetdex20-SupplyMaps-withFlagBeam.mask.fits - final mask map
    # mosaic-blanked-Default-P23Hetdex20-SupplyMaps-withFlagBeam.srl.fits - final source catalogue
    # mosaic-blanked-Default-P23Hetdex20-SupplyMaps-withFlagBeam.gaul.fits final guassian catalogue
    finalrms = imagef[:-5] + "-Default-" + label + "-SupplyMaps.rms.fits"
    finalscat = imagef[:-5] + "-Default-" + label + "-SupplyMaps-withFlagBeam.srl.fits"
    finalgcat = imagef[:-5] + "-Default-" + label + "-SupplyMaps-withFlagBeam.gaul.fits"
    finalresid = resid_ext_emf
    finalmodel = model_final
    finalmask = imagef[:-5] + "-combined-mask.fits"

    # These are needed to read output
    os.system("mv %s %s-%s-final.srl.fits" % (finalscat, imagef[:-5], label))
    os.system("mv %s %s-%s-final.gaul.fits" % (finalgcat, imagef[:-5], label))
    os.system("mv %s %s-%s-final.model.fits" % (finalmodel, imagef[:-5], label))
    os.system("mv %s %s-%s-final.resid.fits" % (finalresid, imagef[:-5], label))

    # These will be removed
    os.system("rm %s" % (finalrms))
    os.system("rm %s" % (finalmask))

    # Tidy up
    intermediate_products.append(f"{imagef}.pybdsf.log")
    for product in intermediate_products:
        if os.path.isfile(product):
            os.system(f"rm {product}")
            if os.path.exists(f"{product}.pybdsf.log"):
                os.system(f"rm {product}.pybdsf.log")
        else:
            logger.warning("Intermediate product %s does not exist" % product)

    # Return paths of final products: catalogs, model array, residual array

    return (
        f"{imagef[:-5]}-{label}-final.srl.fits",
        f"{imagef[:-5]}-{label}-final.gaul.fits",
        f"{imagef[:-5]}-{label}-final.model.fits",
        f"{imagef[:-5]}-{label}-final.resid.fits",
    )


def create_pb_image(imagef_int, imagef_out, pbimagef):
    with fits.open(imagef_int) as dat:
        pbim = fits.getdata(pbimagef)
        dat[0].data *= pbim
        dat.writeto(imagef_out, overwrite=True)


def flatten(f):
    """Flatten a fits file so that it becomes a 2D image. Return new header and data"""

    naxis = f[0].header["NAXIS"]
    if naxis < 2:
        raise RuntimeError("Can't make map from this")
    if naxis == 2:
        f[0].header["WCSAXES"] = 2
        return fits.PrimaryHDU(header=f[0].header, data=f[0].data)

    w = WCS(f[0].header)
    wn = WCS(naxis=2)

    wn.wcs.crpix[0] = w.wcs.crpix[0]
    wn.wcs.crpix[1] = w.wcs.crpix[1]
    wn.wcs.cdelt = w.wcs.cdelt[0:2]
    wn.wcs.crval = w.wcs.crval[0:2]
    wn.wcs.ctype[0] = w.wcs.ctype[0]
    wn.wcs.ctype[1] = w.wcs.ctype[1]

    header = wn.to_header()
    header["NAXIS"] = 2
    header["WCSAXES"] = 2
    copy = (
        "EQUINOX",
        "EPOCH",
        "BMAJ",
        "BMIN",
        "BPA",
        "RESTFRQ",
        "TELESCOP",
        "OBSERVER",
    )
    for k in copy:
        r = f[0].header.get(k)
        if r is not None:
            header[k] = r

    dslice = []
    for i in range(naxis, 0, -1):
        if i <= 2:
            dslice.append(
                np.s_[:],
            )
        else:
            dslice.append(0)

    hdu = fits.PrimaryHDU(header=header, data=f[0].data[tuple(dslice)])
    return hdu
