import logging
import pathlib
from copy import copy
from typing import List, Tuple

import imageio.v3 as iio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.time import Time
from astropy.visualization import ImageNormalize, ZScaleInterval
from astropy.wcs import WCS

CMAP_BONE = matplotlib.cm.bone.copy()
CMAP_BONE.set_bad("black")

logger = logging.getLogger(__file__)


VELOCITY_VECTOR_KWARGS: dict = {
    "gap": 2,  # arcsec
    "scale_factor": 1,  # set default scale factor to be 1
    "color": "#34ebcd",
    "width": 0.1,  # arcsec
    "zorder": 10,
}
CROSSHAIR_DETECTION_KWARGS: dict = {
    "gap": 2,
    "length": 2,
    "color": "#03fc0f",
    "alpha": 1.0,
    "zorder": 9,
}
CROSSHAIR_NON_DETECTION_KWARGS: dict = copy(CROSSHAIR_DETECTION_KWARGS)
CROSSHAIR_NON_DETECTION_KWARGS["color"] = "r"
SUBPLOTS_ADJUST_KWARGS: dict = {
    "hspace": 0.15,
    "wspace": 0.15,
    "left": 0.05,
    "right": 0.95,
    "top": 0.90,
    "bottom": 0.05,
}
CIRCLE_OBJECT_KWARGS: dict = {
    "fill": False,
    "color": "#03fc0f",
    "alpha": 1.0,
    "zorder": 0.9,
}


def add_circle(
    ax: matplotlib.axes.Axes, obj_ra: float, obj_dec: float, rad: float = 2, **kwargs
):
    """
    Add a circle on the location of the detected object.
    ---------
    ax : `~matplotlib.axes.Axes`
        Matplotlib axes (usually a subplot) on which to add the crosshair.
    obj_ra : float
        RA of object detection in degrees.
    obj_dec : float
        Dec of object detection in degrees.
    **kwargs
        Keyword arguments to pass to ax.add_patch
    """

    radius = rad / 3600
    center = (obj_ra, obj_dec)

    ax.add_patch(
        matplotlib.patches.Circle(
            center, radius=radius, transform=ax.get_transform("world"), **kwargs
        )
    )

    return


def add_crosshair(
    ax: matplotlib.axes.Axes,
    ra: float,
    dec: float,
    gap: float = 2,
    length: float = 2,
    **kwargs,
):
    """
    Add a crosshair centered on RA and Dec to the given axes.

    Parameter
    ---------
    ax : `~matplotlib.axes.Axes`
        Matplotlib axes (usually a subplot) on which to add the crosshair.
    ra : float
        Predicted RA in degrees.
    dec : float
        Predicted Dec in degrees.
    gap : float
        Distance from center in arcseconds to start drawing crosshair reticle bar.
    length : float
        Length in arcseconds of an individual bar reticle.
    **kwargs
        Keyword arguments to pass to ax.hlines and ax.vlines.
    """
    # Convert to degrees
    gap_degree = gap / 3600
    length_degree = length / 3600

    # Set the number of points to draw each reticle
    n = 100

    # Set the width of the crosshair
    width_degree = 0.1 / 3600

    # Plot the top graticule of the crosshair (N)
    ax.arrow(
        ra,
        dec + gap_degree,
        0,
        length_degree,
        transform=ax.get_transform("world"),
        width=width_degree,
        head_width=4 * width_degree,
        head_length=4 * width_degree,
        length_includes_head=True,
        **kwargs,
    )
    ax.text(
        ra,
        dec + gap_degree + length_degree + 1 / 3600,
        "N",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.get_transform("world"),
        color=kwargs["color"],
        fontsize=6,
    )

    # Plot the bottom graticule of the crosshair (S)
    ras = np.ones(n) * ra
    decs = np.linspace(dec - gap_degree, dec - gap_degree - length_degree, n)
    ax.plot(ras, decs, transform=ax.get_transform("world"), **kwargs)

    # Plot the left graticule of the crosshair (E)
    ras = np.linspace(ra + gap_degree, ra + gap_degree + length_degree, n)
    decs = np.ones(n) * dec
    ax.plot(ras, decs, transform=ax.get_transform("world"), **kwargs)

    # Plot the left graticule of the crosshair (W)
    ras = np.linspace(ra - gap_degree, ra - gap_degree - length_degree, n)
    decs = np.ones(n) * dec
    ax.plot(ras, decs, transform=ax.get_transform("world"), **kwargs)
    return


def add_velocity_vector(
    ax: matplotlib.axes.Axes,
    ra: float,
    dec: float,
    vra: float,
    vdec: float,
    dt: float,
    scale_factor: float = 10,
    gap: float = 1,
    width: float = 0.1,
    **kwargs,
):
    """
    Add a velocity vector showing the predicted velocity of an object
    to an image.

    If the velocity vector is larger than the image boundary, it will be clipped and the
    color set to red.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Matplotlib axes (usually a subplot) on which to add the velocity vector.
    ra : float
        Predicted RA in degrees.
    dec : float
        Predicted Dec in degrees.
    vra : float
        Predicted RA-velocity in degrees per day.
    vdec : float
        Predicted Dec-velocity in degrees per day.
    dt : float
        Exposure duration in units of seconds. Used to scale the velocity vector.
    scale_factor : float
        Scale factor to multiply the velocity by. Used to scale the velocity vector.
    gap : float
        Distance from (RA, Dec) in arcseconds to start drawing the velocity vector.
    width : float
        Arrow tail width in arcseconds. The head will be 3x the width.
    **kwargs
        Keyword arguments to pass to ax.arrow.
    """
    if np.isnan(ra):
        raise ValueError("The RA must be finite to plot the velocity vector.")
    if np.isnan(dec):
        raise ValueError("The Dec must be finite to plot the velocity vector.")
    if np.isnan(vra):
        raise ValueError(
            "The velocity in RA must be finite to plot the velocity vector."
        )
    if np.isnan(vdec):
        raise ValueError(
            "The velocity in Dec must be finite to plot the velocity vector."
        )
    if np.isnan(dt):
        raise ValueError(
            "The exposure duration must be finite to plot the velocity vector."
        )

    # Calculate the unit vector in the direction of the velocity
    vra_hat = vra / np.sqrt(vra**2 + vdec**2)
    vdec_hat = vdec / np.sqrt(vra**2 + vdec**2)

    # Convert to the correct units
    gap_degree = gap / 3600
    dra = vra * (dt / 86400) * scale_factor
    ddec = vdec * (dt / 86400) * scale_factor
    width_degree = width / 3600

    # Calculate the starting point of the arrow
    ra0 = ra + vra_hat * gap_degree
    dec0 = dec + vdec_hat * gap_degree

    # Store boundaries of the image
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    # Add the velocity vector
    arrow = ax.arrow(
        ra0,
        dec0,
        dra,
        ddec,
        width=width_degree,
        head_width=3 * width_degree,
        head_length=3 * width_degree,
        transform=ax.get_transform("world"),
        length_includes_head=True,
        **kwargs,
    )

    if ax.get_xlim() != x_lim or ax.get_ylim() != y_lim:
        # If the arrow is outside the image boundaries then
        # change the color to red
        arrow.set_color("red")

        # and reset the image boundaries
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    return


def plot_cutout(
    ax: matplotlib.axes.Axes,
    image: npt.NDArray[np.float64],
    ra: float,
    dec: float,
    vra: float,
    vdec: float,
    obj_ra: float,
    obj_dec: float,
    dt: float,
    circle_object: bool = True,
    circle_object_kwargs: dict = CIRCLE_OBJECT_KWARGS,
    crosshair: bool = True,
    crosshair_kwargs: dict = CROSSHAIR_DETECTION_KWARGS,
    velocity_vector: bool = True,
    velocity_vector_kwargs: dict = VELOCITY_VECTOR_KWARGS,
    cmap: matplotlib.cm = CMAP_BONE,
) -> matplotlib.axes.Axes:
    """
    Plot a single cutout on the given axes.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Matplotlib axes (usually a subplot) on which to add cutout.
    image : `~numpy.ndarray` (N, M)
        Image data.
    ra : float
        Predicted RA in degrees.
    dec : float
        Predicted Dec in degrees.
    vra : float
        Predicted RA-velocity in degrees per day.
    vdec : float
        Predicted Dec-velocity in degrees per day.
        Predicted Dec in degrees.
    obj_ra : float
        RA of object detection in degrees.
    obj_dec : float
        Dec of object detection in degrees.
    dt : float
        Exposure duration in units of seconds. Used to scale the velocity vector.
    circe_object : bool, optional
        Add circle around detected object.
    circle_object_kwargs : dict
        Keyword arguments to pass to `~cutouts.plot.add_circle`.
    crosshair : bool, optional
        Add crosshair centered on (RA, Dec).
    crosshair_kwargs : dict
        Keyword arguments to pass to `~cutouts.plot.add_crosshair`.
    velocity_vector : bool, optional
        Add velocity vector showing predicted motion.
    velocity_vector_kwargs : dict
        Keyword arguments to pass to `~cutouts.plot.add_velocity_vector`.
    cmap : `~matplotlib.cm`
        Colormap for the cutout.
    """
    ax.imshow(
        image,
        origin="lower",
        cmap=cmap,
        norm=ImageNormalize(image, interval=ZScaleInterval()),
    )

    if crosshair:
        add_crosshair(ax, ra, dec, **crosshair_kwargs)
    if velocity_vector:
        add_velocity_vector(
            ax,
            ra,
            dec,
            vra,
            vdec,
            dt,
            **velocity_vector_kwargs,
        )
    if circle_object:
        add_circle(ax, obj_ra, obj_dec, **circle_object_kwargs)

    ax.axis("off")
    return ax


def plot_cutouts(
    candidates: pd.DataFrame,
    dpi: int = 200,
    max_cols: int = 4,
    row_height: float = 2.0,
    col_width: float = 2.0,
    cutout_height_arcsec: float = 20,
    cutout_width_arcsec: float = 20,
    include_missing: bool = True,
    circle_object: bool = True,
    circle_object_kwargs: dict = CIRCLE_OBJECT_KWARGS,
    crosshair: bool = True,
    crosshair_detection_kwargs: dict = CROSSHAIR_DETECTION_KWARGS,
    crosshair_non_detection_kwargs: dict = CROSSHAIR_NON_DETECTION_KWARGS,
    velocity_vector: bool = True,
    velocity_vector_kwargs: dict = VELOCITY_VECTOR_KWARGS,
    subplots_adjust_kwargs: dict = SUBPLOTS_ADJUST_KWARGS,
    cmap=CMAP_BONE,
) -> Tuple[matplotlib.figure.Figure, List[matplotlib.axes.Axes]]:
    """
    Plot cutouts on a grid.

    Parameters
    ----------
    candidates : `~pandas.DataFrame`
        DataFrame containing the candidates to plot.
    dpi : int, optional
        DPI of the plot.
    max_cols : int, optional
        Maximum number of columns the grid should have.
    row_height : float, optional
        Height in inches each row should have.
    col_width : float, optional
        Width in inches each column should have.
    cutout_height_arcsec : float, optional
        Desired height of the cutout in arcseconds.
    cutout_width_arcsec : float, optional
        Desired width of the cutout in arcseconds.
    include_missing : bool, optional
        Include an empty placeholder cutout if the cutout was not found (their paths are None).
    circe_object : bool, optional
        Add circle around detected object.
    circle_object_kwargs : dict
        Keyword arguments to pass to `~cutouts.plot.add_circle`.
    crosshair : bool, optional
        Add crosshairs centered on (RA, Dec). If the source is detected (see the magnitude
        keyword argument), then the crosshair_detection_kwargs will be applied to the crosshair.
        If the source is not detected (a NaN value or mag is None) then the crosshair_non_detection_kwargs
        will be applied to the crosshair.
    crosshair_detection_kwargs : dict
        Keyword arguments to pass to `~cutouts.plot.add_crosshair` for detected sources.
    crosshair_non_detection_kwargs : dict
        Keyword arguments to pass to `~cutouts.plot.add_crosshair` for undetected sources.
    velocity_vector : bool, optional
        Add velocity vector showing predicted motion.
    velocity_vector_kwargs : dict
        Keyword arguments to pass to `~cutouts.plot.add_velocity_vector`.
    height : int, optional
        Desired height of the cutout in pixels.
    width : int, optional
        Desired width of the cutout in pixels.
    subplots_adjust_kwargs : dict, optional
        Keyword arguments to pass to `fig.subplots_adjust`.
    cmap : `~matplotlib.cm`
        Colormap for the cutout.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        Matplotlib figure.
    ax : list of `~matplotlib.axes.Axes`
        Matplotlib axes.
    """
    paths = candidates["path"].values
    times = candidates["exposure_start"].values
    ra = candidates["ra"].values
    dec = candidates["dec"].values
    vra = candidates["vra"].values
    vdec = candidates["vdec"].values
    obj_ra = candidates["obj_ra"].values
    obj_dec = candidates["obj_dec"].values
    filters = candidates["filter"].values
    mag = candidates["mag"].values
    mag_sigma = candidates["mag_sigma"].values
    exposure_time = candidates["exposure_duration"].values
    obscode = candidates["obscode"].values

    if include_missing:
        num_obs = len(paths)
    else:
        num_obs = 0
        for paths_i in paths:
            if paths_i is not None:
                num_obs += 1

    # If the number of observations is less than the maximum number of columns,
    # then we can just use the number of observations as the number of columns.
    if num_obs < max_cols:
        num_cols = num_obs
    else:
        num_cols = max_cols

    num_rows = np.ceil(num_obs / num_cols).astype(int)

    fig = plt.figure(figsize=(col_width * num_cols, row_height * num_rows), dpi=dpi)
    fig.subplots_adjust(**subplots_adjust_kwargs)

    axs = []
    j = 0
    for i, (
        path_i,
        ra_i,
        dec_i,
        vra_i,
        vdec_i,
        obj_ra_i,
        obj_dec_i,
        dt_i,
        obscode_i,
    ) in enumerate(
        zip(paths, ra, dec, vra, vdec, obj_ra, obj_dec, exposure_time, obscode)
    ):
        ax = None
        y = 1.0
        cutout_title = ""

        if np.isnan(times[i]):
            cutout_title += f"[{obscode_i}]"
        else:
            time_i = Time(times[i], format="mjd", scale="utc")
            cutout_title += f"{time_i.isot} [{obscode_i}]"

        cutout_title += f"\nRA: {ra_i:.4f}, Dec: {dec_i:.4f}"

        if filters[i] is not None:
            cutout_title += f", {filters[i]}"

        if np.isnan(mag[i]):
            crosshair_kwargs_i = crosshair_non_detection_kwargs
            cutout_title += ": --.--"
        else:
            crosshair_kwargs_i = crosshair_detection_kwargs
            cutout_title += f": {mag[i]:.2f}"
        y -= 0.03

        if np.isnan(mag_sigma[i]):
            cutout_title += "$\pm$--.--"  # noqa: W605
        else:
            cutout_title += f"$\pm{mag_sigma[i]:.2f}$"  # noqa: W605

        if np.isnan(exposure_time[i]):
            cutout_title += ", $\Delta$t: ---s"  # noqa: W605
        else:
            cutout_title += f", $\Delta$t: {exposure_time[i]:.0f}s"  # noqa: W605

        # We expect vra_i, vdec_i to be NaN if we are plotting a comparison candidate
        if np.isnan(vra_i) or np.isnan(vdec_i):
            velocity_vector_i = False
        else:
            velocity_vector_i = velocity_vector

        if path_i is None:
            if include_missing:
                ax = fig.add_subplot(num_rows, num_cols, j + 1)

                # TODO - This currently will result in poorly formatted cutout output when the
                # cutouts requested are rectangular, as any that are missing will not preserve
                # the requested aspect ratio
                image = np.zeros((100, 100), dtype=float)
                ax.imshow(image, origin="lower", cmap=cmap)
                ax.axis("off")
                ax.text(
                    100 / 2,
                    100 / 2,
                    "No image found",
                    horizontalalignment="center",
                    color="w",
                )
                j += 1

        else:

            # Read WCS and add plot projection here since a subplot's
            # projection cannot be altered once created...
            hdu = fits.open(path_i)[0]
            hdr = hdu.header
            wcs = WCS(hdr)

            # Center the cutout on the candidate's position and update
            # the wcs to match
            # Create SkyCoordinate for the center
            center = SkyCoord(ra_i, dec_i, unit="deg", frame="icrs")
            image_centered = Cutout2D(
                hdu.data,
                center,
                (cutout_height_arcsec * u.arcsec, cutout_width_arcsec * u.arcsec),
                wcs=wcs,
                mode="partial",
                fill_value=np.nan,
            )
            wcs = image_centered.wcs

            # Set the projection for this subplot
            ax = fig.add_subplot(num_rows, num_cols, j + 1, projection=wcs)

            # Plot the image
            ax = plot_cutout(
                ax,
                image_centered.data,
                ra_i,
                dec_i,
                vra_i,
                vdec_i,
                obj_ra_i,
                obj_dec_i,
                dt_i,
                circle_object=circle_object,
                circle_object_kwargs=circle_object_kwargs,
                crosshair=crosshair,
                crosshair_kwargs=crosshair_kwargs_i,
                velocity_vector=velocity_vector_i,
                velocity_vector_kwargs=velocity_vector_kwargs,
            )
            j += 1

        if ax is not None:
            ax.set_title(cutout_title, fontsize=5, y=y)
            axs.append(ax)

    return fig, axs


def plot_comparison_cutouts(
    candidates: pd.DataFrame,
    comparison_candidates: pd.DataFrame,
    dpi: int = 200,
    max_cols: int = 4,
    row_height: float = 2.0,
    col_width: float = 2.0,
    cutout_height_arcsec: float = 20,
    cutout_width_arcsec: float = 20,
    include_missing: bool = True,
    circle_object: bool = True,
    circle_object_kwargs: dict = CIRCLE_OBJECT_KWARGS,
    crosshair: bool = True,
    crosshair_detection_kwargs: dict = CROSSHAIR_DETECTION_KWARGS,
    crosshair_non_detection_kwargs: dict = CROSSHAIR_NON_DETECTION_KWARGS,
    velocity_vector: bool = True,
    velocity_vector_kwargs: dict = VELOCITY_VECTOR_KWARGS,
    subplots_adjust_kwargs: dict = SUBPLOTS_ADJUST_KWARGS,
    cmap=CMAP_BONE,
) -> Tuple[List[matplotlib.figure.Figure], List[List[matplotlib.axes.Axes]]]:
    """
    Plot comparison cutouts. First, all comparison cutouts are plotted, then each
    actual observation candidate is added one by one.

    Parameters
    ----------
    candidates : pd.DataFrame
        DataFrame containing the candidate observations.
    comparison_candidates : pd.DataFrame
        DataFrame containing the comparison observations.
    dpi : int, optional
        DPI of the plot.
    max_cols : int, optional
        Maximum number of columns the grid should have.
    row_height : float, optional
        Height in inches each row should have.
    col_width : float, optional
        Width in inches each column should have.
    cutout_height_arcsec : float, optional
        Desired height of the cutout in arcseconds.
    cutout_width_arcsec : float, optional
        Desired width of the cutout in arcseconds.
    include_missing : bool, optional
        Include an empty placeholder cutout if the cutout was not found (their paths are None).
    circe_object : bool, optional
        Add circle around detected object.
    circle_object_kwargs : dict
        Keyword arguments to pass to `~cutouts.plot.add_circle`.
    crosshair : bool, optional
        Add crosshairs centered on (RA, Dec). If the source is detected (see the magnitude
        keyword argument), then the crosshair_detection_kwargs will be applied to the crosshair.
        If the source is not detected (a NaN value or mag is None) then the crosshair_non_detection_kwargs
        will be applied to the crosshair.
    crosshair_detection_kwargs : dict
        Keyword arguments to pass to `~cutouts.plot.add_crosshair` for detected sources.
    crosshair_non_detection_kwargs : dict
        Keyword arguments to pass to `~cutouts.plot.add_crosshair` for undetected sources.
    velocity_vector : bool, optional
        Add velocity vector showing predicted motion.
    velocity_vector_kwargs : dict
        Keyword arguments to pass to `~cutouts.plot.add_velocity_vector`.
    height : int, optional
        Desired height of the cutout in pixels.
    width : int, optional
        Desired width of the cutout in pixels.
    subplots_adjust_kwargs : dict, optional
        Keyword arguments to pass to `fig.subplots_adjust`.
    cmap : `~matplotlib.cm`
        Colormap for the cutout.

    Returns
    -------
    figs : `~matplotlib.figure.Figure`
        Matplotlib figure.
    axs : list of `~matplotlib.axes.Axes`
        Matplotlib axes.
    """
    figs = []
    axs = []

    for i in range(len(candidates) + 1):
        candidates_i = pd.concat(
            [candidates.iloc[:i], comparison_candidates.iloc[i:]], ignore_index=True
        )

        fig, ax = plot_cutouts(
            candidates_i,
            dpi=dpi,
            max_cols=max_cols,
            row_height=row_height,
            col_width=col_width,
            cutout_height_arcsec=cutout_height_arcsec,
            cutout_width_arcsec=cutout_width_arcsec,
            include_missing=include_missing,
            circle_object=circle_object,
            circle_object_kwargs=circle_object_kwargs,
            crosshair=crosshair,
            crosshair_detection_kwargs=crosshair_detection_kwargs,
            crosshair_non_detection_kwargs=crosshair_non_detection_kwargs,
            velocity_vector=velocity_vector,
            velocity_vector_kwargs=velocity_vector_kwargs,
            subplots_adjust_kwargs=subplots_adjust_kwargs,
            cmap=cmap,
        )

        for j, a in enumerate(ax[:i]):

            if not np.isnan(candidates["exposure_start"].values[j]) and not np.isnan(
                comparison_candidates["exposure_start"].values[j]
            ):
                delta_time = (
                    candidates["exposure_start"].values[j]
                    - comparison_candidates["exposure_start"].values[j]
                )
                xlim = a.get_xlim()
                xrange = xlim[1] - xlim[0]
                xlim_min = xlim[0] + 0.05 * xrange

                ylim = a.get_ylim()
                yrange = ylim[1] - ylim[0]
                ylim_max = ylim[1] - 0.1 * yrange

                a.text(
                    xlim_min, ylim_max, f"{delta_time:+.5f} d", c="#03fc0f", fontsize=10
                )

        figs.append(fig)
        axs.append(ax)

    return figs, axs


def generate_gif(
    figs: List[matplotlib.figure.Figure],
    out_dir: pathlib.Path = pathlib.Path("."),
    out_file: pathlib.Path = pathlib.Path("cutout.gif"),
    dpi: int = 200,
    cleanup: bool = True,
):
    """
    Generate a GIF from a list of matplotlib figures.

    Parameters
    ----------
    figs : list of `~matplotlib.figure.Figure`
        Matplotlib figures.
    out_dir : `~pathlib.Path`
        Output directory.
    out_file : `~pathlib.Path`
        Output file.
    dpi : int, optional
        DPI of the output GIF.
    cleanup : bool, optional
        Delete the individual PNG files after generating the GIF.
    """
    files = []
    for i, fig in enumerate(figs):
        file_path = out_dir.joinpath(f"comparison_{i:03d}.png")
        fig.savefig(file_path, dpi=dpi)
        files.append(file_path)

    images = []
    for file in files:
        images.append(iio.imread(file))

    iio.imwrite(
        out_dir.joinpath(out_file),
        images,
        duration=1000,  # time per frame in ms
        loop=0,  # loop forever
    )
    if cleanup:
        for file in files:
            file.unlink()

    return
