import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Tuple
from astropy.time import Time
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import ImageNormalize
from astropy.visualization import ZScaleInterval

CMAP_BONE = matplotlib.cm.bone.copy()
CMAP_BONE.set_bad("black")


def add_crosshair(
    ax: matplotlib.axes.Axes,
    wcs: WCS,
    ra: float,
    dec: float,
    gap: float = 2,
    length: int = 2,
    x_offset: int = 0,
    y_offset: int = 0,
    **kwargs,
):
    """
    Add a crosshair centered on RA and Dec to the given axes.
    Parameter
    ---------
    ax : `~matplotlib.axes.Axes`
        Matplotlib axes (usually a subplot) on which to add the crosshair.
    wcs : `~astropy.wcs.wcs.WCS`
        World Coordinate System (WCS) that maps pixels in an image to RA, Dec.
    ra : float
        Predicted RA in degrees.
    dec : float
        Predicted Dec in degrees.
    gap : float
        Distance from center in arcseconds to start drawing crosshair reticle bar.
    length : float
        Length in arcseconds of an individual bar reticle.
    x_offset : int, optional
        Offset in x-axis pixels from the sky-plane origin of the image (offsets might be non-zero
        due to image centering, padding, and/or trimming).
    y_offset : int, optional
        Offset in y-axis pixels from the sky-plane origin of the image (offsets might be non-zero
        due to image centering, padding, and/or trimming).
    **kwargs
        Keyword arguments to pass to ax.hlines and ax.vlines.
    """
    # Get pixel location of RA and Dec
    y_center, x_center = wcs.world_to_array_index_values(ra, dec)

    x_center = x_center + x_offset
    y_center = y_center + y_offset

    width_pixel_scale, height_pixel_scale = proj_plane_pixel_scales(wcs)

    gap_scaled = (gap / 3600.0) / width_pixel_scale
    length_scaled = (length / 3600.0) / width_pixel_scale

    ax.vlines(
        x_center, y_center + gap_scaled, y_center + gap_scaled + length_scaled, **kwargs
    )
    ax.vlines(
        x_center, y_center - gap_scaled, y_center - gap_scaled - length_scaled, **kwargs
    )
    ax.hlines(
        y_center, x_center + gap_scaled, x_center + gap_scaled + length_scaled, **kwargs
    )
    ax.hlines(
        y_center, x_center - gap_scaled, x_center - gap_scaled - length_scaled, **kwargs
    )
    return


def add_velocity_vector(
    ax: matplotlib.axes.Axes,
    wcs: WCS,
    ra: float,
    dec: float,
    vra: float,
    vdec: float,
    gap: float = 8,
    length: float = 8,
    width: float = 1,
    x_offset: int = 0,
    y_offset: int = 0,
    **kwargs,
):
    """
    Add a velocity vector showing the predicted velocity of an object
    to an image.
    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Matplotlib axes (usually a subplot) on which to add the velocity vector.
    wcs : `~astropy.wcs.wcs.WCS`
        World Coordinate System (WCS) that maps pixels in an image to RA, Dec.
    ra : float
        Predicted RA in degrees.
    dec : float
        Predicted Dec in degrees.
    vra : float
        Predicted RA-velocity in degrees per day.
    vdec : float
        Predicted Dec in degrees in degrees per day.
    gap : float
        Distance from center in percentage of img width to start drawing velocity vector.
    length : float
        Length in percentage of img width of velocity vector.
    width : float
        width in percentage of img width of velocity vector.
    x_offset : int, optional
        Offset in x-axis pixels from the sky-plane origin of the image (offsets might be non-zero
        due to image centering and padding, and/or trimming).
    y_offset : int, optional
        Offset in y-axis pixels from the sky-plane origin of the image (offsets might be non-zero
        due to image centering, padding, and/or trimming)
    **kwargs
        Keyword arguments to pass to ax.arrow.
    """
    # x_center, y_center = wcs.world_to_array_index_values(ra, dec)
    x_center, y_center = wcs.world_to_pixel_values(ra, dec)
    x_center = x_center + x_offset
    y_center = y_center + y_offset

    image_width_pixels, image_height_pixels = wcs.array_shape

    length_scaled = length * image_width_pixels
    gap_scaled = gap * image_width_pixels
    width_scaled = width * image_width_pixels

    dt = 1 / 24 / 2
    x_propagated, y_propagated = wcs.world_to_pixel_values(
        ra + vra * dt, dec + vdec * dt
    )
    x_propagated = x_propagated + x_offset
    y_propagated = y_propagated + y_offset
    vx = (x_propagated - x_center) / dt
    vy = (y_propagated - y_center) / dt

    vx_hat = vx / np.sqrt(vx**2 + vy**2)
    vy_hat = vy / np.sqrt(vx**2 + vy**2)
    # print(3 * width_scaled)
    ax.arrow(
        x_center + gap_scaled * vx_hat,
        y_center + gap_scaled * vy_hat,
        length_scaled * vx_hat,
        length_scaled * vy_hat,
        width=width_scaled,
        head_width=7 * width_scaled,
        # length_includes_head=True,
        **kwargs,
    )
    return


def center_image(
    image: npt.NDArray[np.float64],
    wcs: WCS,
    ra: float,
    dec: float,
    height: int = 115,
    width: int = 115,
) -> npt.NDArray[np.float64]:
    """
    Given an image and its WCS, ensure that (RA, Dec) is actually as near
    to the center of the image as possible. Also, ensure that the image
    is sized as (height, width). If the image is not centered, this function
    will center (RA, Dec) as much as possible, and if the image is not the desired
    shape then this function will pad columns/rows or trim columns/rows until it is the
    desired shape.
    Parameters
    ----------
    image : `~numpy.ndarray` (N, M)
        2D array with image data.
    wcs : `~astropy.wcs.wcs.WCS`
        World Coordinate System (WCS) that maps pixels in an image to RA, Dec.
    ra : float
        RA in degrees of the desired center of the image.
    dec : float
        Dec in degrees of the desired center of the image.
    height : int, optional
        The desired height of the image.
    width : int, optional
        The desired width of the image.
    Returns
    -------
    image_centered : `~numpy.ndarray` (height, width)
        Image with desired height and width with RA, Dec as near to the center as possible.
    x_offset : int
        Offset in x-axis pixels from the sky-plane origin of the image (offsets might be non-zero
        due to image centering and padding, and/or trimming).
    y_offset : int
        Offset in y-axis pixels from the sky-plane origin of the image (offsets might be non-zero
        due to image centering, padding, and/or trimming)
    """
    # Calculate where RA and Dec fall in the actual image
    image_x_center, image_y_center = wcs.world_to_array_index_values(ra, dec)

    # The following is not a typo: note x,y change
    pimage_y_center, pimage_x_center = wcs.world_to_pixel_values(ra, dec)

    image_copy = image.copy()
    num_rows, num_cols = image_copy.shape

    cols_from_left = num_cols - 2 * image_x_center
    rows_from_top = num_rows - 2 * image_y_center
    x_offset, y_offset = 0, 0

    if cols_from_left > 0:
        pad_cols = np.abs(np.ceil(cols_from_left)).astype(int)
        left_padding = np.zeros((image_copy.shape[0], pad_cols))
        image_copy = np.hstack([left_padding, image_copy])
        x_offset += pad_cols

    elif cols_from_left < 0:
        pad_cols = np.abs(np.ceil(-cols_from_left)).astype(int)
        right_padding = np.zeros((image_copy.shape[0], pad_cols))
        image_copy = np.hstack([image_copy, right_padding])

    if rows_from_top > 0:
        pad_rows = np.abs(np.ceil(rows_from_top)).astype(int)
        top_padding = np.zeros((pad_rows, image_copy.shape[1]))
        image_copy = np.vstack([top_padding, image_copy])
        y_offset += pad_rows

    elif rows_from_top < 0:
        pad_rows = np.abs(np.ceil(-rows_from_top)).astype(int)
        bottom_padding = np.zeros((pad_rows, image_copy.shape[1]))
        image_copy = np.vstack([image_copy, bottom_padding])

    # Update shape parameters
    num_rows, num_cols = image_copy.shape

    # Disable xbit, ybit for the time being
    xbit = 0
    if image_x_center - pimage_x_center > 0:
        xbit = 0

    ybit = 0
    if image_y_center - pimage_y_center > 0:
        ybit = 0

    # If the image is not the desired width, pad more columns
    # until it is
    if num_cols < width:
        num_cols = width - num_cols
        for i in range(num_cols):
            padding = np.zeros((image_copy.shape[0], 1))
            if i % 2 == 0 + xbit:
                image_copy = np.hstack([padding, image_copy])
                x_offset += 1
            else:
                image_copy = np.hstack([image_copy, padding])

    # Update shape parameters
    num_rows, num_cols = image_copy.shape

    # If the image is larger than the desired width, remove
    # columns until it is
    if num_cols > width:
        num_cols = num_cols - width
        for i in range(num_cols):
            if i % 2 == 0 + xbit:
                image_copy = image_copy[:, 1:]
                x_offset -= 1
            else:
                image_copy = image_copy[:, :-1]

    # Update shape parameters
    num_rows, num_cols = image_copy.shape

    # If the image is not the desired height, pad more rows
    # until it is
    if num_rows < height:
        num_rows = height - num_rows
        for i in range(num_rows):
            padding = np.zeros((1, image_copy.shape[1]))
            if i % 2 == 0 + ybit:
                image_copy = np.vstack([padding, image_copy])
                y_offset += 1
            else:
                image_copy = np.vstack([image_copy, padding])

    # Update shape parameters
    num_rows, num_cols = image_copy.shape

    # If the image is larger than the desired height, remove
    # rows until it is
    if num_rows > height:
        num_rows = num_rows - height
        for i in range(num_rows):
            if i % 2 == 0 + ybit:
                image_copy = image_copy[1:, :]
                y_offset -= 1
            else:
                image_copy = image_copy[
                    :-1,
                ]

    return image_copy, x_offset, y_offset


def plot_cutout(
    ax: matplotlib.axes.Axes,
    path: str,
    ra: float,
    dec: float,
    vra: float,
    vdec: float,
    height_arcsec: float = 20,
    width_arcsec: float = 20,
    crosshair: bool = True,
    crosshair_kwargs: dict = {
        "gap": 2,
        "length": 2,
        "color": "r",
        "alpha": 0.9,
        "zorder": 9,
    },
    velocity_vector: bool = True,
    velocity_vector_kwargs: dict = {
        "gap": 2,
        "length": 2,
        "color": "#34ebcd",
        "width": 0.2,
        "zorder": 10,
    },
    cmap: matplotlib.cm = CMAP_BONE,
) -> matplotlib.axes.Axes:
    """
    Plot a single cutout on the given axes.
    Note that when height_arcsec and width_arcsec are converted to pixel sizes, these converted
    values are rounded up to an integer number of pixels.
    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Matplotlib axes (usually a subplot) on which to add cutout.
    path : str
        Location of cutout file.
    ra : float
        Predicted RA in degrees.
    dec : float
        Predicted Dec in degrees.
    vra : float
        Predicted RA-velocity in degrees per day.
    vdec : float
        Predicted Dec in degrees in degrees per day.
    height_arcsec : float
        Image height in arcseconds
    width_arcsec : float
        Image width in arcseconds
    crosshair : bool, optional
        Add crosshair centered on (RA, Dec).
    crosshair_kwargs : dict
        Keyword arguments to pass to `~cutouts.plot.add_crosshair`.
    velocity_vector : bool, optional
        Add velocity vector showing predicted motion.
    velocity_vector_kwargs : dict
        Keyword arguments to pass to `~cutouts.plot.add_velocity_vector`.
    height : int, optional
        Desired height of the cutout in pixels.
    width : int, optional
        Desired width of the cutout in pixels.
    cmap : `~matplotlib.cm`
        Colormap for the cutout.
    """
    # Read file and get image
    hdu = fits.open(path)[0]
    image = hdu.data
    hdr = hdu.header
    wcs = WCS(hdr)
    width_pixel_scale, height_pixel_scale = proj_plane_pixel_scales(wcs)
    # TODO - double check image orientation
    height_pix = np.ceil(height_arcsec / height_pixel_scale / 3600.0).astype(int)
    width_pix = np.ceil(width_arcsec / width_pixel_scale / 3600.0).astype(int)
    image_centered, x_offset, y_offset = center_image(
        image, wcs, ra, dec, height=height_pix, width=width_pix
    )
    ax.imshow(
        image_centered,
        origin="lower",
        cmap=cmap,
        norm=ImageNormalize(image, interval=ZScaleInterval()),
    )
    ax.axis("off")

    if crosshair:
        add_crosshair(
            ax, wcs, ra, dec, x_offset=x_offset, y_offset=y_offset, **crosshair_kwargs
        )
    if velocity_vector:
        add_velocity_vector(
            ax,
            wcs,
            ra,
            dec,
            vra,
            vdec,
            x_offset=x_offset,
            y_offset=y_offset,
            **velocity_vector_kwargs,
        )

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
    crosshair: bool = True,
    crosshair_detection_kwargs: dict = {
        "gap": 2,
        "length": 2,
        "color": "#03fc0f",
        "alpha": 1.0,
        "zorder": 9,
    },
    crosshair_non_detection_kwargs: dict = {
        "gap": 2,
        "length": 2,
        "color": "r",
        "alpha": 1.0,
        "zorder": 9,
    },
    velocity_vector: bool = True,
    velocity_vector_kwargs: dict = {
        "gap": 0.05,
        "length": 0.15,
        "color": "#34ebcd",
        "width": 0.005,
        "zorder": 10,
    },
    subplots_adjust_kwargs: dict = {
        "hspace": 0.15,
        "wspace": 0.15,
        "left": 0.05,
        "right": 0.95,
        "top": 0.95,
        "bottom": 0.02,
    },
    cmap=CMAP_BONE,
) -> Tuple[matplotlib.figure.Figure, List[matplotlib.axes.Axes]]:
    """
    Plot cutouts on a grid.
    Parameters
    ----------
    paths : List[str]
        Location of cutout file.
    ra : `~numpy.ndarray` (N)
        Predicted RA in degrees.
    dec : `~numpy.ndarray` (N)
        Predicted Dec in degrees.
    vra : `~numpy.ndarray` (N)
        Predicted RA-velocity in degrees per day.
    vdec : `~numpy.ndarray` (N)
        Predicted Dec in degrees in degrees per day.
    filters : `~numpy.ndarray` (N)
        Filters in which the observations were made.
    mag : `~numpy.ndarray` (N)
        Magnitude of the observation if detected. NaN magnitudes are interpreted
        as undetected.
    mag_sigma: `~numpy.ndarray` (N)
        Magnitude error of the detected observation.
    exposure_time : `~numpy.ndarray` (N)
        Exposure time in seconds.
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
    paths = candidates["path"]
    times = candidates["exposure_start"].map(lambda x: Time(x, scale="utc", format="mjd"))
    ra = candidates["ra"]
    dec = candidates["dec"]
    vra = candidates["vra"]
    vdec = candidates["vdec"]
    filters = candidates["filter"]
    mag = candidates["mag"]
    mag_sigma = candidates["mag_sigma"]
    exposure_time = candidates["exposure_duration"]

    if include_missing:
        num_obs = len(paths)
    else:
        num_obs = 0
        for paths_i in paths:
            if paths_i is not None:
                num_obs += 1
    num_rows = np.ceil(num_obs / max_cols).astype(int)

    include_filters = False
    include_mag = False
    include_mag_sigma = False
    include_exposure_time = False
    if isinstance(filters, np.ndarray):
        include_filters = True
    if isinstance(mag, np.ndarray):
        include_mag = True
    if isinstance(mag_sigma, np.ndarray):
        include_mag_sigma = True
    if isinstance(exposure_time, np.ndarray):
        include_exposure_time = True

    fig = plt.figure(figsize=(col_width * max_cols, row_height * num_rows), dpi=dpi)
    fig.subplots_adjust(**subplots_adjust_kwargs)

    axs = []
    j = 0
    for i, (path_i, ra_i, dec_i, vra_i, vdec_i) in enumerate(
        zip(paths, ra, dec, vra, vdec)
    ):

        ax = None
        y = 1.0
        title = ""
        title += f"{times[i].iso}"
        if include_filters:
            title += f"\n{filters[i]}"

        if include_mag:
            if np.isnan(mag[i]):
                crosshair_kwargs_i = crosshair_non_detection_kwargs
                title += f": --.--"
            else:
                crosshair_kwargs_i = crosshair_detection_kwargs
                title += f": {mag[i]:.2f}"
            y -= 0.03

        else:
            crosshair_kwargs_i = crosshair_detection_kwargs

        if include_mag_sigma:
            if np.isnan(mag_sigma[i]):
                title += f":$\pm$--.--"
            else:
                title += f"$\pm{mag_sigma[i]:.2f}$"

        if include_exposure_time:
            if np.isnan(exposure_time[i]):
                title += f", $\Delta$t: ---s"
            else:
                title += f", $\Delta$t: {exposure_time[i]:.0f}s"

        if crosshair:
            crosshair_size = (
                crosshair_kwargs_i["length"] * 2.0 + crosshair_kwargs_i["gap"]
            )
            title += f', Xhair width: {crosshair_size}"'

        if path_i is None:

            if include_missing:

                ax = fig.add_subplot(num_rows, max_cols, j + 1)

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
            ax = fig.add_subplot(num_rows, max_cols, j + 1)
            ax = plot_cutout(
                ax,
                path_i,
                ra_i,
                dec_i,
                vra_i,
                vdec_i,
                height_arcsec=cutout_height_arcsec,
                width_arcsec=cutout_width_arcsec,
                crosshair=crosshair,
                crosshair_kwargs=crosshair_kwargs_i,
                velocity_vector=velocity_vector,
                velocity_vector_kwargs=velocity_vector_kwargs,
            )
            j += 1

        if ax is not None:
            ax.set_title(title, fontsize=6, y=y)
            axs.append(ax)

    return fig, axs
