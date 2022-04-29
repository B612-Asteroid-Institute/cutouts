import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize
from astropy.visualization import ZScaleInterval

CMAP_BONE = matplotlib.cm.bone.copy()
CMAP_BONE.set_bad("black")

def add_crosshair(ax, wcs, ra, dec, gap=8, length=8, x_offset=0, y_offset=0, **kwargs):

    # Get pixel location of RA and Dec
    x_center, y_center = wcs.world_to_array_index_values(ra, dec)
    #y_center, x_center = wcs.world_to_pixel_values(ra, dec)
    x_center = x_center + x_offset
    y_center = y_center + y_offset

    ax.vlines(
        x_center,
        y_center + gap,
        y_center + gap + length,
        **kwargs
    )
    ax.vlines(
        x_center,
        y_center - gap,
        y_center - gap - length,
        **kwargs
    )
    ax.hlines(
        y_center,
        x_center + gap,
        x_center + gap + length,
        **kwargs
    )
    ax.hlines(
        y_center,
        x_center - gap,
        x_center - gap - length,
        **kwargs
    )
    return

def add_velocity_vector(ax, wcs, ra, dec, vra, vdec, gap=8, length=8, x_offset=0, y_offset=0, **kwargs):

    x_center, y_center = wcs.world_to_array_index_values(ra, dec)
    #y_center, x_center = wcs.world_to_pixel_values(ra, dec)
    x_center = x_center + x_offset
    y_center = y_center + y_offset

    dt = 1/24/2
    xoffset, yoffset = wcs.world_to_array_index_values(ra + vra * dt, dec + vdec * dt)
    #yoffset, xoffset = wcs.world_to_pixel_values(ra + vra * dt, dec + vdec * dt)
    xoffset = xoffset + x_offset
    yoffset = yoffset + y_offset
    vx = (xoffset - x_center) / dt
    vy = (yoffset - y_center) / dt

    vx_hat = vx / np.sqrt(vx**2 + vy**2)
    vy_hat = vy / np.sqrt(vx**2 + vy**2)

    ax.arrow(
        x_center + gap*vx_hat,
        y_center + gap*vy_hat,
        length*vx_hat,
        length*vy_hat,
        length_includes_head=True,
        **kwargs
    )
    return

def center_image(image, wcs, ra, dec, height=115, width=115):

    # Calculate where RA and Dec fall in the actual image
    image_x_center, image_y_center = wcs.world_to_array_index_values(ra, dec)

    # The following is not a typo: note x,y change
    pimage_y_center, pimage_x_center = wcs.world_to_pixel_values(ra, dec)

    image_copy = image.copy()
    num_rows, num_cols = image_copy.shape

    cols_from_left = num_cols - 2*image_x_center
    rows_from_top = num_rows - 2*image_y_center
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
                image_copy = image_copy[:-1, ]

    return image_copy, x_offset, y_offset

def plot_cutout(
        ax,
        path,
        ra,
        dec,
        vra,
        vdec,
        crosshair=True,
        crosshair_kwargs={
            "color": "r",
            "alpha": 0.9,
            "zorder": 9
        },
        velocity_vector=True,
        velocity_vector_kwargs={
            "color": "#34ebcd",
            "width": 0.2,
            "head_width": 2,
            "zorder": 10
        },
        height=115,
        width=115,
        cmap=CMAP_BONE
    ):

    # Read file and get image
    hdu = fits.open(path)[0]
    image = hdu.data
    hdr = hdu.header
    wcs = WCS(hdr)

    image_centered, x_offset, y_offset = center_image(image, wcs, ra, dec, height=height, width=width)

    ax.imshow(
        image_centered,
        origin="lower",
        cmap=cmap,
        norm=ImageNormalize(
            image,
            interval=ZScaleInterval()
        )
    )
    ax.axis("off")

    if crosshair:
        add_crosshair(
            ax,
            wcs,
            ra,
            dec,
            x_offset=x_offset,
            y_offset=y_offset,
            **crosshair_kwargs
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
            **velocity_vector_kwargs
        )

    return ax

def plot_cutouts(
        paths,
        times,
        ra,
        dec,
        vra,
        vdec,
        filter=None,
        mag=None,
        mag_sigma=None,
        dpi=200,
        max_cols=4,
        row_height=2,
        col_width=2,
        cutout_height=75,
        cutout_width=75,
        crosshair=True,
        crosshair_detection_kwargs={
            "color": "#03fc0f",
            "alpha": 1.0,
            "zorder": 9
        },
        crosshair_non_detection_kwargs={
            "color": "r",
            "alpha": 1.0,
            "zorder": 9
        },
        velocity_vector=True,
        velocity_vector_kwargs={
            "color": "#34ebcd",
            "width": 0.2,
            "head_width": 2,
            "zorder": 10
        },
        subplots_adjust_kwargs={
            "hspace": 0.15,
            "wspace": 0.15,
            "left": 0.05,
            "right": 0.95,
            "top": 0.95,
            "bottom": 0.02
        },
        cmap=CMAP_BONE,
    ):

    num_obs = len(paths)
    num_rows = np.ceil(num_obs / max_cols).astype(int)

    include_filter = False
    include_mag = False
    include_mag_sigma = False
    if isinstance(filter, np.ndarray):
        include_filter = True
    if isinstance(mag, np.ndarray):
        include_mag = True
    if isinstance(mag_sigma, np.ndarray):
        include_mag_sigma = True

    fig = plt.figure(figsize=(col_width*max_cols, row_height*num_rows), dpi=dpi)
    fig.subplots_adjust(**subplots_adjust_kwargs)

    axs = []
    for i, (path_i, ra_i, dec_i, vra_i, vdec_i) in enumerate(zip(paths, ra, dec, vra, vdec)):

        y = 1.0
        title = ""
        title += f"{times[i].iso}"
        if include_filter:
            title += f"\n{filter[i]}"

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

        ax = fig.add_subplot(num_rows, max_cols, i+1)

        if path_i is None:
            image = np.zeros((cutout_height, cutout_width), dtype=float)
            ax.imshow(
                image,
                origin="lower",
                cmap=cmap
            )
            ax.axis("off")
            ax.text(
                cutout_height/2,
                cutout_width/2,
                "No image found",
                horizontalalignment="center",
                color="w"
            )

        else:
            ax = plot_cutout(
                ax,
                path_i,
                ra_i,
                dec_i,
                vra_i,
                vdec_i,
                height=cutout_height,
                width=cutout_width,
                crosshair=crosshair,
                crosshair_kwargs=crosshair_kwargs_i,
                velocity_vector=velocity_vector,
                velocity_vector_kwargs=velocity_vector_kwargs
            )

        ax.set_title(title, fontsize=6, y=y)

        axs.append(ax)

    return fig, axs
