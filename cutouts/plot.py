import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize
from astropy.visualization import ZScaleInterval

CMAP_BONE = matplotlib.cm.bone.copy()
CMAP_BONE.set_bad("black", 100.)

def add_crosshair(ax, wcs, ra, dec, gap=8, length=8, x_offset=0, y_offset=0, **kwargs):

    # Get pixel location of RA and Dec
    xcenter, ycenter = wcs.world_to_pixel_values(ra, dec)
    xcenter = xcenter + x_offset
    ycenter = ycenter + y_offset

    ax.vlines(
        xcenter,
        ycenter + gap,
        ycenter + gap + length,
        **kwargs
    )
    ax.vlines(
        xcenter,
        ycenter - gap,
        ycenter - gap - length,
        **kwargs
    )
    ax.hlines(
        ycenter,
        xcenter + gap,
        xcenter + gap + length,
        **kwargs
    )
    ax.hlines(
        ycenter,
        xcenter - gap,
        xcenter - gap - length,
        **kwargs
    )
    return

def add_velocity_vector(ax, wcs, ra, dec, vra, vdec, gap=8, length=8, x_offset=0, y_offset=0, **kwargs):

    # Get pixel location of RA and Dec
    xcenter, ycenter = wcs.world_to_pixel_values(ra, dec)
    xcenter = xcenter + x_offset
    ycenter = ycenter + y_offset

    dt = 1/24/2
    xoffset, yoffset = wcs.world_to_pixel_values(ra + vra * dt, dec + vdec * dt)
    xoffset = xoffset + x_offset
    yoffset = yoffset + y_offset
    vx = (xoffset - xcenter) / dt
    vy = (yoffset - ycenter) / dt

    vx_hat = vx / np.sqrt(vx**2 + vy**2)
    vy_hat = vy / np.sqrt(vx**2 + vy**2)


    ax.arrow(
        xcenter + gap*vx_hat,
        ycenter + gap*vy_hat,
        length*vx_hat,
        length*vy_hat,
        length_includes_head=True,
        **kwargs
    )
    return

def center_image(image, wcs, ra, dec, height=115, width=115):

    # Calculate where RA and Dec fall in the actual image
    image_y_center, image_x_center = wcs.world_to_array_index_values(ra, dec)
    # The following is not a typo: note x,y change
    pimage_x_center, pimage_y_center = wcs.world_to_pixel_values(ra, dec)

    xbit = 0
    if image_x_center - pimage_x_center > 0:
        xbit = 1

    ybit = 0
    if image_y_center - pimage_y_center > 0:
        ybit = 1

    image_copy = image.copy()
    y_len, x_len = image_copy.shape
    y_offset = np.round(y_len / 2, 0).astype(int) - image_y_center
    x_offset = np.round(x_len / 2, 0).astype(int) - image_x_center

    if y_offset > 0:
        top_padding = np.zeros((y_offset, image_copy.shape[1]))
        image_copy = np.vstack([top_padding, image_copy])
    elif y_offset < 0:
        bottom_padding = np.zeros((-y_offset, image_copy.shape[1]))
        image_copy = np.vstack([image_copy, bottom_padding])

    if x_offset < 0:
        left_padding = np.zeros((image_copy.shape[0], -x_offset))
        image_copy = np.hstack([left_padding, image_copy])
    elif x_offset > 0:
        right_padding = np.zeros((image_copy.shape[0], x_offset))
        image_copy = np.hstack([image_copy, right_padding])

    y_len, x_len = image_copy.shape

    # If the image is not the desired height, pad more rows
    # until it is
    if y_len < height:
        num_rows = height - y_len
        for i in range(num_rows):
            padding = np.zeros((1, image_copy.shape[1]))
            if i % 2 == (0 + ybit):
                y_offset += 1
                image_copy = np.vstack([padding, image_copy])
            else:
                image_copy = np.vstack([image_copy, padding])
    # If the image is larger than the desired height, remove
    # rows until it is
    if y_len > height:
        num_rows = y_len - height
        for i in range(num_rows):
            if i % 2 == (0 + ybit):
                image_copy = image_copy[1:, :]
                y_offset -= 1
            else:
                image_copy = image_copy[:-1, ]

    # If the image is not the desired width, pad more columns
    # until it is
    if x_len < width:
        num_cols = width - x_len
        for i in range(num_cols):
            padding = np.zeros((image_copy.shape[0], 1))
            if i % 2 == (0 + xbit):
                image_copy = np.hstack([padding, image_copy])
                x_offset += 1
            else:
                image_copy = np.hstack([image_copy, padding])

    # If the image is larger than the desired width, remove
    # columns until it is
    if x_len > width:
        num_cols = x_len - width
        for i in range(num_cols):
            if i % 2 == (0 + xbit):
                image_copy = image_copy[:, 1:]
                x_offset -= 1
            else:
                image_copy = image_copy[:, :-1]

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
        subplots_adjust_kwargs={
            "hspace": 0.15,
            "wspace": 0.15,
            "left": 0.05,
            "right": 0.95,
            "top": 0.95,
            "bottom": 0.02
        },
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

        ax = fig.add_subplot(num_rows, max_cols, i+1)
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
            crosshair_kwargs=crosshair_kwargs,
            velocity_vector=velocity_vector,
            velocity_vector_kwargs=velocity_vector_kwargs
        )
        axs.append(ax)

        y = 1.0
        title = ""
        title += f"{times[i].iso}"
        if include_filter:
            title += f"\n{filter[i]}"
        if include_mag:
            title += f": {mag[i]:.2f}"
            y -= 0.03
        if include_mag_sigma:
            title += f"$\pm{mag_sigma[i]:.2f}$"

        ax.set_title(title, fontsize=6, y=y)

    return fig, axs
