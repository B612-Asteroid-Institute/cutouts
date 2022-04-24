import numpy as np

def add_crosshair(ax, wcs, ra, dec, gap=8, length=8, x_offset=0, y_offset=0, **kwargs):

    # Get pixel location of RA and Dec
    ycenter, xcenter = wcs.world_to_pixel_values(ra, dec)
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
    ycenter, xcenter = wcs.world_to_pixel_values(ra, dec)
    xcenter = xcenter + x_offset
    ycenter = ycenter + y_offset

    dt = 1/24/2
    yoffset, xoffset = wcs.world_to_pixel_values(ra + vra * dt, dec + vdec * dt)
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

    # Calculate where RA and Dec should fall in the centered
    # image
    x_center = np.round(width / 2, 0).astype(int)
    y_center = np.round(height / 2, 0).astype(int)

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

    y_offset = y_center - image_y_center
    x_offset = x_center - image_x_center
    image_copy = image.copy()

    # Center ra, dec t
    if y_offset > 0:
        top_padding = np.zeros((y_offset, image_copy.shape[1]))
        image_copy = np.vstack([top_padding, image_copy])
    elif y_offset < 0:
        bottom_padding = np.zeros((-y_offset, image_copy.shape[1]))
        image_copy = np.vstack([image_copy, bottom_padding])

    if x_offset > 0:
        left_padding = np.zeros((image_copy.shape[0], x_offset))
        image_copy = np.hstack([left_padding, image_copy])
    elif x_offset < 0:
        right_padding = np.zeros((image_copy.shape[0], -x_offset))
        image_copy = np.hstack([image_copy, right_padding])

    y_len, x_len = image_copy.shape

    # If the image is not the desired height, pad more rows
    # until it is
    if y_len < height:
        num_rows = height - y_len
        for i in range(num_rows):
            padding = np.zeros((1, image_copy.shape[1]))
            if i % 2 == 0 + ybit:
                image_copy = np.vstack([padding, image_copy])
            else:
                image_copy = np.vstack([image_copy, padding])
    # If the image is larger than the desired height, remove
    # rows until it is
    if y_len > height:
        num_rows = y_len - height
        for i in range(num_rows):
            if i % 2 == 0 + ybit:
                image_copy = image_copy[1:, :]
            else:
                image_copy = image_copy[:-1, ]

    # If the image is not the desired width, pad more columns
    # until it is
    if x_len < width:
        num_cols = width - x_len
        for i in range(num_cols):
            padding = np.zeros((image_copy.shape[0], 1))
            if i % 2 == 0 + xbit:
                image_copy = np.hstack([padding, image_copy])
            else:
                image_copy = np.hstack([image_copy, padding])

    # If the image is larger than the desired width, remove
    # columns until it is
    if x_len > width:
        num_cols = x_len - width
        for i in range(num_cols):
            if i % 2 == 0 + xbit:
                image_copy = image_copy[:, :-1]
            else:
                image_copy = image_copy[:, 1:]

    return image_copy, x_offset, y_offset