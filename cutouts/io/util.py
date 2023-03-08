def exposure_id_from_url(
    url: str, preamble: str = "siaRef=", postamble: str = ".fits.fz"
) -> str:
    """
    Attempt to determine the exposure ID from a cutout URL.

    Parameters
    ----------
    url : str
        URL to remote cutout.
    preamble : str, optional
        URL component expected directly infront of the exposure ID.
    postamble : str, optional
        URL component expected directly after the exposure ID. This
        is sometimes the file extension.

    Returns
    -------
    exposure_id : str
        Exposure ID read from URL.
    """
    id_start = url.find(preamble)
    id_end = url.find(postamble)
    exposure_id = url[id_start + len(preamble) : id_end]
    return exposure_id
