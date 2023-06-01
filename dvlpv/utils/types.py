from dataclasses import dataclass

import numpy as np
from dlib import rectangle
from PIL import Image


@dataclass
class FaceSearchResult:
    face: rectangle | None
    eyes_y: int | None
    chin_y: int | None
    top_head_y: int | None
    nose_x: int | None


@dataclass
class Images:
    pil_image: Image
    nd_array: np.ndarray


class ImageFormat:
    RGB = "RGB"


class SPFLandMark:
    """
    SPF stands for 'Shape Predictor Face'

    You'll find more about landmarks here:
    https://miro.medium.com/v2/resize:fit:1400/format:webp/1*tn5D7BMcvq57-T8qy7_tUQ.png
    """

    RIGHT_EYE_INNER_CORNER = 40
    RIGHT_EYE_OUTER_CORNER = 37
    LEFT_EYE_INNER_CORNER = 43
    LEFT_EYE_OUTER_CORNER = 46
    CHIN_LOWER_PART = 9
    NOSE_TIP = 33


class ExifTagID:
    PHOTO_DATE = 36867
