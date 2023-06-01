from dataclasses import dataclass
from datetime import datetime
from typing import Union

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
    cropped_image: Image = None


class ImageFormat:
    RGB = "RGB"


class SPFLandMark:
    """
    SPF stands for 'Shape Predictor Face'

    You can find more about landmarks here:
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


@dataclass
class Field:
    is_valid: bool
    value: Union[str, int, bool, datetime.date]


@dataclass
class ValidationResult:

    """
    This class contains the results of all checks that have been performed on the photo.
    """

    is_valid: bool = False
    photo_width: Field = None
    photo_height: Field = None
    file_size: Field = None
    compression_ratio: Field = None
    horizontal_resolution: Field = None
    vertical_resolution: Field = None
    top_head_point: Field = None
    low_head_point: Field = None
    head_size: Field = None
    eyes_line: Field = None
    eyes_height: Field = None
    eyes_open: Field = None
    red_eyes: Field = None
    head_rotated: Field = None
    face_centered: Field = None
    face_shadows: Field = None
    file_type: Field = None
    color_space: Field = None
    color_depth: Field = None
    background_uniform: Field = None
    background_neutral: Field = None
    background_light: Field = None
    background_changed: Field = None
    photo_date: Field = None
    file_name_valid: Field = None
