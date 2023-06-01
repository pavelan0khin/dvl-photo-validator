import io
import math
from datetime import datetime
from functools import cached_property
from typing import BinaryIO

import cv2
import dlib
import numpy as np
from dlib import rectangle, shape_predictor
from PIL import ExifTags, Image, ImageStat

from dvlpv.utils import settings, types
from dvlpv.utils.exceptions import PhotoValidatorException


class PhotoValidator:
    def __init__(self, image: str | bytes | io.BytesIO | BinaryIO):
        """
        Initialize PhotoValidator instance. Photo verification is done in accordance
        with the requirements of the US Department of State:
        https://travel.state.gov/content/travel/en/us-visas/visa-information-resources/photos.html

        :param image : str | bytes | io.BytesIO | BinaryIO
        The image object which can be provided in one of the following formats:
        - a string representing the file path to the image;
        - bytes representing the image;
        - io.BytesIO or BinaryIO objects containing the image.
        """
        self._image = image
        self.images: types.Images
        self._init_images()
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(settings.SHAPE_PREDICTOR_PATH)
        self.shape: shape_predictor

    def _init_images(self):
        if isinstance(self._image, bytes):
            self._image = io.BytesIO(self._image)
        pil_image = self._fix_image_orientation(Image.open(self._image))
        nd_array = np.array(pil_image)  # noqa
        self.images = types.Images(pil_image=pil_image, nd_array=nd_array)

    @property
    def allowed_head_rotation_percent(self) -> float:
        current_value = settings.ALLOWED_HEAD_ROTATION_PERCENT
        if 100 > current_value > 0:
            return current_value / 100
        raise ValueError(
            f"The value of the ALLOWED_HEAD_ROTATION_PERCENT environment variable must "
            f"be between 0 and 100, but the current value is {current_value}"
        )

    @staticmethod
    def _fix_image_orientation(image: Image) -> Image:
        try:
            exif = image._getexif()
            orientation = None
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag) == "Orientation":
                    orientation = value
                    break
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            pass
        return image

    @cached_property
    def face(self) -> rectangle:
        faces = self.face_detector(self.images.nd_array)
        if not faces:
            raise PhotoValidatorException("Faces not found")
        return faces[0]

    @cached_property
    def eyes_y(self) -> int:
        left_eye_corner = self.shape.part(types.SPFLandMark.LEFT_EYE_INNER_CORNER)
        right_eye_corner = self.shape.part(types.SPFLandMark.RIGHT_EYE_INNER_CORNER)
        return int((left_eye_corner.y + right_eye_corner.y) / 2)

    @cached_property
    def chin_y(self) -> int:
        return int(self.shape.part(types.SPFLandMark.CHIN_LOWER_PART).y)

    @cached_property
    def nose_x(self) -> int:
        return int(self.shape.part(types.SPFLandMark.NOSE_TIP).x)

    @cached_property
    def head_top_y(self) -> int:
        gray_image = cv2.cvtColor(self.images.nd_array, cv2.COLOR_BGR2GRAY)
        gauss_k_size = 5, 5
        median_k_size = 7
        gauss_sigma_x = 0
        brightness_normalize_value = 255
        block_size = 11
        average_const = 2
        new_array_shape = 3, 3
        blurred_image = cv2.GaussianBlur(gray_image, gauss_k_size, gauss_sigma_x)
        threshold_image = cv2.adaptiveThreshold(
            blurred_image,
            brightness_normalize_value,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            average_const,
        )
        kernel = np.ones(new_array_shape, np.uint8)
        threshold_image = cv2.morphologyEx(
            threshold_image, cv2.MORPH_CLOSE, kernel, iterations=2
        )
        threshold_image = cv2.medianBlur(threshold_image, median_k_size)
        contours, _ = cv2.findContours(
            threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest_contour = max(contours, key=cv2.contourArea)
        return int(min(largest_contour, key=lambda point: point[0][1])[0][1])

    @cached_property
    def date_valid(self) -> bool:
        max_photo_age_days = 180
        try:
            raw_date = self.images.pil_image._getexif()[types.ExifTagID.PHOTO_DATE]
        except (KeyError, TypeError, AttributeError):
            return True
        photo_date = datetime.strptime(raw_date, "%Y:%m:%d %H:%M:%S").date()
        current_date = datetime.now().date()
        delta = current_date - photo_date
        return delta.days < max_photo_age_days

    @cached_property
    def image_grayscale(self) -> bool:
        stat = ImageStat.Stat(self.images.pil_image)
        if sum(stat.sum) / 3 == stat.sum[0]:
            return True
        else:
            return False

    @cached_property
    def size_valid(self) -> bool:
        width, height = self.images.pil_image.size
        return width > 600 and height > 600

    @cached_property
    def face_centered(self) -> bool:
        """
        This method checks if the head in the photo is rotated (horizontally). The
        check is made by calculating the real value of the position of the tip of
        the nose, as well as by calculating the expected value of the nose
        (the middle between the two pupils)
        :return: True if the difference between the actual position of the nose and the expected
                 position does not exceed the value from the property-method allowed_head_rotation_percent.
                 You can set your own value for this percentage via the 'ALLOWED_HEAD_ROTATION_PERCENT'
                 environment variable
        """
        left_eye_outer_corner = self.shape.part(
            types.SPFLandMark.LEFT_EYE_OUTER_CORNER
        ).x
        left_eye_inner_corner = self.shape.part(
            types.SPFLandMark.LEFT_EYE_INNER_CORNER
        ).x
        left_eye_center_x = (left_eye_outer_corner + left_eye_inner_corner) / 2
        right_eye_outer_corner = self.shape.part(
            types.SPFLandMark.RIGHT_EYE_OUTER_CORNER
        ).x
        right_eye_inner_corner = self.shape.part(
            types.SPFLandMark.RIGHT_EYE_INNER_CORNER
        ).x
        right_eye_center_x = (right_eye_outer_corner + right_eye_inner_corner) / 2
        real_nose_x_coordinate = self.nose_x
        expected_nose_x_coordinate = int(left_eye_center_x + right_eye_center_x) / 2
        return abs(
            real_nose_x_coordinate - expected_nose_x_coordinate
        ) <= self.allowed_head_rotation_percent * abs(expected_nose_x_coordinate)

    def _crop_image(self):
        head_height = self.chin_y - self.head_top_y
        image_width, image_height = self.images.pil_image.size
        min_height = (head_height * 100) / 69
        max_height = (head_height * 100) / 50
        if min_height > image_height or max_height > image_height:
            raise ValueError("The image is too small for the specified head size")
        new_height = min(max_height, image_height)
        bottom = self.eyes_y + int(new_height * (1 - (1 - 0.62)))
        top = bottom - new_height
        if top < 0:
            top = 0
            bottom = new_height
        elif bottom > image_height:
            top = image_height - new_height
            bottom = image_height
        new_width = new_height
        left = self.nose_x - new_width // 2
        right = left + new_width
        if left < 0:
            left = 0
            right = new_width
        elif right > image_width:
            left = image_width - new_width
            right = image_width
        cropped = self.images.pil_image.crop((left, top, right, bottom))
        if new_height != 600:
            final_image = cropped.copy()
            final_image.thumbnail((600, 600))
        else:
            final_image = cropped
        self.images.cropped_image = final_image

    @property
    def compression_ratio(self) -> int:
        original_width, original_height = self.images.pil_image.size
        final_width, final_height = self.images.cropped_image.size
        original_area = original_width * original_height
        final_area = final_width * final_height
        return math.ceil(original_area / final_area)

    def validate(self):
        self.shape = self.shape_predictor(self.images.nd_array, self.face)
        return None
