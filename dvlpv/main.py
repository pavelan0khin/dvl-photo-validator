import io
import os.path
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
import tempfile


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
        self.shape: shape_predictor | None = None
        self._eyes_y = None
        self._head_top_y = None
        self._chin_y = None
        self._nose_x = None
        self._face = None

    tmp_image_path = os.path.join(settings.ROOT_DIR, "media")
    image_size = (600, 600)
    max_image_size_bt = 240000
    max_compression_ratio = 20
    min_dpi = 72

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

    @property
    def face(self) -> rectangle:
        if not self._face:
            faces = self.face_detector(self.images.nd_array)
            if not faces:
                raise PhotoValidatorException("Faces not found")
            self._face = faces[0]
        return self._face

    @face.setter
    def face(self, value: rectangle):
        self._face = value

    @property
    def eyes_y(self) -> int:
        if not self._eyes_y:
            left_eye_corner = self.shape.part(types.SPFLandMark.LEFT_EYE_INNER_CORNER)
            right_eye_corner = self.shape.part(types.SPFLandMark.RIGHT_EYE_INNER_CORNER)
            self._eyes_y = int((left_eye_corner.y + right_eye_corner.y) / 2)
        return self._eyes_y

    @eyes_y.setter
    def eyes_y(self, value: int):
        self._eyes_y = value

    @property
    def chin_y(self) -> int:
        if not self._chin_y:
            self._chin_y = int(self.shape.part(types.SPFLandMark.CHIN_LOWER_PART).y)
        return self._chin_y

    @chin_y.setter
    def chin_y(self, value: int):
        self._chin_y = value

    @property
    def nose_x(self) -> int:
        if not self._nose_x:
            self._nose_x = int(self.shape.part(types.SPFLandMark.NOSE_TIP).x)
        return self._nose_x

    @nose_x.setter
    def nose_x(self, value: int):
        self._nose_x = value

    @property
    def head_top_y(self):
        if not self._head_top_y:
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
            self._head_top_y = int(min(largest_contour, key=lambda point: point[0][1])[0][1])
        return self._head_top_y

    @head_top_y.setter
    def head_top_y(self, value: int):
        self._head_top_y = value

    @cached_property
    def photo_date(self) -> datetime.date:
        try:
            raw_date = self.images.pil_image._getexif()[types.ExifTagID.PHOTO_DATE]
        except (KeyError, TypeError, AttributeError):
            return True
        photo_date = datetime.strptime(raw_date, "%Y:%m:%d %H:%M:%S").date()
        return photo_date

    @cached_property
    def image_grayscale(self) -> bool:
        stat = ImageStat.Stat(self.images.pil_image)
        if sum(stat.sum) / 3 == stat.sum[0]:
            return True
        else:
            return False

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

    @cached_property
    def original_image_size(self) -> int:
        with tempfile.NamedTemporaryFile(suffix=".jpg", dir=self.tmp_image_path) as jpg:
            self.images.pil_image.save(jpg)
            size = os.path.getsize(jpg.name)
        return size

    @property
    def cropped_image_size(self) -> int:
        with tempfile.NamedTemporaryFile(suffix=".jpg", dir=self.tmp_image_path) as jpg:
            self.images.cropped_image.save(jpg)
            size = os.path.getsize(jpg.name)
        return size

    @property
    def compression_ratio(self) -> int:
        return math.ceil(self.original_image_size / self.cropped_image_size)

    @property
    def horizontal_dpi(self) -> int:
        return int(self.images.cropped_image.info.get('dpi', (None, None))[0])

    @property
    def vertical_dpi(self) -> int:
        return int(self.images.cropped_image.info.get('dpi', (None, None))[1])

    def _crop_image(self):
        head_height = self.chin_y - self.head_top_y
        head_size_percent = 0.6
        eyes_line_percent = 0.63
        expected_image_height = int(head_height / head_size_percent)
        eyes_line = int(expected_image_height * eyes_line_percent)
        top = self.eyes_y - (expected_image_height - eyes_line)
        bottom = top + expected_image_height
        left = self.nose_x - int(expected_image_height / 2)
        right = left + expected_image_height
        cropped_image = self.images.pil_image.crop((left, top, right, bottom))
        self.images.cropped_image = cropped_image.resize(self.image_size, Image.LANCZOS)

    @property
    def _validate_photo_width(self) -> types.Field:
        width = self.images.cropped_image.size[0]
        result = types.Field(width == 600, width)
        return result

    @property
    def _validate_photo_height(self) -> types.Field:
        height = self.images.cropped_image.size[1]
        result = types.Field(height == 600, height)
        return result

    @property
    def _validate_file_size(self) -> types.Field:
        size = self.cropped_image_size
        return types.Field(size < self.max_image_size_bt, size)

    @property
    def _validate_compression_ratio(self) -> types.Field:
        ratio = self.compression_ratio
        return types.Field(ratio < 20, f"{ratio}:1")

    @property
    def _validate_horizontal_resolution(self) -> types.Field:
        dpi = int(self.images.cropped_image.info.get('dpi', (None, None))[0])
        return types.Field(dpi >= self.min_dpi, dpi)

    @property
    def _validate_vertical_resolution(self) -> types.Field:
        dpi = int(self.images.cropped_image.info.get('dpi', (None, None))[1])
        return types.Field(dpi >= self.min_dpi, dpi)

    def validate(self):
        result = types.ValidationResult()
        self.shape = self.shape_predictor(self.images.nd_array, self.face)
        self._crop_image()
        result_attributes = [attr for attr in types.ValidationResult.__dict__.keys() if not attr.startswith("_")]
        for attr in result_attributes:
            method_name = f"_validate_{attr}"
            if hasattr(self, method_name):
                setattr(result, attr, getattr(self, method_name))
