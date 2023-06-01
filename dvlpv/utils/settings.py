import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

SHAPE_PREDICTOR_PATH = os.getenv(
    "SHAPE_PREDICTOR_PATH",
    os.path.join(ROOT_DIR, "bin", "shape_predictor_68_face_landmarks.dat"),
)

ALLOWED_HEAD_ROTATION_PERCENT = int(os.getenv("ALLOWED_HEAD_ROTATION_PERCENT", 5))
