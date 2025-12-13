from __future__ import annotations

import tempfile
from typing import Any, Dict, Iterator, Tuple

import cv2
from PIL import Image

from .config import IMAGE_WIDTH_MAX, IMAGE_WIDTH_MIN, IMGSZ_MAX


def choose_imgsz(width: int, height: int) -> int:
    """Choose inference size by shorter side to avoid over-large resolution."""
    short = min(width, height)
    if short >= 1600:
        return min(960, IMGSZ_MAX)
    if short >= 960:
        return min(768, IMGSZ_MAX)
    return min(640, IMGSZ_MAX)


def choose_display_width(width: int) -> int:
    return min(max(width, IMAGE_WIDTH_MIN), IMAGE_WIDTH_MAX)


def read_image(uploaded_file) -> Image.Image:
    """Load an uploaded image as RGB."""
    return Image.open(uploaded_file).convert("RGB")


def save_temp_video(uploaded_file, suffix: str = ".mp4") -> str:
    """Persist uploaded video to a temp file and return the path."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(uploaded_file.read())
    return tfile.name


def tally_boxes(boxes, names: Dict[int, str]) -> Dict[str, int]:
    if not boxes:
        return {}
    counts: Dict[str, int] = {}
    for cls_idx in boxes.cls:
        name = names[int(cls_idx)]
        counts[name] = counts.get(name, 0) + 1
    return counts


def predict_image(model, image: Image.Image, conf_threshold: float):
    img_w, img_h = image.size
    imgsz = choose_imgsz(img_w, img_h)
    display_w = choose_display_width(img_w)
    results = model.predict(image, conf=conf_threshold, imgsz=imgsz)
    res_plotted = results[0].plot(line_width=1, font_size=10)
    counts = tally_boxes(results[0].boxes, model.names)
    return res_plotted, counts, imgsz, display_w


def inspect_video(cap: cv2.VideoCapture) -> Tuple[int, int] | None:
    """Return (imgsz, display_width) based on first frame, or None if unreadable."""
    first_ret, first_frame = cap.read()
    if not first_ret:
        return None
    height, width, _ = first_frame.shape
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return choose_imgsz(width, height), choose_display_width(width)


def infer_video_stream(
    cap: cv2.VideoCapture,
    model,
    conf_threshold: float,
    frame_step: int,
    video_imgsz: int,
) -> Iterator[Tuple[int, int, Any, Dict[str, int]]]:
    """
    Generate annotated frames with counts for every sampled frame.

    Yields (frame_index, shown_index, plotted_frame, cls_counts).
    """
    names = model.names
    cls_counts = {names[i]: 0 for i in names}
    frame_index = 0
    shown_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        if frame_index % frame_step != 0:
            continue

        height, width, _ = frame.shape
        scale = video_imgsz / min(height, width)
        if scale != 1:
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(
            frame_rgb,
            conf=conf_threshold,
            imgsz=video_imgsz,
            verbose=False,
        )
        plotted = results[0].plot(line_width=1, font_size=10)

        boxes = results[0].boxes
        if boxes:
            for cls_idx in boxes.cls:
                cls_name = names[int(cls_idx)]
                cls_counts[cls_name] = cls_counts.get(cls_name, 0) + 1

        shown_index += 1
        yield frame_index, shown_index, plotted, cls_counts

    cap.release()
