from __future__ import annotations

SUPPORTED_LANG = ("English", "中文")

TEXT = {
    "title": {"English": "😷 Your Mask Plz: Face Mask Detection", "中文": "😷 Your Mask Plz 口罩检测"},
    "intro": {
        "English": "Upload an image or video to detect mask wearing (YOLOv11n ONNX).",
        "中文": "上传图片或视频，进行口罩佩戴检测（基于 YOLOv11n ONNX）。",
    },
    "settings": {"English": "Settings", "中文": "设置"},
    "conf": {"English": "Confidence Threshold", "中文": "置信度阈值"},
    "model": {"English": "Model Path", "中文": "模型路径"},
    "frame_step": {"English": "Video frame step (skip N-1 frames)", "中文": "视频抽帧步长 (跳过 N-1 帧)"},
    "loaded": {"English": "Model loaded", "中文": "模型加载成功"},
    "upload": {"English": "Choose image or video...", "中文": "选择图片或视频..."},
    "orig_img": {"English": "Original", "中文": "原始图片"},
    "detect_btn": {"English": "Detect", "中文": "开始检测"},
    "detecting": {"English": "Detecting...", "中文": "检测中..."},
    "result_img": {"English": "Result", "中文": "检测结果"},
    "no_detect": {"English": "No mask/face detected.", "中文": "未检测到口罩/未佩戴口罩目标。"},
    "count": {"English": "Detected {n} objects, counts: {c}", "中文": "检测到 {n} 个目标，类别计数：{c}"},
    "orig_vid": {"English": "Original Video", "中文": "原始视频"},
    "result_vid": {"English": "Detection", "中文": "检测结果"},
    "detect_vid_btn": {"English": "Detect Video", "中文": "开始检测视频"},
    "video_done": {"English": "Video processed!", "中文": "视频处理完成！"},
    "video_summary": {
        "English": "Processed frames: {p}, shown: {s}, class counts: {c}",
        "中文": "已处理帧数：{p}，展示帧数：{s}，类别累计：{c}",
    },
    "no_frame": {"English": "Cannot read video frame", "中文": "无法读取视频帧"},
    "model_missing": {"English": "Model not found: {p}", "中文": "未找到模型文件：{p}"},
    "model_err": {"English": "Load model error: {e}", "中文": "加载模型出错：{e}"},
    "model_warn": {"English": "Model missing. Check path.", "中文": "未能加载模型，请检查模型路径是否正确。"},
}


def t(key: str, lang: str, **kwargs) -> str:
    """Translate key for lang with optional formatting."""
    template = TEXT[key][lang]
    return template.format(**kwargs) if kwargs else template
