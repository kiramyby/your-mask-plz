import os
import tempfile

import cv2
import streamlit as st
from PIL import Image
from ultralytics.models import YOLO

# Page config
st.set_page_config(
    page_title="Your Mask Plz - Detection App",
    page_icon="😷",
    layout="wide"
)

lang = st.sidebar.radio("Language / 语言", ("English", "中文"), horizontal=True)

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

st.title(TEXT["title"][lang])
st.markdown(TEXT["intro"][lang])

# Sidebar settings
st.sidebar.header(TEXT["settings"][lang])
conf_threshold = st.sidebar.slider(TEXT["conf"][lang], 0.0, 1.0, 0.25, 0.05)
model_path = st.sidebar.text_input(TEXT["model"][lang], "runs/detect/train/weights/best.onnx")
video_frame_step = st.sidebar.slider(TEXT["frame_step"][lang], 1, 5, 1, 1)

# 根据输入尺寸自适应推理与展示参数
def choose_imgsz(width: int, height: int) -> int:
    """依据最短边选择推理尺寸，避免过大分辨率拖慢推理。"""
    short = min(width, height)
    if short >= 1600:
        return 960
    if short >= 960:
        return 768
    return 640

# 载入模型（缓存以避免重复加载）
@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        st.error(TEXT["model_missing"][lang].format(p=path))
        return None
    try:
        return YOLO(path, task="detect")
    except Exception as e:  # noqa: BLE001
        st.error(TEXT["model_err"][lang].format(e=e))
        return None

model = load_model(model_path)

# Main content
if model:
    st.sidebar.success(TEXT["loaded"][lang])

    # 文件上传
    uploaded_file = st.file_uploader(
        TEXT["upload"][lang], type=["jpg", "jpeg", "png", "mp4", "mov", "avi"]
    )

    if uploaded_file is not None:
        file_type = uploaded_file.type.split("/")[0]

        if file_type == "image":
            # 读取图片并转换为 RGB
            image = Image.open(uploaded_file).convert("RGB")
            img_w, img_h = image.size
            imgsz = choose_imgsz(img_w, img_h)
            # 依据原图宽度设定展示宽度（不让过大/过小）
            display_w = min(max(img_w, 480), 1200)

            st.subheader("图片输入 / 输出" if lang == "中文" else "Image Input / Output")
            col_in, col_out = st.columns(2)
            with col_in:
                st.caption(TEXT["orig_img"][lang])
                st.image(image, width=display_w)

            if st.button(TEXT["detect_btn"][lang], type="primary"):
                with st.spinner(TEXT["detecting"][lang]):
                    # 推理
                    results = model.predict(image, conf=conf_threshold, imgsz=imgsz)
                    # 调整框线与文字大小，避免遮挡
                    res_plotted = results[0].plot(line_width=1, font_size=10)
                    with col_out:
                        st.caption(f"{TEXT['result_img'][lang]} (imgsz={imgsz})")
                        st.image(res_plotted, width=display_w)

                    # 统计检测数量
                    boxes = results[0].boxes
                    if boxes:
                        names = model.names
                        cls_counts = {}
                        for c in boxes.cls:
                            c_name = names[int(c)]
                            cls_counts[c_name] = cls_counts.get(c_name, 0) + 1
                        st.info(TEXT["count"][lang].format(n=len(boxes), c=cls_counts))
                    else:
                        st.warning(TEXT["no_detect"][lang])

        elif file_type == "video":
            # 将上传的视频保存到临时文件
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            st.video(video_path)

            if st.button(TEXT["detect_vid_btn"][lang], type="primary"):
                col_vid_in, col_vid_out = st.columns(2)
                with col_vid_in:
                    st.caption(TEXT["orig_vid"][lang])
                    st.video(video_path)
                with col_vid_out:
                    st.caption(TEXT["result_vid"][lang])
                    st_frame = st.empty()
                cap = cv2.VideoCapture(video_path)

                progress_bar = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_count = 0
                shown_count = 0
                # 类别计数累积
                names = model.names
                cls_counts = {names[i]: 0 for i in names}
                processed_frames = 0

                # 读取首帧获取尺寸，自动决定推理尺寸与展示宽度
                first_ret, first_frame = cap.read()
                if first_ret:
                    h0, w0, _ = first_frame.shape
                    video_imgsz = choose_imgsz(w0, h0)
                    display_w = min(max(w0, 480), 1200)
                    # 首帧推回队列开头，方便主循环统一处理
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    st.error(TEXT["no_frame"][lang])
                    cap.release()
                    # 无法读取首帧，直接退出视频处理分支
                    st.stop()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0))

                    # 按步长抽帧，减少负载（用户可调）
                    if frame_count % video_frame_step != 0:
                        continue

                    # 等比缩放到设定的最短边，降低计算量
                    h, w, _ = frame.shape
                    scale = video_imgsz / min(h, w)
                    if scale != 1:
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                    # BGR -> RGB，再推理
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = model.predict(
                        frame_rgb,
                        conf=conf_threshold,
                        imgsz=video_imgsz,
                        verbose=False,
                    )
                    res_plotted = results[0].plot(line_width=1, font_size=10)

                    # 累积类别计数
                    boxes = results[0].boxes
                    if boxes:
                        for c in boxes.cls:
                            cls_name = names[int(c)]
                            cls_counts[cls_name] = cls_counts.get(cls_name, 0) + 1

                    processed_frames += 1

                    shown_count += 1
                    st_frame.image(
                        res_plotted,
                        caption=f"Frame {frame_count} (shown {shown_count})",
                        width=display_w,
                    )

                cap.release()
                st.success(TEXT["video_done"][lang])
                st.info(
                    TEXT["video_summary"][lang].format(
                        p=processed_frames, s=shown_count, c=cls_counts
                    )
                )
else:
    st.warning(TEXT["model_warn"][lang])
