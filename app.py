import streamlit as st

from src.config import (
    CONF_STEP,
    DEFAULT_CONF,
    FRAME_STEP_DEFAULT,
    FRAME_STEP_MAX,
    FRAME_STEP_MIN,
    MODEL_PATH_DEFAULT,
)
from src.i18n import SUPPORTED_LANG, TEXT
from src.services.model_loader import load_model
from src.ui.image_flow import render_image_flow
from src.ui.video_flow import render_video_flow

# Page config
st.set_page_config(
    page_title="Your Mask Plz - Detection App",
    page_icon="😷",
    layout="wide",
)

lang = st.sidebar.radio("Language / 语言", SUPPORTED_LANG, horizontal=True)

st.title(TEXT["title"][lang])
st.markdown(TEXT["intro"][lang])

# Sidebar settings
st.sidebar.header(TEXT["settings"][lang])
conf_threshold = st.sidebar.slider(TEXT["conf"][lang], 0.0, 1.0, DEFAULT_CONF, CONF_STEP)
model_path = st.sidebar.text_input(TEXT["model"][lang], MODEL_PATH_DEFAULT)
video_frame_step = st.sidebar.slider(
    TEXT["frame_step"][lang], FRAME_STEP_MIN, FRAME_STEP_MAX, FRAME_STEP_DEFAULT, 1
)


model = load_model(model_path, lang)

if model:
    st.sidebar.success(TEXT["loaded"][lang])

    uploaded_file = st.file_uploader(
        TEXT["upload"][lang], type=["jpg", "jpeg", "png", "mp4", "mov", "avi"]
    )

    if uploaded_file is not None:
        file_type = uploaded_file.type.split("/")[0]

        if file_type == "image":
            render_image_flow(uploaded_file, model, conf_threshold, lang)

        elif file_type == "video":
            render_video_flow(uploaded_file, model, conf_threshold, video_frame_step, lang)

else:
    st.warning(TEXT["model_warn"][lang])
