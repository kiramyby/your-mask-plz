import os

import streamlit as st
from ultralytics.models import YOLO

from src.i18n import t


@st.cache_resource
def load_model(path: str, lang: str):
    """Load YOLO model with basic error handling and caching."""
    if not os.path.exists(path):
        st.error(t("model_missing", lang, p=path))
        return None
    try:
        return YOLO(path, task="detect")
    except Exception as e:  # noqa: BLE001
        st.error(t("model_err", lang, e=e))
        return None
