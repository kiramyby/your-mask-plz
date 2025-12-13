import streamlit as st

from src.i18n import TEXT
from src.pipeline import choose_display_width, predict_image, read_image


def render_image_flow(uploaded_file, model, conf_threshold: float, lang: str):
    """Render image upload, detection trigger, and results."""
    image = read_image(uploaded_file)
    img_w, _ = image.size

    st.subheader("图片输入 / 输出" if lang == "中文" else "Image Input / Output")
    col_in, col_out = st.columns(2)
    with col_in:
        st.caption(TEXT["orig_img"][lang])
        st.image(image, width=choose_display_width(img_w))

    if st.button(TEXT["detect_btn"][lang], type="primary"):
        with st.spinner(TEXT["detecting"][lang]):
            res_plotted, cls_counts, imgsz, display_w = predict_image(
                model, image, conf_threshold
            )
            with col_out:
                st.caption(f"{TEXT['result_img'][lang]} (imgsz={imgsz})")
                st.image(res_plotted, width=display_w)

            if cls_counts:
                st.info(TEXT["count"][lang].format(n=sum(cls_counts.values()), c=cls_counts))
            else:
                st.warning(TEXT["no_detect"][lang])
