import cv2
import streamlit as st

from src.i18n import TEXT
from src.pipeline import (
    choose_display_width,
    infer_video_stream,
    inspect_video,
    save_temp_video,
)


def render_video_flow(uploaded_file, model, conf_threshold: float, frame_step: int, lang: str):
    """Render video upload flow with detection, slider, and state handling."""
    state = st.session_state.setdefault("video_state", {})
    file_id = f"{uploaded_file.name}-{getattr(uploaded_file, 'size', 0)}"
    slider_key = f"video_frame_slider-{file_id}"

    if state.get("file_id") != file_id:
        state.clear()
        state.update(
            {
                "file_id": file_id,
                "frames": None,
                "counts": {},
                "processed": 0,
                "display_w": None,
                "video_path": None,
                "selected_frame": None,
                "last_run_complete": False,
            }
        )
        st.session_state.pop(slider_key, None)

    video_path = save_temp_video(uploaded_file, suffix=".mp4")
    state["video_path"] = video_path

    st.subheader("视频输入 / 输出" if lang == "中文" else "Video Input / Output")
    col_vid_in, col_vid_out = st.columns(2)

    with col_vid_in:
        st.caption(TEXT["orig_vid"][lang])
        st.video(video_path)

    with col_vid_out:
        st.caption(TEXT["result_vid"][lang])
        result_area = st.empty()
        stats_area = st.empty()

    ctrl_detect_col, ctrl_slider_col = st.columns([1, 3])
    with ctrl_detect_col:
        start_detection = st.button(TEXT["detect_vid_btn"][lang], type="primary")
    with ctrl_slider_col:
        frames = state.get("frames")
        sel = None
        if frames and len(frames) >= 2:
            default_sel = state.get("selected_frame") or len(frames)
            sel = st.slider(
                "浏览检测帧 / Browse frames",
                min_value=1,
                max_value=len(frames),
                value=min(default_sel, len(frames)),
                step=1,
                key=slider_key,
            )
            state["selected_frame"] = sel
        elif frames and len(frames) == 1:
            sel = 1
            state["selected_frame"] = 1
            st.slider(
                "浏览检测帧 / Browse frames",
                min_value=1,
                max_value=2,
                value=1,
                step=1,
                disabled=True,
                key=slider_key,
            )
        else:
            st.slider(
                "浏览检测帧 / Browse frames",
                min_value=0,
                max_value=1,
                value=0,
                step=1,
                disabled=True,
                key=slider_key,
            )

    if frames:
        chosen = frames[sel - 1]
        display_w = state.get("display_w") or choose_display_width(640)
        result_area.image(
            chosen[2],
            caption=f"Frame {chosen[0]} (shown {chosen[1]})",
            width=display_w,
        )
        stats_area.info(
            TEXT["video_summary"][lang].format(
                p=state.get("processed", len(frames)),
                s=chosen[1],
                c=state.get("counts", {}),
            )
        )
        if state.get("last_run_complete"):
            st.success(TEXT["video_done"][lang])
    else:
        result_area.info(TEXT["detect_vid_btn"][lang])
        stats_area.empty()

    if start_detection:
        state["last_run_complete"] = False
        with col_vid_out:
            progress_bar = st.progress(0)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_meta = inspect_video(cap)
        if video_meta is None:
            st.error(TEXT["no_frame"][lang])
            st.stop()
        video_imgsz, display_w = video_meta

        processed_frames = 0
        last_counts = {}
        plotted_frames = []

        for frame_index, shown_index, plotted, cls_counts in infer_video_stream(
            cap, model, conf_threshold, frame_step, video_imgsz
        ):
            if total_frames > 0:
                progress_bar.progress(min(frame_index / total_frames, 1.0))

            processed_frames += 1
            last_counts = cls_counts
            plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            plotted_frames.append((frame_index, shown_index, plotted_rgb))
            stats_area.info(
                TEXT["video_summary"][lang].format(
                    p=processed_frames, s=shown_index, c=last_counts
                )
            )

        cap.release()
        state.update(
            {
                "frames": plotted_frames,
                "counts": last_counts,
                "processed": processed_frames,
                "display_w": display_w,
                "selected_frame": len(plotted_frames) if plotted_frames else None,
                "last_run_complete": True,
            }
        )

        if plotted_frames:
            st.rerun()
