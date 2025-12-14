# Your Mask Plz 😷

**Deep Learning Course Final Project | 深度学习课程 大作业**

A lightweight, real-time face mask detection application, designed for local deployment. Native support for Mac Mini M4, MacOS 26.0.1

一个轻量级、实时的面部口罩检测应用，专为本地部署设计原生支持 Mac Mini M4，MacOS 26.0.1

![Streamlit App](streamlit_app.png)

## Features / 功能特性

- **Core Model**: Powered by **YOLOv11n (Nano)**. Lightweight and high-speed inference, optimized for CPU (ONNX).
- **Interactive UI**: Built with **Streamlit**. Seamless Python integration for instant deep learning visualization.
- **Media Support**: Detect masks in both **Images** and **Videos**.
- **Bilingual**: Full support for **English** and **Chinese (中文)** interfaces.
- **Customizable**: Adjustable confidence thresholds and video frame skipping for performance tuning.

---

- **核心模型**：基于 **YOLOv11n (Nano)**轻量级，提供高速推理，无需 CUDA 支持（使用 ONNX）
- **交互式界面**：使用 **Streamlit** 构建无缝集成 Python，即时可视化深度学习结果
- **多媒体支持**：支持 **图片** 和 **视频** 的口罩检测
- **双语支持**：完全支持 **英文** 和 **中文** 界面切换
- **可定制化**：支持调节置信度阈值和视频抽帧步长，以平衡性能与精度

## Project Structure / 项目结构

```text
.
├── app.py                  # Streamlit Application Entry / 应用主程序
├── src/                    # App modules / 应用模块
│   ├── config.py           # UI defaults & constants
│   ├── i18n.py             # Text + translation helper
│   ├── pipeline.py         # Image/video preprocessing & inference helpers
│   ├── services/           # Service layer (model loading, etc.)
│   │   └── model_loader.py # Cached YOLO loader with UI errors
│   └── ui/                 # UI flows for image/video
│       ├── image_flow.py   # Image upload + detect + display
│       └── video_flow.py   # Video upload + detect + slider state
├── face_mask.yaml          # Dataset Configuration / 数据集配置文件
├── dataset/                # Dataset Directory / 数据集目录
│   ├── get_dataset.sh      # Script to download & prep data / 数据准备脚本
│   ├── images/             # Image data / 图片数据
│   └── labels/             # Label data / 标签数据
├── runs/                   # Training Outputs / 训练输出
│   └── detect/train/weights/best.onnx  # Trained Model / 训练好的模型
└── README.md               # Documentation / 项目文档
```

## Quick Start / 快速开始

### 1. Environment Setup / 环境安装
Requires Python 3.10+. / 需要 Python 3.10 及以上版本

```zsh
pip install "ultralytics>=8.3" opencv-python pillow streamlit
```

### 2. Run Application / 运行应用
Launch the web interface locally. / 在本地启动 Web 界面

```zsh
streamlit run app.py
```

Follow the terminal output to visit the app (usually `http://localhost:8501`).
请跟随终端输出访问应用（通常为 `http://localhost:8501`）

Notes / 说明：
- Default model path is `runs/detect/train/weights/best.onnx`. Update in the sidebar if you export elsewhere.
- Video inference uses a frame step slider to balance speed vs. coverage.

---
- 默认模型路径为 `runs/detect/train/weights/best.onnx`，如果导出到其他位置请在侧边栏更新。
- 视频推理使用帧步长滑块，可在性能与覆盖率之间进行权衡。

## Training Pipeline / 训练流程

### 1. Prepare Dataset / 准备数据集
Download dataset from Kaggle and format it. / 从 Kaggle 下载并格式化数据集

```zsh
cd dataset
chmod +x get_dataset.sh
sh ./get_dataset.sh
cd ..
```

**Dataset Config (`face_mask.yaml`)**:
```yaml
path: dataset/images
train: train
val: valid
test: test
names: [no_mask, mask]
```

### 2. Train / 训练
Train using YOLOv11n baseline. Adjust epochs/batch as needed.
使用 YOLOv11n 基线进行训练按需调整 epoch 和 batch 大小

```zsh
yolo task=detect mode=train model=yolo11n.pt data=face_mask.yaml epochs=100 imgsz=640 batch=16 lr0=0.01 warmup_epochs=3 cos_lr=True
```

### 3. Evaluate / 评估
Validate the model performance. / 验证模型性能

**Validation Set (验证集)**:
Used for tuning and checking training progress. / 用于调优和检查训练进度。
```zsh
yolo mode=val model=runs/detect/train/weights/best.pt data=face_mask.yaml
```

**Test Set (测试集)**:
Used for final performance evaluation on unseen data. / 用于最终评估模型在未见数据上的表现。
```zsh
yolo mode=val model=runs/detect/train/weights/best.pt data=face_mask.yaml split=test
```

### 4. Predict Check / 推理自检
Run a quick prediction on validation set. / 在验证集上进行快速推理检查

```zsh
yolo mode=predict model=runs/detect/train/weights/best.pt source=dataset/images/valid save=True
```

### 5. Export / 导出
Export to ONNX format for the Streamlit app. / 导出为 ONNX 格式以供 Streamlit 应用使用

```zsh
yolo mode=export model=runs/detect/train/weights/best.pt format=onnx
```

My Export Record / 我的导出记录

![Result Export](result_export.png)
