import argparse
import datetime as dt
import os
import queue
import shutil
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v"}

_APP: FaceAnalysis | None = None
_TARGET_EMBEDDING: np.ndarray | None = None
_ARGS: argparse.Namespace | None = None
_CONFIRM_DIR: Path | None = None
_TIMING_LOG: Path | None = None
_DEBUG_LOG: Path | None = None
_LOG_LOCK = None
_PROCESSED_LOCK = None
_PREVIEW_QUEUE = None
_STOP_EVENT = None


def run_with_args_safe(args: argparse.Namespace, error_queue) -> None:
    try:
        run_with_args(args)
    except Exception as exc:
        error_queue.put(str(exc))


@dataclass
class MatchResult:
    matched: bool
    best_score: float
    best_frame: np.ndarray | None
    best_timestamp_s: float | None
    best_frame_index: int | None
    best_bbox: tuple[int, int, int, int] | None


def load_processed_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line}


def append_processed(path: Path, video_name: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{video_name}\n")


def log_line(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def format_timestamp(value: dt.datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def log_line_locked(path: Path, message: str) -> None:
    if _LOG_LOCK is None:
        log_line(path, message)
        return
    with _LOG_LOCK:
        log_line(path, message)




def enqueue_preview(crop: np.ndarray, video_path: Path, timestamp_s: float | None) -> None:
    if _PREVIEW_QUEUE is None:
        return
    try:
        if _CONFIRM_DIR is not None:
            video_path = _CONFIRM_DIR / video_path.name
        _PREVIEW_QUEUE.put_nowait(
            {"crop": crop, "video_path": str(video_path), "timestamp_s": timestamp_s}
        )
    except Exception:
        return


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(embedding)
    if denom <= 0:
        return embedding
    return embedding / denom


def has_motion(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    diff_threshold: int,
    min_ratio: float,
) -> bool:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    changed = cv2.countNonZero(mask)
    total = mask.shape[0] * mask.shape[1]
    if total <= 0:
        return True
    return (changed / total) >= min_ratio


def build_face_app(det_size: tuple[int, int], max_faces: int, ctx_id: int) -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    app.det_model.nms_threshold = 0.4
    app.det_model.max_num = max_faces
    return app


def compute_target_embedding(app: FaceAnalysis, image_paths: list[Path]) -> np.ndarray:
    embeddings = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        faces = app.get(image)
        if not faces:
            raise ValueError(f"No face found in image: {image_path}")
        embeddings.append(faces[0].embedding)
    return normalize_embedding(np.mean(np.stack(embeddings, axis=0), axis=0))


def normalize_target_embedding(embedding: np.ndarray, aggregation: str) -> np.ndarray:
    if embedding.ndim == 1:
        return normalize_embedding(embedding)
    if embedding.ndim == 2:
        if aggregation == "first":
            return normalize_embedding(embedding[0])
        return normalize_embedding(np.mean(embedding, axis=0))
    raise ValueError(f"Unsupported embedding shape: {embedding.shape}")


def clamp_bbox(frame_shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_from_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    clamped = clamp_bbox(frame.shape, bbox)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    return frame[y1:y2, x1:x2].copy()


def match_video(
    app: FaceAnalysis,
    video_path: Path,
    target_embedding: np.ndarray,
    sample_fps: float,
    sample_every_n_frames: int | None,
    threshold: float,
    enable_motion_filter: bool,
    motion_threshold: int,
    motion_min_ratio: float,
    stop_event=None,
    on_face_crop=None,
) -> MatchResult:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "Failed to open video: "
            f"{video_path} (try enabling ffmpeg or check file integrity)"
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if sample_every_n_frames is not None:
        frame_interval = max(sample_every_n_frames, 1)
    else:
        frame_interval = max(int(round(fps / sample_fps)), 1) if fps > 0 else 1
    best_score = -1.0
    best_frame = None
    best_timestamp = None
    best_frame_index = None
    best_bbox = None
    prev_frame = None

    frame_index = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            break
        ret, frame = cap.read()
        if not ret:
            break
        if enable_motion_filter and prev_frame is not None:
            if not has_motion(prev_frame, frame, motion_threshold, motion_min_ratio):
                prev_frame = frame
                for _ in range(frame_interval - 1):
                    if not cap.grab():
                        cap.release()
                        return MatchResult(
                            False,
                            best_score,
                            best_frame,
                            best_timestamp,
                            best_frame_index,
                            best_bbox,
                        )
                    frame_index += 1
                frame_index += 1
                continue
        faces = app.get(frame)
        for face in faces:
            score = cosine_similarity(face.embedding, target_embedding)
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_timestamp = frame_index / fps if fps > 0 else None
                best_frame_index = frame_index
                bbox = face.bbox.astype(int)
                best_bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            if best_score >= threshold:
                if on_face_crop is not None and best_bbox is not None:
                    crop = crop_from_bbox(best_frame, best_bbox)
                    if crop is not None:
                        on_face_crop(crop, video_path, best_timestamp)
                cap.release()
                return MatchResult(
                    True,
                    best_score,
                    best_frame,
                    best_timestamp,
                    best_frame_index,
                    best_bbox,
                )
        prev_frame = frame
        for _ in range(frame_interval - 1):
            if not cap.grab():
                cap.release()
                return MatchResult(
                    False,
                    best_score,
                    best_frame,
                    best_timestamp,
                    best_frame_index,
                    best_bbox,
                )
            frame_index += 1
        frame_index += 1

    cap.release()
    return MatchResult(
        False,
        best_score,
        best_frame,
        best_timestamp,
        best_frame_index,
        best_bbox,
    )


def match_video_ffmpeg(
    app: FaceAnalysis,
    video_path: Path,
    target_embedding: np.ndarray,
    frame_interval: int,
    threshold: float,
    ffmpeg_path: str,
    ffmpeg_scale: tuple[int, int] | None,
    enable_motion_filter: bool,
    motion_threshold: int,
    motion_min_ratio: float,
    stop_event=None,
    on_face_crop=None,
) -> MatchResult:
    fps = get_video_fps(video_path)
    if ffmpeg_scale:
        width, height = ffmpeg_scale
    else:
        width, height = get_video_size(video_path)

    best_score = -1.0
    best_frame = None
    best_timestamp = None
    best_frame_index = None
    best_bbox = None
    prev_frame = None

    select_filter = f"select=not(mod(n\\,{frame_interval}))"
    if ffmpeg_scale:
        scale_filter = f"scale={ffmpeg_scale[0]}:{ffmpeg_scale[1]}"
        vf_filter = f"{select_filter},{scale_filter}"
    else:
        vf_filter = select_filter
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "cuda",
        "-i",
        str(video_path),
        "-vf",
        vf_filter,
        "-vsync",
        "vfr",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    frame_size = width * height * 3
    if frame_size <= 0:
        raise RuntimeError(f"Invalid frame size for {video_path}")

    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as process:
        if process.stdout is None:
            raise RuntimeError("Failed to capture ffmpeg output.")
        index = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                process.kill()
                break
            data = process.stdout.read(frame_size)
            if not data:
                break
            frame = np.frombuffer(data, np.uint8).reshape((height, width, 3))
            if enable_motion_filter and prev_frame is not None:
                if not has_motion(prev_frame, frame, motion_threshold, motion_min_ratio):
                    prev_frame = frame
                    index += 1
                    continue
            faces = app.get(frame)
            frame_index = index * frame_interval
            timestamp = frame_index / fps if fps > 0 else None
            for face in faces:
                score = cosine_similarity(face.embedding, target_embedding)
                if score > best_score:
                    best_score = score
                    best_frame = frame.copy()
                    best_timestamp = timestamp
                    best_frame_index = frame_index
                    bbox = face.bbox.astype(int)
                    best_bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            if best_score >= threshold:
                if on_face_crop is not None and best_bbox is not None:
                    crop = crop_from_bbox(best_frame, best_bbox)
                    if crop is not None:
                        on_face_crop(crop, video_path, best_timestamp)
                process.kill()
                return MatchResult(
                    True,
                    best_score,
                    best_frame,
                    best_timestamp,
                    best_frame_index,
                    best_bbox,
                )
            prev_frame = frame
            index += 1

    return MatchResult(
        False,
        best_score,
        best_frame,
        best_timestamp,
        best_frame_index,
        best_bbox,
    )


def save_match_snapshot(
    confirm_dir: Path,
    video_path: Path,
    frame: np.ndarray,
    timestamp_s: float | None,
    frame_index: int | None,
    score: float,
    bbox: tuple[int, int, int, int] | None,
) -> Path:
    suffix = "match"
    if timestamp_s is not None:
        suffix = f"match_{timestamp_s:.2f}s"
    if frame_index is not None:
        suffix = f"{suffix}_frame{frame_index}"
    snapshot_name = f"{video_path.stem}_{suffix}_{score:.3f}.jpg"
    snapshot_path = confirm_dir / snapshot_name
    output = frame.copy()
    if bbox is not None:
        clamped = clamp_bbox(frame.shape, bbox)
        if clamped is not None:
            x1, y1, x2, y2 = clamped
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Keep bbox visualization for confirmation snapshots.
    cv2.imwrite(str(snapshot_path), output)
    return snapshot_path


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def iter_videos(
    video_dir: Path,
    include_subdirs: bool = False,
    exclude_dir: Path | None = None,
) -> list[Path]:
    candidates = video_dir.rglob("*") if include_subdirs else video_dir.iterdir()
    videos = []
    exclude_dir = exclude_dir.resolve() if exclude_dir is not None else None
    for path in candidates:
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if exclude_dir is not None and _is_relative_to(path.resolve(), exclude_dir):
            continue
        videos.append(path)
    return sorted(videos)


def get_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    cap.release()
    return fps


def get_video_size(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return width, height


def format_duration(seconds: float) -> str:
    seconds_int = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_time(value: str) -> dt.time:
    try:
        hour_str, minute_str = value.split(":")
        return dt.time(int(hour_str), int(minute_str))
    except ValueError as exc:
        raise ValueError(f"Invalid time format: {value} (expected HH:MM)") from exc


def is_time_in_window(current: dt.time, start: dt.time, end: dt.time) -> bool:
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end


def is_motion_filter_enabled(
    video_path: Path,
    args: argparse.Namespace,
    window_start: Optional[dt.time],
    window_end: Optional[dt.time],
) -> bool:
    if args.motion_filter_disabled or args.disable_motion_filter:
        return False
    if window_start is None or window_end is None:
        return True
    mtime = dt.datetime.fromtimestamp(video_path.stat().st_mtime).time()
    return is_time_in_window(mtime, window_start, window_end)


def parse_scale(value: str) -> Optional[tuple[int, int]]:
    if not value:
        return None
    parts = value.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise ValueError("缩放格式应为 例如 1280x720")
    return int(parts[0]), int(parts[1])


def run_gui() -> None:
    import importlib.util

    from PySide6 import QtCore, QtGui, QtWidgets

    translations = {
        "zh": {
            "language": "语言",
            "window_title": "人脸视频检索",
            "path_group": "路径设置",
            "video_dir": "视频目录",
            "confirm_dir": "确认目录",
            "embedding_file": "特征文件",
            "images_optional": "目标图片",
            "include_subdirs": "包含子文件夹视频",
            "browse": "浏览…",
            "param_group": "检测参数",
            "processes": "进程数",
            "every_n_frames": "每 N 帧检测",
            "threshold": "相似度阈值",
            "threshold_tip": "相似度阈值范围 0~1，数值越大越严格。",
            "face_group": "人脸预览",
            "advanced_toggle": "高级设置",
            "stop": "停止",
            "start": "开始",
            "cpu_label": "CPU",
            "gpu_label": "GPU",
            "enable_motion": "启用运动筛选",
            "motion_settings": "设置…",
            "enable_ffmpeg": "启用 ffmpeg",
            "ffmpeg_settings": "设置…",
            "use_fps": "按 FPS 采样",
            "cpu_only": "仅用 CPU 推理",
            "max_faces": "最大人脸数",
            "motion_dialog_title": "运动筛选配置",
            "motion_start": "时间窗开始",
            "motion_end": "时间窗结束",
            "motion_threshold": "运动阈值",
            "motion_ratio": "变化比例",
            "ffmpeg_dialog_title": "ffmpeg 配置",
            "ffmpeg_path": "ffmpeg 路径",
            "ffmpeg_scale": "缩放(可选)",
            "scale_placeholder": "1280x720",
            "select_all": "全选",
            "copy": "复制",
            "clear": "清空",
            "scroll_here": "滚动到此处",
            "top": "顶部",
            "bottom": "底部",
            "page_up": "上一页",
            "page_down": "下一页",
            "line_up": "向上滚动",
            "line_down": "向下滚动",
            "open_video": "打开视频",
            "no_match": "暂无匹配视频",
            "open_failed": "无法打开视频",
            "open_failed_detail": "无法使用 VLC/ffplay 打开视频，或文件不存在。",
            "select_dir": "选择文件夹",
            "select_embedding": "选择 embedding 文件",
            "select_images": "选择目标图片",
            "error_title": "错误",
            "select_video_dir": "请先选择视频目录。",
            "run_failed": "运行失败",
            "stop_requested": "已请求停止，等待当前任务结束...",
            "na": "不可用",
        },
        "en": {
            "language": "Language",
            "window_title": "Face Video Search",
            "path_group": "Paths",
            "video_dir": "Video Directory",
            "confirm_dir": "Confirm Directory",
            "embedding_file": "Feature File",
            "images_optional": "Target Images",
            "include_subdirs": "Include Subfolder Videos",
            "browse": "Browse…",
            "param_group": "Detection",
            "processes": "Processes",
            "every_n_frames": "Every N Frames",
            "threshold": "Similarity Threshold",
            "threshold_tip": "Threshold range 0~1. Higher is stricter.",
            "face_group": "Face Preview",
            "advanced_toggle": "Advanced",
            "stop": "Stop",
            "start": "Start",
            "cpu_label": "CPU",
            "gpu_label": "GPU",
            "enable_motion": "Enable Motion Filter",
            "motion_settings": "Settings…",
            "enable_ffmpeg": "Enable ffmpeg",
            "ffmpeg_settings": "Settings…",
            "use_fps": "Sample by FPS",
            "cpu_only": "CPU Only",
            "max_faces": "Max Faces",
            "motion_dialog_title": "Motion Filter Settings",
            "motion_start": "Window Start",
            "motion_end": "Window End",
            "motion_threshold": "Motion Threshold",
            "motion_ratio": "Change Ratio",
            "ffmpeg_dialog_title": "ffmpeg Settings",
            "ffmpeg_path": "ffmpeg Path",
            "ffmpeg_scale": "Scale (Optional)",
            "scale_placeholder": "1280x720",
            "select_all": "Select All",
            "copy": "Copy",
            "clear": "Clear",
            "scroll_here": "Scroll Here",
            "top": "Top",
            "bottom": "Bottom",
            "page_up": "Page Up",
            "page_down": "Page Down",
            "line_up": "Scroll Up",
            "line_down": "Scroll Down",
            "open_video": "Open Video",
            "no_match": "No matched video",
            "open_failed": "Unable to open video",
            "open_failed_detail": "Unable to open the video with VLC/ffplay, or file missing.",
            "select_dir": "Select Folder",
            "select_embedding": "Select embedding file",
            "select_images": "Select target images",
            "error_title": "Error",
            "select_video_dir": "Please select a video directory first.",
            "run_failed": "Run Failed",
            "stop_requested": "Stop requested, waiting for tasks to finish...",
            "na": "N/A",
        },
    }
    current_lang = "zh"

    def t(key: str) -> str:
        return translations[current_lang].get(key, key)

    app = QtWidgets.QApplication([])
    icon_path = Path(__file__).resolve().parent / "ICO" / "app_icon.ico"
    if icon_path.exists():
        app_icon = QtGui.QIcon(str(icon_path))
        app.setWindowIcon(app_icon)
    window = QtWidgets.QWidget()
    if icon_path.exists():
        window.setWindowIcon(app_icon)
    window.setWindowTitle(t("window_title"))
    main_layout = QtWidgets.QVBoxLayout(window)
    main_layout.setSpacing(10)
    main_layout.setContentsMargins(12, 12, 12, 12)

    def create_label(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setFixedWidth(96)
        return label

    def create_browse_button(text: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setFixedWidth(72)
        return button

    path_group = QtWidgets.QGroupBox()
    path_layout = QtWidgets.QGridLayout(path_group)
    path_layout.setHorizontalSpacing(8)
    path_layout.setVerticalSpacing(8)

    video_dir_edit = QtWidgets.QLineEdit()
    confirm_dir_edit = QtWidgets.QLineEdit()
    confirm_dir_edit.setPlaceholderText("confirm")
    embedding_edit = QtWidgets.QLineEdit()
    images_edit = QtWidgets.QLineEdit()

    video_dir_button = create_browse_button("…")
    confirm_dir_button = create_browse_button("…")
    embedding_button = create_browse_button("…")
    images_button = create_browse_button("…")

    video_dir_label = create_label("")
    confirm_dir_label = create_label("")
    embedding_label = create_label("")
    images_label = create_label("")

    path_layout.addWidget(video_dir_label, 0, 0)
    path_layout.addWidget(video_dir_edit, 0, 1)
    path_layout.addWidget(video_dir_button, 0, 2)

    path_layout.addWidget(confirm_dir_label, 1, 0)
    path_layout.addWidget(confirm_dir_edit, 1, 1)
    path_layout.addWidget(confirm_dir_button, 1, 2)

    path_layout.addWidget(embedding_label, 2, 0)
    path_layout.addWidget(embedding_edit, 2, 1)
    path_layout.addWidget(embedding_button, 2, 2)

    path_layout.addWidget(images_label, 3, 0)
    path_layout.addWidget(images_edit, 3, 1)
    path_layout.addWidget(images_button, 3, 2)

    param_group = QtWidgets.QGroupBox()
    param_layout = QtWidgets.QGridLayout(param_group)
    param_layout.setHorizontalSpacing(12)
    param_layout.setVerticalSpacing(8)

    workers_spin = QtWidgets.QSpinBox()
    workers_spin.setMinimum(1)
    workers_spin.setValue(2)
    sample_frames_spin = QtWidgets.QSpinBox()
    sample_frames_spin.setMinimum(1)
    sample_frames_spin.setValue(6)
    threshold_spin = QtWidgets.QDoubleSpinBox()
    threshold_spin.setRange(0.0, 1.0)
    threshold_spin.setSingleStep(0.01)
    threshold_spin.setDecimals(2)
    threshold_spin.setValue(0.6)
    threshold_tip = QtWidgets.QToolButton()
    threshold_tip.setText("?")

    left_param_layout = QtWidgets.QGridLayout()
    left_param_layout.setHorizontalSpacing(8)
    left_param_layout.setVerticalSpacing(8)
    processes_label = create_label("")
    every_n_frames_label = create_label("")
    threshold_label = create_label("")
    left_param_layout.addWidget(processes_label, 0, 0)
    left_param_layout.addWidget(workers_spin, 0, 1)
    left_param_layout.addWidget(every_n_frames_label, 1, 0)
    left_param_layout.addWidget(sample_frames_spin, 1, 1)

    threshold_row = QtWidgets.QHBoxLayout()
    threshold_row.addWidget(threshold_spin)
    threshold_row.addWidget(threshold_tip)
    threshold_row.addStretch()
    left_param_layout.addWidget(threshold_label, 2, 0)
    left_param_layout.addLayout(threshold_row, 2, 1)

    face_group = QtWidgets.QGroupBox()
    face_layout = QtWidgets.QVBoxLayout(face_group)
    face_layout.setContentsMargins(8, 8, 8, 8)
    face_preview = QtWidgets.QLabel()
    face_preview.setFixedSize(200, 160)
    face_preview.setAlignment(QtCore.Qt.AlignCenter)
    face_preview.setScaledContents(False)
    face_preview.setStyleSheet("border: 1px solid #cfcfcf;")
    face_layout.addWidget(face_preview)
    face_preview.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

    last_match = {"video_path": None, "timestamp_s": None}

    def open_video_at_timestamp(video_path: str, timestamp_s: float | None) -> bool:
        if not video_path:
            return False
        path = Path(video_path)
        if not path.exists():
            return False
        start_time = max(timestamp_s or 0.0, 0.0)
        vlc_path = shutil.which("vlc")
        if vlc_path:
            subprocess.Popen(
                [
                    vlc_path,
                    "--play-and-exit",
                    "--start-time",
                    f"{start_time:.2f}",
                    "--width",
                    "1920",
                    "--height",
                    "1080",
                    str(path),
                ]
            )
            return True
        ffplay_path = shutil.which("ffplay")
        if ffplay_path:
            subprocess.Popen(
                [
                    ffplay_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{start_time:.2f}",
                    "-i",
                    str(path),
                    "-x",
                    "1920",
                    "-y",
                    "1080",
                ]
            )
            return True
        return False

    def open_matched_video() -> None:
        video_path = last_match.get("video_path")
        timestamp_s = last_match.get("timestamp_s")
        if not video_path:
            QtWidgets.QMessageBox.information(window, t("open_video"), t("no_match"))
            return
        if not open_video_at_timestamp(video_path, timestamp_s):
            QtWidgets.QMessageBox.warning(
                window, t("open_failed"), t("open_failed_detail")
            )

    def open_face_menu(position: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(face_preview)
        menu.setStyleSheet(
            "QMenu { padding: 2px; }"
            "QMenu::item { padding: 2px 8px; }"
            "QMenu::icon { width: 0px; }"
        )
        open_action = menu.addAction(t("open_video"))
        open_action.setEnabled(bool(last_match.get("video_path")))
        open_action.triggered.connect(open_matched_video)
        menu.exec(face_preview.mapToGlobal(position))

    face_preview.customContextMenuRequested.connect(open_face_menu)

    param_layout.addLayout(left_param_layout, 0, 0, 3, 2)

    advanced_toggle = QtWidgets.QToolButton()
    advanced_toggle.setText("")
    advanced_toggle.setCheckable(True)
    advanced_toggle.setChecked(False)
    advanced_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
    advanced_toggle.setArrowType(QtCore.Qt.RightArrow)

    header_layout = QtWidgets.QHBoxLayout()
    header_layout.addWidget(advanced_toggle)
    header_layout.addStretch()
    stop_button = QtWidgets.QPushButton("")
    start_button = QtWidgets.QPushButton("")
    start_button.setDefault(True)
    stop_button.setEnabled(False)
    cpu_label = QtWidgets.QLabel("")
    gpu_label = QtWidgets.QLabel("")
    header_layout.addWidget(cpu_label)
    header_layout.addWidget(gpu_label)
    header_layout.addWidget(stop_button)
    header_layout.addWidget(start_button)

    language_widget = QtWidgets.QWidget()
    language_layout = QtWidgets.QHBoxLayout(language_widget)
    language_layout.setContentsMargins(0, 0, 0, 0)
    language_layout.setSpacing(0)
    language_cn = QtWidgets.QToolButton()
    language_cn.setText("CN")
    language_en = QtWidgets.QToolButton()
    language_en.setText("EN")
    language_separator = QtWidgets.QLabel("|")
    language_cn.setAutoRaise(True)
    language_en.setAutoRaise(True)
    language_cn.setStyleSheet("QToolButton { background: transparent; border: none; }")
    language_en.setStyleSheet("QToolButton { background: transparent; border: none; }")
    language_cn.setCursor(QtCore.Qt.PointingHandCursor)
    language_en.setCursor(QtCore.Qt.PointingHandCursor)
    language_cn.setFocusPolicy(QtCore.Qt.NoFocus)
    language_en.setFocusPolicy(QtCore.Qt.NoFocus)
    cpu_font = cpu_label.font()
    language_cn.setFont(cpu_font)
    language_en.setFont(cpu_font)
    language_separator.setFont(cpu_font)
    language_layout.addWidget(language_cn)
    language_layout.addWidget(language_separator)
    language_layout.addWidget(language_en)
    header_layout.insertWidget(1, language_widget)

    advanced_container = QtWidgets.QWidget()
    advanced_layout = QtWidgets.QGridLayout(advanced_container)
    advanced_layout.setHorizontalSpacing(8)
    advanced_layout.setVerticalSpacing(8)
    advanced_container.setSizePolicy(
        QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
    )

    enable_motion_check = QtWidgets.QCheckBox()
    motion_settings_button = create_browse_button("…")
    enable_ffmpeg_check = QtWidgets.QCheckBox()
    ffmpeg_settings_button = create_browse_button("…")
    use_fps_check = QtWidgets.QCheckBox()
    sample_fps_spin = QtWidgets.QDoubleSpinBox()
    sample_fps_spin.setRange(0.1, 120.0)
    sample_fps_spin.setDecimals(2)
    sample_fps_spin.setSingleStep(0.1)
    sample_fps_spin.setValue(1.0)
    cpu_only_check = QtWidgets.QCheckBox()
    max_face_spin = QtWidgets.QSpinBox()
    max_face_spin.setRange(1, 50)
    max_face_spin.setValue(5)

    advanced_layout.addWidget(enable_motion_check, 0, 0)
    advanced_layout.addWidget(motion_settings_button, 0, 1, alignment=QtCore.Qt.AlignLeft)
    advanced_layout.addWidget(enable_ffmpeg_check, 1, 0)
    advanced_layout.addWidget(ffmpeg_settings_button, 1, 1, alignment=QtCore.Qt.AlignLeft)
    advanced_layout.addWidget(use_fps_check, 2, 0)
    advanced_layout.addWidget(sample_fps_spin, 2, 1, alignment=QtCore.Qt.AlignLeft)
    advanced_layout.addWidget(cpu_only_check, 3, 0)
    max_faces_label = create_label("")
    advanced_layout.addWidget(max_faces_label, 4, 0)
    advanced_layout.addWidget(max_face_spin, 4, 1, alignment=QtCore.Qt.AlignLeft)
    include_subdirs_check = QtWidgets.QCheckBox()
    advanced_layout.addWidget(include_subdirs_check, 5, 0, 1, 2)

    advanced_container.setVisible(False)
    advanced_container.setMaximumHeight(0)

    def toggle_advanced(checked: bool) -> None:
        advanced_container.setVisible(checked)
        if checked:
            advanced_container.setMaximumHeight(16777215)
        else:
            advanced_container.setMaximumHeight(0)
        advanced_toggle.setArrowType(
            QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
        )
        main_layout.activate()
        window.adjustSize()

    advanced_toggle.toggled.connect(toggle_advanced)

    def apply_language(lang: str) -> None:
        nonlocal current_lang
        current_lang = lang
        language_cn.setFont(cpu_font)
        language_en.setFont(cpu_font)
        if current_lang == "zh":
            bold_font = QtGui.QFont(cpu_font)
            bold_font.setBold(True)
            language_cn.setFont(bold_font)
        else:
            bold_font = QtGui.QFont(cpu_font)
            bold_font.setBold(True)
            language_en.setFont(bold_font)
        window.setWindowTitle(t("window_title"))
        path_group.setTitle(t("path_group"))
        param_group.setTitle(t("param_group"))
        face_group.setTitle(t("face_group"))
        advanced_toggle.setText(t("advanced_toggle"))
        video_dir_label.setText(t("video_dir"))
        confirm_dir_label.setText(t("confirm_dir"))
        embedding_label.setText(t("embedding_file"))
        images_label.setText(t("images_optional"))
        video_dir_button.setText(t("browse"))
        confirm_dir_button.setText(t("browse"))
        embedding_button.setText(t("browse"))
        images_button.setText(t("browse"))
        processes_label.setText(t("processes"))
        every_n_frames_label.setText(t("every_n_frames"))
        threshold_label.setText(t("threshold"))
        threshold_tip.setToolTip(t("threshold_tip"))
        stop_button.setText(t("stop"))
        start_button.setText(t("start"))
        enable_motion_check.setText(t("enable_motion"))
        motion_settings_button.setText(t("motion_settings"))
        enable_ffmpeg_check.setText(t("enable_ffmpeg"))
        ffmpeg_settings_button.setText(t("ffmpeg_settings"))
        use_fps_check.setText(t("use_fps"))
        cpu_only_check.setText(t("cpu_only"))
        max_faces_label.setText(t("max_faces"))
        include_subdirs_check.setText(t("include_subdirs"))

    def on_language_cn() -> None:
        apply_language("zh")
        update_usage()

    def on_language_en() -> None:
        apply_language("en")
        update_usage()

    language_cn.clicked.connect(on_language_cn)
    language_en.clicked.connect(on_language_en)
    apply_language(current_lang)

    motion_window_start = ""
    motion_window_end = ""
    motion_threshold = 25
    motion_ratio = 0.003
    ffmpeg_path = "ffmpeg"
    ffmpeg_scale = ""

    def open_motion_settings() -> None:
        dialog = QtWidgets.QDialog(window)
        dialog.setWindowTitle(t("motion_dialog_title"))
        layout = QtWidgets.QFormLayout(dialog)
        start_edit = QtWidgets.QLineEdit(motion_window_start)
        end_edit = QtWidgets.QLineEdit(motion_window_end)
        threshold_edit = QtWidgets.QSpinBox()
        threshold_edit.setRange(0, 255)
        threshold_edit.setValue(motion_threshold)
        ratio_edit = QtWidgets.QDoubleSpinBox()
        ratio_edit.setRange(0.0, 1.0)
        ratio_edit.setDecimals(3)
        ratio_edit.setSingleStep(0.001)
        ratio_edit.setValue(motion_ratio)
        layout.addRow(t("motion_start"), start_edit)
        layout.addRow(t("motion_end"), end_edit)
        layout.addRow(t("motion_threshold"), threshold_edit)
        layout.addRow(t("motion_ratio"), ratio_edit)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        layout.addRow(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            nonlocal_motion_window_start = start_edit.text().strip()
            nonlocal_motion_window_end = end_edit.text().strip()
            nonlocal_motion_threshold = threshold_edit.value()
            nonlocal_motion_ratio = ratio_edit.value()
            nonlocal_vars["motion_window_start"] = nonlocal_motion_window_start
            nonlocal_vars["motion_window_end"] = nonlocal_motion_window_end
            nonlocal_vars["motion_threshold"] = nonlocal_motion_threshold
            nonlocal_vars["motion_ratio"] = nonlocal_motion_ratio

    def open_ffmpeg_settings() -> None:
        dialog = QtWidgets.QDialog(window)
        dialog.setWindowTitle(t("ffmpeg_dialog_title"))
        layout = QtWidgets.QFormLayout(dialog)
        path_edit = QtWidgets.QLineEdit(ffmpeg_path)
        scale_edit = QtWidgets.QLineEdit(ffmpeg_scale)
        scale_edit.setPlaceholderText(t("scale_placeholder"))
        layout.addRow(t("ffmpeg_path"), path_edit)
        layout.addRow(t("ffmpeg_scale"), scale_edit)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        layout.addRow(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            nonlocal_vars["ffmpeg_path"] = path_edit.text().strip() or "ffmpeg"
            nonlocal_vars["ffmpeg_scale"] = scale_edit.text().strip()

    nonlocal_vars = {
        "motion_window_start": motion_window_start,
        "motion_window_end": motion_window_end,
        "motion_threshold": motion_threshold,
        "motion_ratio": motion_ratio,
        "ffmpeg_path": ffmpeg_path,
        "ffmpeg_scale": ffmpeg_scale,
    }

    motion_settings_button.clicked.connect(open_motion_settings)
    ffmpeg_settings_button.clicked.connect(open_ffmpeg_settings)
    enable_motion_check.setChecked(False)
    enable_ffmpeg_check.setChecked(False)
    motion_settings_button.setEnabled(False)
    ffmpeg_settings_button.setEnabled(False)

    def update_motion_enabled(checked: bool) -> None:
        motion_settings_button.setEnabled(checked)

    def update_ffmpeg_enabled(checked: bool) -> None:
        ffmpeg_settings_button.setEnabled(checked)

    enable_motion_check.toggled.connect(update_motion_enabled)
    enable_ffmpeg_check.toggled.connect(update_ffmpeg_enabled)
    update_motion_enabled(enable_motion_check.isChecked())
    update_ffmpeg_enabled(enable_ffmpeg_check.isChecked())
    use_fps_check.setChecked(False)
    sample_fps_spin.setEnabled(False)

    def update_fps_enabled() -> None:
        enabled = use_fps_check.isChecked() and controls_enabled
        sample_fps_spin.setEnabled(enabled)
        sample_frames_spin.setEnabled(controls_enabled and not use_fps_check.isChecked())

    use_fps_check.stateChanged.connect(lambda _state: update_fps_enabled())

    content_layout = QtWidgets.QHBoxLayout()
    content_layout.addWidget(param_group, 2)
    content_layout.addWidget(face_group, 1)

    main_layout.addWidget(path_group)
    main_layout.addLayout(content_layout)
    main_layout.addLayout(header_layout)
    main_layout.addWidget(advanced_container)

    log_output = QtWidgets.QPlainTextEdit()
    log_output.setReadOnly(True)
    log_output.setVisible(False)
    log_output.setMinimumHeight(140)
    log_output.setStyleSheet("background-color: white;")
    log_output.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
    log_scrollbar = log_output.verticalScrollBar()
    log_scrollbar.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

    def open_log_menu(position: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(log_output)
        select_action = menu.addAction(t("select_all"))
        copy_action = menu.addAction(t("copy"))
        clear_action = menu.addAction(t("clear"))
        select_action.triggered.connect(log_output.selectAll)
        copy_action.triggered.connect(log_output.copy)
        clear_action.triggered.connect(log_output.clear)
        menu.exec(log_output.mapToGlobal(position))

    log_output.customContextMenuRequested.connect(open_log_menu)

    def open_scrollbar_menu(position: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(log_output)
        scroll_here_action = menu.addAction(t("scroll_here"))
        menu.addSeparator()
        top_action = menu.addAction(t("top"))
        bottom_action = menu.addAction(t("bottom"))
        menu.addSeparator()
        page_up_action = menu.addAction(t("page_up"))
        page_down_action = menu.addAction(t("page_down"))
        menu.addSeparator()
        line_up_action = menu.addAction(t("line_up"))
        line_down_action = menu.addAction(t("line_down"))

        def set_scroll_here() -> None:
            height = max(log_scrollbar.height(), 1)
            ratio = position.y() / height
            ratio = max(0.0, min(ratio, 1.0))
            minimum = log_scrollbar.minimum()
            maximum = log_scrollbar.maximum()
            value = int(minimum + (maximum - minimum) * ratio)
            log_scrollbar.setValue(value)

        scroll_here_action.triggered.connect(set_scroll_here)
        top_action.triggered.connect(lambda: log_scrollbar.setValue(log_scrollbar.minimum()))
        bottom_action.triggered.connect(lambda: log_scrollbar.setValue(log_scrollbar.maximum()))
        page_up_action.triggered.connect(
            lambda: log_scrollbar.setValue(log_scrollbar.value() - log_scrollbar.pageStep())
        )
        page_down_action.triggered.connect(
            lambda: log_scrollbar.setValue(log_scrollbar.value() + log_scrollbar.pageStep())
        )
        line_up_action.triggered.connect(
            lambda: log_scrollbar.setValue(log_scrollbar.value() - log_scrollbar.singleStep())
        )
        line_down_action.triggered.connect(
            lambda: log_scrollbar.setValue(log_scrollbar.value() + log_scrollbar.singleStep())
        )
        menu.exec(log_scrollbar.mapToGlobal(position))

    log_scrollbar.customContextMenuRequested.connect(open_scrollbar_menu)
    main_layout.addWidget(log_output)

    def select_dir(target: QtWidgets.QLineEdit) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(window, t("select_dir"))
        if selected:
            target.setText(selected)

    def select_file(target: QtWidgets.QLineEdit, title: str, filter_text: str) -> None:
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            window, title, "", filter_text
        )
        if selected:
            target.setText(selected)

    def select_files(target: QtWidgets.QLineEdit, title: str, filter_text: str) -> None:
        selected, _ = QtWidgets.QFileDialog.getOpenFileNames(
            window, title, "", filter_text
        )
        if selected:
            target.setText(";".join(selected))

    video_dir_button.clicked.connect(lambda: select_dir(video_dir_edit))
    confirm_dir_button.clicked.connect(lambda: select_dir(confirm_dir_edit))
    embedding_button.clicked.connect(
        lambda: select_file(embedding_edit, t("select_embedding"), "NPY Files (*.npy)")
    )
    images_button.clicked.connect(
        lambda: select_files(images_edit, t("select_images"), "Images (*.jpg *.png *.jpeg)")
    )

    controls_enabled = True

    def update_target_inputs() -> None:
        if not controls_enabled:
            return
        has_embedding = bool(embedding_edit.text().strip())
        has_images = bool(images_edit.text().strip())
        embedding_active = not has_images
        images_active = not has_embedding
        embedding_edit.setEnabled(embedding_active)
        embedding_button.setEnabled(embedding_active)
        images_edit.setEnabled(images_active)
        images_button.setEnabled(images_active)

    def set_controls_enabled(enabled: bool) -> None:
        nonlocal controls_enabled
        controls_enabled = enabled
        video_dir_edit.setEnabled(enabled)
        confirm_dir_edit.setEnabled(enabled)
        video_dir_button.setEnabled(enabled)
        confirm_dir_button.setEnabled(enabled)
        embedding_edit.setEnabled(enabled)
        images_edit.setEnabled(enabled)
        embedding_button.setEnabled(enabled)
        images_button.setEnabled(enabled)
        workers_spin.setEnabled(enabled)
        threshold_spin.setEnabled(enabled)
        enable_motion_check.setEnabled(enabled)
        enable_ffmpeg_check.setEnabled(enabled)
        use_fps_check.setEnabled(enabled)
        cpu_only_check.setEnabled(enabled)
        max_face_spin.setEnabled(enabled)
        include_subdirs_check.setEnabled(enabled)
        update_fps_enabled()
        motion_settings_button.setEnabled(enabled and enable_motion_check.isChecked())
        ffmpeg_settings_button.setEnabled(enabled and enable_ffmpeg_check.isChecked())
        update_target_inputs()

    embedding_edit.textChanged.connect(lambda _text: update_target_inputs())
    images_edit.textChanged.connect(lambda _text: update_target_inputs())
    update_target_inputs()

    class Runner(QtCore.QObject):
        finished = QtCore.Signal()
        failed = QtCore.Signal(str)
        face_crop = QtCore.Signal(object)

        def __init__(self, args: argparse.Namespace) -> None:
            super().__init__()
            self.args = args

        @QtCore.Slot()
        def run(self) -> None:
            try:
                import multiprocessing as mp

                ctx = mp.get_context("spawn")
                error_queue = ctx.Queue()
                process = ctx.Process(
                    target=run_with_args_safe,
                    args=(self.args, error_queue),
                )
                process.start()
                process.join()
                try:
                    error_message = error_queue.get_nowait()
                except Exception:
                    error_message = None
                if error_message:
                    self.failed.emit(error_message)
            except Exception as exc:
                self.failed.emit(str(exc))
            finally:
                self.finished.emit()

    def build_args() -> argparse.Namespace:
        video_dir = video_dir_edit.text().strip()
        confirm_dir = confirm_dir_edit.text().strip() or "confirm"
        target_images = images_edit.text().strip()
        embedding = embedding_edit.text().strip()
        args = argparse.Namespace(
            video_dir=video_dir,
            confirm_dir=confirm_dir,
            processed_list="processed_videos.txt",
            timing_log="processing_times.txt",
            debug_log="processing_debug.txt",
            include_subdirs=include_subdirs_check.isChecked(),
            target_images=target_images.split(";") if target_images else None,
            target_embedding=embedding or None,
            embedding_aggregation="mean",
            sample_fps=float(sample_fps_spin.value()),
            sample_every_n_frames=(
                None if use_fps_check.isChecked() else int(sample_frames_spin.value())
            ),
            threshold=float(threshold_spin.value()),
            max_face_detect=int(max_face_spin.value()),
            det_size=[640, 640],
            use_ffmpeg=enable_ffmpeg_check.isChecked(),
            ffmpeg_path=nonlocal_vars["ffmpeg_path"],
            ffmpeg_scale=None,
            cpu_only=cpu_only_check.isChecked(),
            motion_threshold=int(nonlocal_vars["motion_threshold"]),
            motion_min_ratio=float(nonlocal_vars["motion_ratio"]),
            disable_motion_filter=False,
            motion_filter_disabled=not enable_motion_check.isChecked(),
            motion_filter_window=None,
            workers=int(workers_spin.value()),
            gui=True,
            stop_event=None,
        )
        if nonlocal_vars["motion_window_start"] and nonlocal_vars["motion_window_end"]:
            args.motion_filter_window = [
                nonlocal_vars["motion_window_start"],
                nonlocal_vars["motion_window_end"],
            ]
        if nonlocal_vars["ffmpeg_scale"]:
            args.ffmpeg_scale = list(parse_scale(nonlocal_vars["ffmpeg_scale"]))
        return args

    thread_ref = {"thread": None, "worker": None, "timer": None, "stop_event": None}
    cpu_usage_provider = None
    gpu_usage_provider = None

    if importlib.util.find_spec("psutil") is not None:
        import psutil

        cpu_usage_provider = lambda: psutil.cpu_percent(interval=None)

    if importlib.util.find_spec("nvidia_smi") is not None:
        import nvidia_smi

        def build_gpu_provider() -> callable:
            try:
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                return lambda: None

            def read_gpu() -> float | None:
                try:
                    utilization = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                except Exception:
                    return None
                return float(utilization.gpu)

            return read_gpu

        gpu_usage_provider = build_gpu_provider()
    elif importlib.util.find_spec("pynvml") is not None:
        import pynvml

        def build_gpu_provider() -> callable:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                return lambda: None

            def read_gpu() -> float | None:
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                except Exception:
                    return None
                return float(utilization.gpu)

            return read_gpu

        gpu_usage_provider = build_gpu_provider()

    def start_run() -> None:
        if not video_dir_edit.text().strip():
            QtWidgets.QMessageBox.warning(window, t("error_title"), t("select_video_dir"))
            return

        log_output.clear()
        log_output.setVisible(True)
        face_preview.clear()
        start_button.setEnabled(False)
        stop_button.setEnabled(True)
        set_controls_enabled(False)
        args = build_args()
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        stop_event = ctx.Event()
        preview_queue = ctx.Queue()
        args.stop_event = stop_event
        thread_ref["stop_event"] = stop_event
        thread_ref["preview_queue"] = preview_queue
        args.preview_queue = preview_queue
        video_root = Path(args.video_dir)
        log_path = video_root / args.timing_log
        log_state = {"offset": 0}
        if log_path.exists():
            log_state["offset"] = log_path.stat().st_size
        timer = QtCore.QTimer(window)
        preview_timer = QtCore.QTimer(window)

        ui_handler = None

        def flush_log() -> None:
            if not log_path.exists():
                return
            with log_path.open("r", encoding="utf-8") as handle:
                handle.seek(log_state["offset"])
                new_data = handle.read()
                if new_data:
                    for line in new_data.splitlines():
                        if line:
                            log_output.appendPlainText(line)
                log_state["offset"] = handle.tell()

        def poll_log() -> None:
            if not log_path.exists():
                return
            with log_path.open("r", encoding="utf-8") as handle:
                handle.seek(log_state["offset"])
                new_data = handle.read()
                if new_data:
                    for line in new_data.splitlines():
                        if line:
                            log_output.appendPlainText(line)
                            if line.startswith("[批次结束]") and ui_handler is not None:
                                QtCore.QTimer.singleShot(0, ui_handler.on_finish)
                log_state["offset"] = handle.tell()

        timer.timeout.connect(poll_log)

        def poll_preview() -> None:
            preview_queue = thread_ref.get("preview_queue")
            if preview_queue is None:
                return
            while True:
                try:
                    crop = preview_queue.get_nowait()
                except queue.Empty:
                    break
                except Exception:
                    return
                if crop is None:
                    continue
                if isinstance(crop, dict):
                    item = crop
                    crop = item.get("crop")
                    last_match["video_path"] = item.get("video_path")
                    last_match["timestamp_s"] = item.get("timestamp_s")
                if crop is None:
                    continue
                height, width = crop.shape[:2]
                bytes_per_line = width * 3
                image = QtGui.QImage(
                    crop.data,
                    width,
                    height,
                    bytes_per_line,
                    QtGui.QImage.Format_BGR888,
                )
                pixmap = QtGui.QPixmap.fromImage(image)
                face_preview.setPixmap(
                    pixmap.scaled(
                        face_preview.size(),
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation,
                    )
                )

        preview_timer.timeout.connect(poll_preview)
        thread = QtCore.QThread()
        worker = Runner(args)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        class UiHandler(QtCore.QObject):
            def __init__(self) -> None:
                super().__init__()
                self._done = False

            @QtCore.Slot()
            def on_finish(self) -> None:
                if self._done:
                    return
                self._done = True
                flush_log()
                start_button.setEnabled(True)
                stop_button.setEnabled(False)
                timer.stop()
                preview_timer.stop()
                set_controls_enabled(True)
                thread.quit()
                thread.wait(1000)

            @QtCore.Slot(str)
            def on_fail(self, message: str) -> None:
                if self._done:
                    return
                self._done = True
                flush_log()
                start_button.setEnabled(True)
                stop_button.setEnabled(False)
                timer.stop()
                preview_timer.stop()
                set_controls_enabled(True)
                QtWidgets.QMessageBox.critical(window, t("run_failed"), message)
                thread.quit()
                thread.wait(1000)

            @QtCore.Slot(object)
            def on_face_crop(self, crop) -> None:
                if crop is None:
                    return
                height, width = crop.shape[:2]
                bytes_per_line = width * 3
                image = QtGui.QImage(
                    crop.data,
                    width,
                    height,
                    bytes_per_line,
                    QtGui.QImage.Format_BGR888,
                )
                pixmap = QtGui.QPixmap.fromImage(image)
                face_preview.setPixmap(
                    pixmap.scaled(
                        face_preview.size(),
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation,
                    )
                )

        timer.start(500)
        preview_timer.start(1000)
        ui_handler = UiHandler()
        worker.finished.connect(ui_handler.on_finish, QtCore.Qt.QueuedConnection)
        thread.finished.connect(ui_handler.on_finish, QtCore.Qt.QueuedConnection)
        worker.failed.connect(ui_handler.on_fail, QtCore.Qt.QueuedConnection)
        worker.face_crop.connect(ui_handler.on_face_crop, QtCore.Qt.QueuedConnection)
        thread.start()
        thread_ref["thread"] = thread
        thread_ref["worker"] = worker
        thread_ref["timer"] = timer

    usage_timer = QtCore.QTimer(window)

    def format_usage_text(label_key: str, value: float | None) -> str:
        if value is None:
            return f"{t(label_key)}: {t('na')}"
        return f"{t(label_key)}: {value:.0f}%"

    def update_usage() -> None:
        if cpu_usage_provider is not None:
            cpu_usage = cpu_usage_provider()
        else:
            cpu_usage = None
        cpu_label.setText(format_usage_text("cpu_label", cpu_usage))

        if gpu_usage_provider is not None:
            gpu_usage = gpu_usage_provider()
        else:
            gpu_usage = None
        gpu_label.setText(format_usage_text("gpu_label", gpu_usage))

    usage_timer.timeout.connect(update_usage)
    usage_timer.start(1000)
    update_usage()

    def stop_run() -> None:
        stop_event = thread_ref.get("stop_event")
        if stop_event is None:
            return
        stop_event.set()
        log_output.appendPlainText(t("stop_requested"))
        stop_button.setEnabled(False)
        start_button.setEnabled(False)
        set_controls_enabled(False)

    start_button.clicked.connect(start_run)
    stop_button.clicked.connect(stop_run)

    window.setMinimumWidth(640)
    window.show()
    app.exec()


def run_with_args(args: argparse.Namespace) -> None:
    if not args.video_dir:
        raise ValueError("未指定视频目录。")
    video_dir = Path(args.video_dir).resolve()
    confirm_dir = (
        Path(args.confirm_dir)
        if Path(args.confirm_dir).is_absolute()
        else video_dir / args.confirm_dir
    )
    confirm_dir.mkdir(parents=True, exist_ok=True)

    processed_path = Path(args.processed_list)
    if not processed_path.is_absolute():
        processed_path = video_dir / processed_path

    timing_log = Path(args.timing_log)
    if not timing_log.is_absolute():
        timing_log = video_dir / timing_log
    debug_log = Path(args.debug_log)
    if not debug_log.is_absolute():
        debug_log = video_dir / debug_log

    if not args.target_embedding and not args.target_images:
        default_embedding = Path(__file__).with_name("target_embedding.npy")
        if default_embedding.exists():
            args.target_embedding = str(default_embedding)
        else:
            raise ValueError(
                "请提供 --target-embedding 或 --target-images "
                "（或将 target_embedding.npy 放在脚本同目录）。"
            )

    ctx_id = -1 if args.cpu_only else 0
    if args.target_embedding:
        target_embedding = np.load(args.target_embedding)
    else:
        app = build_face_app(tuple(args.det_size), args.max_face_detect, ctx_id)
        target_embedding = compute_target_embedding(app, [Path(p) for p in args.target_images])
    target_embedding = normalize_target_embedding(target_embedding, args.embedding_aggregation)
    window_start = None
    window_end = None
    if args.motion_filter_window:
        window_start = parse_time(args.motion_filter_window[0])
        window_end = parse_time(args.motion_filter_window[1])

    processed = load_processed_list(processed_path)
    stop_event = getattr(args, "stop_event", None)
    preview_queue = getattr(args, "preview_queue", None)
    on_face_crop = None
    if preview_queue is not None:
        def on_face_crop(crop: np.ndarray, video_path: Path, timestamp_s: float | None) -> None:
            try:
                target_path = confirm_dir / video_path.name
                preview_queue.put_nowait(
                    {"crop": crop, "video_path": str(target_path), "timestamp_s": timestamp_s}
                )
            except Exception:
                return

    batch_start = dt.datetime.now()
    log_line(timing_log, f"[批次开始] {format_timestamp(batch_start)}")

    all_videos = iter_videos(
        video_dir,
        include_subdirs=bool(args.include_subdirs),
        exclude_dir=confirm_dir if confirm_dir.exists() else None,
    )
    if not all_videos:
        raise ValueError("当前目录无视频")
    pending_videos = [path for path in all_videos if path.name not in processed]
    if not pending_videos:
        raise ValueError("所有视频已处理")
    total_videos = len(pending_videos)
    completed = 0
    total_duration = 0.0
    if args.workers <= 1:
        app = build_face_app(tuple(args.det_size), args.max_face_detect, ctx_id)
        for video_path in pending_videos:
            if stop_event is not None and stop_event.is_set():
                break
            enable_motion_filter = is_motion_filter_enabled(
                video_path, args, window_start, window_end
            )
            start_time = dt.datetime.now()
            log_line(
                timing_log,
                f"[开始] {video_path.name} | {format_timestamp(start_time)}",
            )
            try:
                fps = get_video_fps(video_path)
                if args.sample_every_n_frames is not None:
                    frame_interval = max(args.sample_every_n_frames, 1)
                else:
                    frame_interval = max(int(round(fps / args.sample_fps)), 1) if fps > 0 else 1
                if args.use_ffmpeg:
                    result = match_video_ffmpeg(
                        app,
                        video_path,
                        target_embedding,
                        frame_interval,
                        args.threshold,
                        args.ffmpeg_path,
                        tuple(args.ffmpeg_scale) if args.ffmpeg_scale else None,
                        enable_motion_filter,
                        args.motion_threshold,
                        args.motion_min_ratio,
                        stop_event,
                        on_face_crop,
                    )
                else:
                    result = match_video(
                        app,
                        video_path,
                        target_embedding,
                        args.sample_fps,
                        args.sample_every_n_frames,
                        args.threshold,
                        enable_motion_filter,
                        args.motion_threshold,
                        args.motion_min_ratio,
                        stop_event,
                        on_face_crop,
                    )
                if result.matched and result.best_frame is not None:
                    save_match_snapshot(
                        confirm_dir,
                        video_path,
                        result.best_frame,
                        result.best_timestamp_s,
                        result.best_frame_index,
                        result.best_score,
                        result.best_bbox,
                    )
                    shutil.move(str(video_path), confirm_dir / video_path.name)
            except Exception:
                error_time = format_timestamp(dt.datetime.now())
                log_line(debug_log, f"Error time: {error_time}")
                log_line(debug_log, f"Video: {video_path}")
                log_line(debug_log, traceback.format_exc().rstrip())
                log_line(debug_log, "-" * 80)
            finally:
                append_processed(processed_path, video_path.name)
                end_time = dt.datetime.now()
                duration = (end_time - start_time).total_seconds()
                total_duration += duration
                completed += 1
                log_line(
                    timing_log,
                    f"[结束] {video_path.name} | {format_timestamp(end_time)} | {duration:.2f}s",
                )
                remaining = total_videos - completed
                if completed > 0 and remaining > 0:
                    avg_duration = total_duration / completed
                    eta_seconds = avg_duration * remaining
                    eta_time = end_time + dt.timedelta(seconds=eta_seconds)
                    eta_line = (
                        f"[进度] {completed}/{total_videos} | "
                        f"ETA {format_duration(eta_seconds)} "
                        f"(预计完成: {format_timestamp(eta_time)})"
                    )
                    log_line(timing_log, eta_line)
    else:
        args.on_face_crop = None
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        log_lock = ctx.Lock()
        processed_lock = ctx.Lock()
        preview_queue = getattr(args, "preview_queue", None) or ctx.Queue()
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=init_worker,
            initargs=(
                args,
                target_embedding,
                str(confirm_dir),
                str(timing_log),
                str(debug_log),
                log_lock,
                processed_lock,
                preview_queue,
                window_start,
                window_end,
            ),
        )
        try:
            futures = {}
            for video_path in pending_videos:
                if stop_event is not None and stop_event.is_set():
                    break
                futures[executor.submit(process_video_with_resume, video_path, processed_path)] = (
                    video_path
                )
            for future in as_completed(futures):
                try:
                    duration = future.result()
                except Exception:
                    error_time = format_timestamp(dt.datetime.now())
                    log_line(debug_log, f"Error time: {error_time}")
                    log_line(debug_log, "Video: worker failure")
                    log_line(debug_log, traceback.format_exc().rstrip())
                    log_line(debug_log, "-" * 80)
                    duration = 0.0
                total_duration += duration
                completed += 1
                if stop_event is not None and stop_event.is_set():
                    for pending in futures:
                        pending.cancel()
                    executor.shutdown(wait=True, cancel_futures=True)
                    break
                remaining = total_videos - completed
                if completed > 0 and remaining > 0:
                    avg_duration = total_duration / completed
                    eta_seconds = avg_duration * remaining
                    eta_time = dt.datetime.now() + dt.timedelta(seconds=eta_seconds)
                    eta_line = (
                        f"[进度] {completed}/{total_videos} | "
                        f"ETA {format_duration(eta_seconds)} "
                        f"(预计完成: {format_timestamp(eta_time)})"
                    )
                    log_line(timing_log, eta_line)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    batch_end = dt.datetime.now()
    log_line(timing_log, f"[批次结束] {format_timestamp(batch_end)}")


def init_worker(
    args: argparse.Namespace,
    target_embedding: np.ndarray,
    confirm_dir: str,
    timing_log: str,
    debug_log: str,
    log_lock,
    processed_lock,
    preview_queue,
    window_start: Optional[dt.time],
    window_end: Optional[dt.time],
) -> None:
    global _APP, _TARGET_EMBEDDING, _ARGS, _CONFIRM_DIR, _TIMING_LOG, _DEBUG_LOG
    global _LOG_LOCK, _PROCESSED_LOCK, _PREVIEW_QUEUE, _STOP_EVENT
    _ARGS = args
    _TARGET_EMBEDDING = target_embedding
    _CONFIRM_DIR = Path(confirm_dir)
    _TIMING_LOG = Path(timing_log)
    _DEBUG_LOG = Path(debug_log)
    _LOG_LOCK = log_lock
    _PROCESSED_LOCK = processed_lock
    _PREVIEW_QUEUE = preview_queue
    _STOP_EVENT = getattr(args, "stop_event", None)
    _ARGS.motion_filter_window_start = window_start
    _ARGS.motion_filter_window_end = window_end
    _ARGS.on_face_crop = None
    ctx_id = -1 if args.cpu_only else 0
    _APP = build_face_app(tuple(args.det_size), args.max_face_detect, ctx_id)


def append_processed_locked(path: Path, video_name: str) -> None:
    if _PROCESSED_LOCK is None:
        append_processed(path, video_name)
        return
    with _PROCESSED_LOCK:
        append_processed(path, video_name)


def is_cuda_failure(error_text: str) -> bool:
    lowered = error_text.lower()
    if "cuda" not in lowered:
        return False
    return "onnxruntime" in lowered or "cuda failure" in lowered or "cudart" in lowered


def process_video(video_path: Path) -> float:
    if _APP is None or _TARGET_EMBEDDING is None or _ARGS is None:
        raise RuntimeError("Worker not initialized.")
    if _CONFIRM_DIR is None or _TIMING_LOG is None or _DEBUG_LOG is None:
        raise RuntimeError("Worker logging not initialized.")
    if _STOP_EVENT is not None and _STOP_EVENT.is_set():
        return 0.0
    on_face_crop = enqueue_preview if _PREVIEW_QUEUE is not None else None
    window_start = getattr(_ARGS, "motion_filter_window_start", None)
    window_end = getattr(_ARGS, "motion_filter_window_end", None)
    enable_motion_filter = is_motion_filter_enabled(video_path, _ARGS, window_start, window_end)
    start_time = dt.datetime.now()
    log_line_locked(
        _TIMING_LOG,
        f"[开始] {video_path.name} | {format_timestamp(start_time)}",
    )
    try:
        fps = get_video_fps(video_path)
        if _ARGS.sample_every_n_frames is not None:
            frame_interval = max(_ARGS.sample_every_n_frames, 1)
        else:
            frame_interval = max(int(round(fps / _ARGS.sample_fps)), 1) if fps > 0 else 1
        if _ARGS.use_ffmpeg:
            result = match_video_ffmpeg(
                _APP,
                video_path,
                _TARGET_EMBEDDING,
                frame_interval,
                _ARGS.threshold,
                _ARGS.ffmpeg_path,
                tuple(_ARGS.ffmpeg_scale) if _ARGS.ffmpeg_scale else None,
                enable_motion_filter,
                _ARGS.motion_threshold,
                _ARGS.motion_min_ratio,
                _STOP_EVENT,
                on_face_crop,
            )
        else:
            result = match_video(
                _APP,
                video_path,
                _TARGET_EMBEDDING,
                _ARGS.sample_fps,
                _ARGS.sample_every_n_frames,
                _ARGS.threshold,
                enable_motion_filter,
                _ARGS.motion_threshold,
                _ARGS.motion_min_ratio,
                _STOP_EVENT,
                on_face_crop,
            )
        if result.matched and result.best_frame is not None:
            save_match_snapshot(
                _CONFIRM_DIR,
                video_path,
                result.best_frame,
                result.best_timestamp_s,
                result.best_frame_index,
                result.best_score,
                result.best_bbox,
            )
            shutil.move(str(video_path), _CONFIRM_DIR / video_path.name)
    except Exception:
        trace = traceback.format_exc().rstrip()
        error_time = format_timestamp(dt.datetime.now())
        if is_cuda_failure(trace):
            log_line_locked(_DEBUG_LOG, "Detected CUDA failure. Stopping further processing.")
            if _STOP_EVENT is not None:
                _STOP_EVENT.set()
        log_line_locked(_DEBUG_LOG, f"Error time: {error_time}")
        log_line_locked(_DEBUG_LOG, f"Video: {video_path}")
        log_line_locked(_DEBUG_LOG, trace)
        log_line_locked(_DEBUG_LOG, "-" * 80)
    finally:
        end_time = dt.datetime.now()
        duration = (end_time - start_time).total_seconds()
        log_line_locked(
            _TIMING_LOG,
            f"[结束] {video_path.name} | {format_timestamp(end_time)} | {duration:.2f}s",
        )
        return duration


def process_video_with_resume(video_path: Path, processed_path: Path) -> float:
    if _STOP_EVENT is not None and _STOP_EVENT.is_set():
        return 0.0
    try:
        return process_video(video_path)
    finally:
        if _STOP_EVENT is None or not _STOP_EVENT.is_set():
            append_processed_locked(processed_path, video_path.name)


def main() -> None:
    if os.name == "nt":
        try:
            import ctypes

            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 0)
        except Exception:
            pass
    run_gui()


if __name__ == "__main__":
    main()
