# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import hashlib
import json
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
import requests
from omegaconf import OmegaConf

cur_dir = Path(__file__).resolve().parent
project_dir = cur_dir.parent
cfg_path = project_dir / "config.yaml"

cfg = OmegaConf.load(cfg_path)
FFMPEG_PATH = cfg.ffmpeg_path
FFPROBE_PATH = cfg.ffprobe_path


def generate_equal_segments(total_length, num_segments):
    """
    生成均分的索引段

    参数:
        total_length (int): 总长度/范围
        num_segments (int): 要分成几段

    返回:
        list: 包含(start, end)元组的列表，表示每段的索引范围
    """
    if num_segments <= 0 or total_length <= 0:
        return []

    segment_size = total_length // num_segments
    remainder = total_length % num_segments

    segments = []
    start = 0

    for i in range(num_segments):
        # 计算当前段的大小（前面的段可能会多1个元素）
        current_size = segment_size + (1 if i < remainder else 0)
        end = start + current_size
        segments.append((start, end))
        start = end

    return segments


def iou(box0: np.ndarray, box1: np.ndarray):
    """计算一对一交并比

    Parameters
    ----------
    box0, box1: `~np.ndarray` of shape `(4, )`
        边界框
    """
    xy_max = np.minimum(box0[2:], box1[2:])
    xy_min = np.maximum(box0[:2], box1[:2])

    # 计算交集
    inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)
    inter = inter[0] * inter[1]

    # 计算并集
    area_0 = (box0[2] - box0[0]) * (box0[3] - box0[1])
    area_1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    union = area_0 + area_1 - inter

    return inter / union


def read_img(img_path: Union[str, Path]) -> np.ndarray:
    return cv2.imread(str(img_path))


def save_img(save_path: Union[str, Path], img: np.ndarray):
    cv2.imwrite(str(save_path), img)


def get_unique_value_by_md5(txt: Union[str, Path]) -> str:
    txt = str(txt).strip()
    hash_object = hashlib.md5(txt.encode())
    unique_value = hash_object.hexdigest()
    return unique_value


def extract_frames_by_ffmpeg(video_path: str, save_dir: Path):
    save_img_pattern = str(save_dir / "%04d.jpg")
    all_frames_cmd = [
        FFMPEG_PATH,
        "-i",
        video_path,
        "-q:v",
        "2",
        save_img_pattern,
    ]
    result = subprocess.run(all_frames_cmd, check=True, capture_output=True)
    if result.returncode != 0:
        print(f"ffmpeg warning/error output: {result.stderr}")


def load_npz_v1(file_path: Union[str, Path]):
    return np.load(file_path)["data"]


def save_npz_v1(save_path: Union[str, Path], data: np.ndarray):
    np.savez_compressed(str(save_path), data=data)


def load_npz(filename: str) -> np.ndarray:
    """加载高效保存的二值掩码
    Args:
        filename: 输入文件名
    Returns:
        masks: 形状为 (F, N, H, W) 的二值掩码数组
    """
    data = np.load(filename)
    packed_masks = data["data"]
    F, N, H, W = data["shape"]
    # 解包位数组
    unpacked = np.unpackbits(packed_masks, axis=1)
    # 可能需要截断多余的位
    unpacked = unpacked[:, : H * W]
    # 重塑为原始形状
    masks = unpacked.reshape(F, N, H, W).astype(np.uint8)
    return masks


def save_npz(save_path: Union[str, Path], data: np.ndarray):
    """将二值掩码高效保存为压缩文件
    Args:
        masks: 形状为 (F, N, H, W) 的二值掩码数组
        filename: 输出文件名
    """
    # 将浮点掩码转换为位掩码
    binary_masks = data.astype(bool)

    # 获取形状信息
    F, N, H, W = binary_masks.shape

    # 将每个掩码打包为位数组
    packed_masks = np.packbits(binary_masks.reshape(F * N, H * W), axis=1)

    # 保存压缩数据
    np.savez_compressed(str(save_path), data=packed_masks, shape=(F, N, H, W))


class ReadVideo:
    @classmethod
    def get_frame(
        cls, video_path: Union[str, Path], frame_idx: int
    ) -> Optional[np.ndarray]:
        with cls.video_capture(str(video_path)) as v:
            v.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = v.read()
            if not ret:
                return None
            return frame

    @classmethod
    def get_video_info(cls, video_path: Union[str, Path]) -> Dict[str, Any]:
        with cls.video_capture(str(video_path)) as v:
            fps = v.get(cv2.CAP_PROP_FPS)
            width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            return {
                "fps": fps,
                "width": width,
                "height": height,
                "total_frames": total_frames,
                "duration": duration,
            }

    @staticmethod
    @contextmanager
    def video_capture(img):
        cap = cv2.VideoCapture(img)
        try:
            yield cap
        finally:
            cap.release()


def compress_video(input_path: str, output_path: str, verbose: bool = False):
    """
    使用 ffmpeg 压缩视频文件为 libx264 格式
    :param input_path: 原始视频路径
    :param output_path: 压缩后的视频路径
    """
    if verbose:
        command = [
            FFMPEG_PATH,
            "-i",
            input_path,
            "-c:v",
            "libx264",
            "-an",
            "-y",
            output_path,
        ]
    else:
        command = [
            FFMPEG_PATH,
            "-i",
            input_path,
            "-c:v",
            "libx264",
            "-an",
            "-y",
            "-loglevel",
            "quiet",
            output_path,
        ]
    subprocess.run(command, check=True)


def read_csv(csv_path):
    return pd.read_csv(csv_path)


def load_config(config_path: Union[str, Path]):
    return OmegaConf.load(config_path)


def mkdir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def read_json(json_path: Union[str, Path]):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_json(save_path, content: Union[List, Dict], mode="w"):
    with open(save_path, mode, encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False)


def read_jsonl(jsonl_path: Union[str, Path]) -> List[Dict[Any, Any]]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(v) for v in f]
    return data


def write_jsonl(save_path: Union[str, Path], content: List[Any], mode: str = "w"):
    if not isinstance(content, list):
        content = [content]

    with open(save_path, mode, encoding="utf-8") as f:
        for value in content:
            value = json.dumps(value)
            f.write(f"{value}\n")


def read_txt(txt_path: Union[Path, str]) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        data = [v.rstrip("\n") for v in f]
    return data


def write_txt(
    save_path: Union[str, Path], contents: Union[List[str], str], mode: str = "w"
) -> None:
    if not isinstance(contents, list):
        contents = [contents]

    with open(save_path, mode, encoding="utf-8") as f:
        for value in contents:
            f.write(f"{value}\n")


def cp_file(file_path, dst_dir):
    img_dst_path = dst_dir / Path(file_path).name
    if not img_dst_path.exists():
        shutil.copy(file_path, dst_dir)


def download_oss_file(url, save_path):
    response = requests.get(url, stream=True, timeout=60)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        return True
    return False


def check_video_codec(video_path):
    # 检查视频编码是否为H.264
    command = [
        FFPROBE_PATH,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.decode()}")

    metadata = json.loads(result.stdout)
    for stream in metadata.get("streams", []):
        if "codec_name" in stream:
            return stream["codec_name"] == "h264"
    return False


def calculate_frame_md5(frame):
    return hashlib.md5(frame.tobytes()).hexdigest()


def calculate_file_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compare_videos_by_md5(video_path1, video_path2):
    md5_1 = calculate_file_md5(video_path1)
    md5_2 = calculate_file_md5(video_path2)

    if md5_1 == md5_2:
        return True
    return False
