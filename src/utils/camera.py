# utils/camera.py

import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Tuple, Dict
import torch

def get_camera_matrices(
    width: int, 
    height: int, 
    focal_length_m: float, 
    sensor_width_m: float,
    location: List[float] = [0,0,0], 
    rotation: List[float] = [0,0,0]
) -> Tuple[np.ndarray, np.ndarray]:
    """カメラの内部・外部パラメータ行列を計算

    Args:
        width: 画像の幅
        height: 画像の高さ
        focal_length_m: 焦点距離（メートル）
        sensor_width_m: センサー幅（メートル）
        location: カメラ位置 [x,y,z]
        rotation: カメラ回転 [rx,ry,rz]（度）

    Returns:
        K: カメラ内部パラメータ行列 (3, 3)
        RT: カメラ外部パラメータ行列 (3, 4)
    """
    # 内部パラメータ
    focal_px = (width * focal_length_m) / sensor_width_m
    K = np.array([
        [focal_px, 0, width/2],
        [0, focal_px, height/2],
        [0, 0, 1]
    ])
    # normalize K with pixel width as like AdaMPI
    K = K / width
    # 外部パラメータ
    rotation_rad = np.array(rotation) * np.pi / 180
    R = Rotation.from_euler('xyz', rotation_rad).as_matrix()
    t = np.array(location).reshape(3, 1)
    RT = np.hstack([R, t])

    return K, RT

def compute_relative_transform(RT1: torch.tensor, RT2: torch.Tensor) -> torch.Tensor:
    """2つの変換行列間の相対変換を計算

    Args:
        RT1: 基準となる変換行列 (3, 4)
        RT2: 目標となる変換行列 (3, 4)

    Returns:
        relative: 相対変換行列 (3, 4)
    """
    device = RT1.device
    RT1_4x4 = torch.eye(4, device=device)
    RT1_4x4 = RT1.clone()
    RT2_4x4 = torch.eye(4, device=device)
    RT2_4x4 = RT2.clone()

    relative = RT2_4x4 @ torch.inverse(RT1_4x4)
    return relative[:3,:]

def get_perspective_from_intrinsic(cam_int: torch.tensor, near: float, far: float) -> torch.tensor:
    """カメラの内部パラメータから透視投影行列を計算

    Args:
        cam_int: カメラ内部パラメータ行列 (B, 3, 3)
        near: 近平面までの距離
        far: 遠平面までの距離

    Returns:
        persp: 透視投影行列 (B, 4, 4)
    """

    fx, fy = cam_int[:, 0, 0], cam_int[:, 1, 1]
    cx, cy = cam_int[:, 0, 2], cam_int[:, 1, 2]
        
    one = torch.ones_like(cx)
    zero = torch.zeros_like(cx)

    near_z, far_z = near * one, far * one
    a = (near_z + far_z) / (far_z - near_z)
    b = -2.0 * near_z * far_z / (far_z - near_z)

    matrix = [[2.0 * fx, zero, 2.0 * cx - 1.0, zero],
              [zero, 2.0 * fy, 2.0 * cy - 1.0, zero],
              [zero, zero, a, b],
              [zero, zero, one, zero]]
    
    persp = torch.stack([torch.stack(row, dim=-1) for row in matrix], dim=-2)
    return persp