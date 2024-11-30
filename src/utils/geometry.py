# utils/geometry.py

import torch
import numpy as np
from typing import Tuple

def get_screen_pixel_coord(h: int, w: int, device: torch.device) -> torch.Tensor:
    """画面上の正規化されたピクセル座標を取得
    
    Args:
        h: 画像の高さ
        w: 画像の幅
        device: 計算に使用するデバイス
        
    Returns:
        pixel_coord: 正規化されたピクセル座標 (1, h, w, 2)
    """
    x = torch.arange(w, device=device)
    y = torch.arange(h, device=device)
    x = (x + 0.5) / w
    y = (y + 0.5) / h
    x = x[None, None, ..., None].repeat(1, h, 1, 1)
    y = y[None, ..., None, None].repeat(1, 1, w, 1)
    pixel_coord = torch.cat([x, y], dim=-1)
    return pixel_coord

def lift_to_homo(coord: torch.Tensor) -> torch.Tensor:
    """座標を同次座標に変換
    
    Args:
        coord: 入力座標 (..., k)
        
    Returns:
        homo_coord: 同次座標 (..., k+1)
    """
    ones = torch.ones_like(coord[..., -1:])
    return torch.cat([coord, ones], dim=-1)

def norm_depth(depth: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
    """デプス値を正規化
    
    Args:
        depth: デプス値
        min_val: 最小値
        max_val: 最大値
        
    Returns:
        normalized_depth: 正規化されたデプス値
    """
    return (depth - min_val) / (max_val - min_val)

def denorm_depth(depth: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
    """正規化されたデプス値を元のスケールに戻す
    
    Args:
        depth: 正規化されたデプス値
        min_val: 最小値
        max_val: 最大値
        
    Returns:
        denormalized_depth: 元のスケールのデプス値
    """
    return depth * (max_val - min_val) + min_val