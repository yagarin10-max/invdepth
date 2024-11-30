# utils/mesh.py

import torch
from typing import Dict, List, Tuple
from .geometry import get_screen_pixel_coord, lift_to_homo
import torch.nn.functional as F

def get_faces(h: int, w: int, device: torch.device) -> torch.Tensor:
    """メッシュの面接続情報を取得
    
    Args:
        h: 画像の高さ
        w: 画像の幅
        device: 計算に使用するデバイス
        
    Returns:
        faces: 面接続情報 (1, nface, 3)
    """
    x = torch.arange(w - 1, device=device)
    y = torch.arange(h - 1, device=device)
    x = x[None, None, ..., None].repeat(1, h - 1, 1, 1)
    y = y[None, ..., None, None].repeat(1, 1, w - 1, 1)

    tl = y * w + x
    tr = y * w + x + 1
    bl = (y + 1) * w + x
    br = (y + 1) * w + x + 1

    faces_l = torch.cat([tl, bl, br], dim=-1).reshape(1, -1, 3)
    faces_r = torch.cat([br, tr, tl], dim=-1).reshape(1, -1, 3)

    return torch.cat([faces_l, faces_r], dim=1)

def construct_mesh(
    image: torch.Tensor, 
    invdepth: torch.Tensor, 
    cam_int: torch.Tensor,
    device: torch.device,
    eps: float = 1e-6
) -> Dict[str, torch.Tensor]:
    """メッシュを構築
    
    Args:
        image: 入力画像 (B, C, H, W)
        invdepth: インバースデプス画像 (B, 1, H, W)
        cam_int: カメラ内部パラメータ行列 (B, 3, 3)
        device: 計算に使用するデバイス
        eps: 数値安定性のための小さな値
        
    Returns:
        mesh_dict: メッシュ情報を含む辞書
            - vertice: 頂点座標 (B, H*W, 3)
            - faces: 面接続情報 (B, nface, 3)
            - attributes: 頂点属性 (B, H*W, 4)
            - size: メッシュサイズ [H, W]
    """
    b, c, h, w = image.shape
    invdepth = invdepth.permute(0, 2, 3, 1)
    image = image.permute(0, 2, 3, 1)

    # デプスの正規化とマスク生成
    depth = torch.reciprocal(invdepth + eps)

    # ピクセル座標の取得と3D投影
    pixel_2d = get_screen_pixel_coord(h, w, device)
    pixel_2d_homo = lift_to_homo(pixel_2d)

    cam_int_inv = torch.inverse(cam_int)
    pixel_3d = torch.matmul(
        cam_int_inv[:, None, None, :, :], 
        pixel_2d_homo[..., None]
    ).squeeze(-1)
    
    pixel_3d = pixel_3d * depth
    vertice = pixel_3d.reshape(b, h * w, 3)

    # 面接続情報の構築
    faces = get_faces(h, w, device)
    faces = faces.repeat(b, 1, 1)

    # 頂点属性の計算
    attr_color = image.reshape(b, h * w, 3)
    attr_mask = get_visible_mask(invdepth).reshape(b, h * w, 1)
    attr = torch.cat([attr_color, attr_mask], dim=-1)

    return {
        "vertice": vertice,
        "faces": faces,
        "attributes": attr,
        "size": [h, w]
    }

def get_visible_mask(
    invdepth: torch.Tensor, 
    beta: float = 10.0, 
    alpha_threshold: float = 0.3
) -> torch.Tensor:
    """視認可能な領域のマスクを生成
    
    Args:
        invdepth: インバースデプスマップ (B, H, W, 1)
        beta: エッジ検出の感度パラメータ
        alpha_threshold: マスク生成の閾値
        
    Returns:
        vis_mask: 視認可能マスク
    """
    device = invdepth.device
    b, h, w, _ = invdepth.size()
    invdepth = invdepth.reshape(b, 1, h, w)  # [b,1,h,w]
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device).unsqueeze(0).unsqueeze(0).float()
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device).unsqueeze(0).unsqueeze(0).float()
    sobel_x = F.conv2d(invdepth, kernel_x, padding=(1, 1))  # [b,1,h,w]
    sobel_y = F.conv2d(invdepth, kernel_y, padding=(1, 1))  # [b,1,h,w]
    sobel_mag = torch.sqrt(sobel_x ** 2 + sobel_y ** 2).reshape(b, h, w, 1)  # [b,h,w,1]
    alpha = torch.exp(-1.0 * beta * sobel_mag)  # [b,h,w,1]
    vis_mask = torch.greater(alpha, alpha_threshold).float()
    return vis_mask