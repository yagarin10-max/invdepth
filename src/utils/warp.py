# utils/warp.py

import torch
from typing import Tuple
from .mesh import construct_mesh
from .render import render_mesh

def warp_with_inverse_depth_mesh(
    image: torch.Tensor,
    invdepth: torch.Tensor,
    device: torch.device,
    K: torch.Tensor,
    RT_ref_to_view: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """逆深度マップを用いたメッシュベースの画像ワーピング
    
    Args:
        image: 入力画像 (B, C, H, W)
        invdepth: インバースデプス画像 (B, 1, H, W)
        device: 計算に使用するデバイス
        K: カメラ内部パラメータ行列 (B, 3, 3)
        RT_ref_to_view: 参照視点から目標視点への変換行列 (B, 3, 4)
        
    Returns:
        warped_image: ワープされた画像 (B, C, H, W)
        mask: 有効領域のマスク (B, 1, H, W)
    """
    # 入力テンソルの形状を統一
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    if len(K.shape) == 2:
        K = K.unsqueeze(0)
    if len(RT_ref_to_view.shape) == 2:
        RT_ref_to_view = RT_ref_to_view.unsqueeze(0)

    # メッシュの構築とレンダリング
    mesh = construct_mesh(image, invdepth, K, device)
    warped_image, _, mask = render_mesh(mesh, K, RT_ref_to_view, device)

    return warped_image, mask

def patch_based_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int = 3
) -> torch.Tensor:
    """パッチベースのロス計算
    
    Args:
        pred: 予測画像 (B, C, H, W)
        target: 目標画像 (B, C, H, W)
        mask: マスク画像 (B, 1, H, W)
        patch_size: パッチサイズ
        
    Returns:
        loss: パッチベースのロス値 (H, W)
    """
    h, w = pred.shape[-2:]
    pad = patch_size // 2

    pred_padded = torch.nn.functional.pad(
        pred, (pad, pad, pad, pad), mode='reflect'
    )
    target_padded = torch.nn.functional.pad(
        target, (pad, pad, pad, pad), mode='reflect'
    )
    mask_padded = torch.nn.functional.pad(
        mask, (pad, pad, pad, pad), mode='reflect'
    )

    pred_patches = torch.nn.functional.unfold(
        pred_padded, 
        kernel_size=patch_size, 
        stride=1
    )
    target_patches = torch.nn.functional.unfold(
        target_padded, 
        kernel_size=patch_size, 
        stride=1
    )
    mask_patches = torch.nn.functional.unfold(
        mask_padded, 
        kernel_size=patch_size, 
        stride=1
    )

    valid_patches = (mask_patches.sum(dim=1) == patch_size ** 2).float()
    patch_losses = torch.abs(pred_patches - target_patches).mean(dim=1)

    total_loss = (patch_losses * valid_patches).sum()
    num_valid_patches = valid_patches.sum() + 1e-6

    return total_loss / num_valid_patches

def pixel_wise_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ピクセルごとのロス計算
    
    Args:
        pred: 予測画像
        target: 目標画像
        mask: マスク画像
        
    Returns:
        loss: ピクセルごとのロス値
    # """
    pixel_losses = torch.abs(pred - target)
    total_loss = (pixel_losses * mask).sum()
    num_valid_pixels = mask.sum() + 1e-6
    return total_loss / num_valid_pixels
    # return torch.abs(pred - target)

def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    use_patch_loss: bool = True,
    patch_size: int = 3
) -> torch.Tensor:
    """ロスの計算
    
    Args:
        pred: 予測画像
        target: 目標画像
        mask: マスク画像
        use_patch_loss: パッチベースのロスを使用するかどうか
        patch_size: パッチサイズ
        
    Returns:
        loss: 計算されたロス値
    """
    if use_patch_loss:
        return patch_based_loss(pred, target, mask, patch_size)
    else:
        return pixel_wise_loss(pred, target, mask)