# utils/render.py

import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.ops import interpolate_face_attributes
from typing import Dict, Tuple
from .geometry import lift_to_homo
from .camera import get_perspective_from_intrinsic

def render_mesh(
    mesh_dict: Dict[str, torch.Tensor], 
    cam_int: torch.Tensor, 
    cam_ext: torch.Tensor, 
    device: torch.device,
    eps: float = 1e-3,
    near: float = 0.1,
    far: float = 10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """メッシュをレンダリング
    
    Args:
        mesh_dict: construct_mesh関数の出力
        cam_int: カメラ内部パラメータ行列 (B, 3, 3)
        cam_ext: カメラ外部パラメータ行列 (B, 3, 4)
        device: 計算に使用するデバイス
        eps: 数値安定性のための小さな値
        near: 近平面までの距離
        far: 遠平面までの距離
        
    Returns:
        render: レンダリングされた画像 (B, 3, H, W)
        invdepth: レンダリングされたインバースデプスマップ (B, 1, H, W)
        mask: マスク (B, 1, H, W)
    """
    vertice = mesh_dict["vertice"]
    faces = mesh_dict["faces"]
    attributes = mesh_dict["attributes"]
    h, w = mesh_dict["size"]

    # print("変換前の頂点座標範囲:")
    # print(f"X: [{vertice[..., 0].min():.4f}, {vertice[..., 0].max():.4f}]")
    # print(f"Y: [{vertice[..., 1].min():.4f}, {vertice[..., 1].max():.4f}]")
    # print(f"Z: [{vertice[..., 2].min():.4f}, {vertice[..., 2].max():.4f}]")

    # NDC空間への変換
    vertice_homo = lift_to_homo(vertice)
    vertice_world = torch.matmul(
        cam_ext.unsqueeze(1), 
        vertice_homo[..., None]
    ).squeeze(-1)
    
    vertice_depth = vertice_world[..., -1:]
    attributes = torch.cat([attributes, vertice_depth], dim=-1)
    
    # print("\nカメラ空間での座標範囲:")
    # print(f"X: [{vertice_world[..., 0].min():.4f}, {vertice_world[..., 0].max():.4f}]")
    # print(f"Y: [{vertice_world[..., 1].min():.4f}, {vertice_world[..., 1].max():.4f}]")
    # print(f"Z: [{vertice_world[..., 2].min():.4f}, {vertice_world[..., 2].max():.4f}]")

    vertice_world_homo = lift_to_homo(vertice_world)
    persp = get_perspective_from_intrinsic(cam_int, near, far)
    vertice_ndc = torch.matmul(
        persp.unsqueeze(1), 
        vertice_world_homo[..., None]
    ).squeeze(-1)
    
    vertice_ndc = vertice_ndc[..., :-1] / vertice_ndc[..., -1:]
    vertice_ndc[..., :-1] *= -1
    vertice_ndc[..., 0] *= w / h

    # print("\nNDC空間での座標範囲:")
    # print(f"X: [{vertice_ndc[..., 0].min():.4f}, {vertice_ndc[..., 0].max():.4f}]")
    # print(f"Y: [{vertice_ndc[..., 1].min():.4f}, {vertice_ndc[..., 1].max():.4f}]")
    # print(f"Z: [{vertice_ndc[..., 2].min():.4f}, {vertice_ndc[..., 2].max():.4f}]")

    # レンダリング
    mesh = Meshes(vertice_ndc, faces)
    pix_to_face, _, bary_coords, _ = rasterize_meshes(
        mesh, 
        (h, w), 
        faces_per_pixel=1, 
        blur_radius=1e-6
    )

    b, nf, _ = faces.size()
    faces = faces.reshape(b, nf * 3, 1).repeat(1, 1, 5)
    face_attributes = torch.gather(attributes, dim=1, index=faces)
    face_attributes = face_attributes.reshape(b * nf, 3, 5)
    
    output = interpolate_face_attributes(pix_to_face, bary_coords, face_attributes)
    output = output.squeeze(-2).permute(0, 3, 1, 2)

    render = output[:, :3]
    mask = output[:, 3:4]
    invdepth = torch.reciprocal(output[:, 4:] + eps)

    return render, invdepth, mask