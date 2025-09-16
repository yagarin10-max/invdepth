import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh import rasterize_meshes
from torch import Tensor
import cv2
import os
from pathlib import Path
import json
import sys
from typing import List, Tuple

eps = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 【追加】可視化用のヘルパー関数
def visualize_results(
    src_img, disparity_map, initial_mask,
    tgt_img, warped_mask, warped_src_img,
    overlayed_img, output_dir, sample_idx
):
    """
    結果をまとめて可視化し、2つの画像ファイルとして保存する関数
    """
    # PyTorchテンソルをMatplotlibで表示できるNumpy配列に変換
    src_img_np = src_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    disparity_map_np = disparity_map.squeeze().cpu().numpy()
    initial_mask_np = initial_mask.squeeze().cpu().numpy()
    tgt_img_np = np.array(tgt_img)
    warped_mask_np = warped_mask.squeeze().cpu().numpy()
    warped_src_img_np = warped_src_img.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # ファイルパスを準備
    output_dir.mkdir(exist_ok=True)
    summary_path = output_dir / f"{sample_idx:06d}_summary.png"
    comparison_path = output_dir / f"{sample_idx:06d}_comparison.png"

    # --- 1. プロセス全体のサマリー画像を生成 ---
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Processing Summary for Sample {sample_idx:06d}', fontsize=16)

    # Source Image
    axs[0, 0].imshow(src_img_np)
    axs[0, 0].set_title('1. Source Image')
    axs[0, 0].axis('off')

    # Disparity Map
    axs[0, 1].imshow(disparity_map_np, cmap='viridis')
    axs[0, 1].set_title('2. Disparity Map')
    axs[0, 1].axis('off')

    # Initial Visibility Mask
    axs[0, 2].imshow(initial_mask_np, cmap='gray')
    axs[0, 2].set_title('3. Initial Visibility Mask')
    axs[0, 2].axis('off')

    # Target Image
    axs[1, 0].imshow(tgt_img_np)
    axs[1, 0].set_title('4. Target Image')
    axs[1, 0].axis('off')

    # Warped Occlusion Mask
    axs[1, 1].imshow(warped_mask_np, cmap='gray')
    axs[1, 1].set_title('5. Warped Occlusion Mask')
    axs[1, 1].axis('off')

    # Overlayed Image
    axs[1, 2].imshow(overlayed_img)
    axs[1, 2].set_title('6. Target + Occlusion Overlay')
    axs[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(summary_path)
    plt.close(fig)
    print(f"  Saved visualization: {summary_path}")

    # --- 2. ワーピング結果の比較画像を生成 ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Warping Comparison for Sample {sample_idx:06d}', fontsize=16)

    # Target Image
    axs[0].imshow(tgt_img_np)
    axs[0].set_title('Target Image')
    axs[0].axis('off')

    # Warped Source Image
    axs[1].imshow(warped_src_img_np)
    axs[1].set_title('Warped Source Image')
    axs[1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(comparison_path)
    plt.close(fig)
    print(f"  Saved visualization: {comparison_path}")

def get_screen_pixel_coord(h, w, device):
    """
    正規化されたピクセル座標をスクリーン上で取得
    """
    x = torch.arange(w).to(device)
    y = torch.arange(h).to(device)
    x = (x + 0.5) / w
    y = (y + 0.5) / h
    x = x[None, None, ..., None].repeat(1, h, 1, 1)
    y = y[None, ..., None, None].repeat(1, 1, w, 1)
    pixel_coord = torch.cat([x, y], dim=-1)
    return pixel_coord

def lift_to_homo(coord):
    """座標を同次座標に変換"""
    ones = torch.ones_like(coord[..., -1:])
    return torch.cat([coord, ones], dim=-1)

def get_faces(h, w, device):
    """メッシュのface接続情報を取得"""
    x = torch.arange(w - 1).to(device)
    y = torch.arange(h - 1).to(device)
    x = x[None, None, ..., None].repeat(1, h - 1, 1, 1)
    y = y[None, ..., None, None].repeat(1, 1, w - 1, 1)

    tl = y * w + x
    tr = y * w + x + 1
    bl = (y + 1) * w + x
    br = (y + 1) * w + x + 1

    faces_l = torch.cat([tl, bl, br], dim=-1).reshape(1, -1, 3)
    faces_r = torch.cat([br, tr, tl], dim=-1).reshape(1, -1, 3)

    return torch.cat([faces_l, faces_r], dim=1)

def get_visible_mask(disparity, beta=10, alpha_threshold=0.3, device='cuda'):
    """
    視差マップをSobelカーネルでフィルタリングし、エッジ（デプス不連続性）をマスクアウト
    """
    b, h, w, _ = disparity.size()
    disparity = disparity.reshape(b, 1, h, w).to(device)
    
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float().to(device)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).float().to(device)
    
    sobel_x = F.conv2d(disparity, kernel_x, padding=(1, 1))
    sobel_y = F.conv2d(disparity, kernel_y, padding=(1, 1))
    sobel_mag = torch.sqrt(sobel_x ** 2 + sobel_y ** 2).reshape(b, h, w, 1)
    
    alpha = torch.exp(-1.0 * beta * sobel_mag)
    vis_mask = torch.greater(alpha, alpha_threshold).float()
    return vis_mask

def get_perspective_from_intrinsic(cam_int):
    """
    内部パラメータから透視投影行列を取得
    """
    near = 1e-4
    far = 1e4
    fx, fy = cam_int[:, 0, 0], cam_int[:, 1, 1]
    cx, cy = cam_int[:, 0, 2], cam_int[:, 1, 2]

    if not isinstance(near, torch.Tensor):
        near = torch.tensor(near, dtype=cam_int.dtype, device=cam_int.device)
    if not isinstance(far, torch.Tensor):
        far = torch.tensor(far, dtype=cam_int.dtype, device=cam_int.device)
    
    bs = cam_int.shape[0]
    if near.dim() == 0:
        near = near.expand(bs).to(cam_int.device)
    if far.dim() == 0:
        far = far.expand(bs).to(cam_int.device)
    
    one = torch.ones_like(cx).to(cam_int.device)
    zero = torch.zeros_like(cx).to(cam_int.device)
    near_z, far_z = near * one, far * one

    a = (far + near) / (far - near)
    b = (-2.0 * far * near) / (far - near)
    
    matrix = [[2.0 * fx, zero, 2.0 * cx - 1.0, zero],
              [zero, 2.0 * fy, 2.0 * cy - 1.0, zero],
              [zero, zero, a, b],
              [zero, zero, one, zero]]
    
    persp = torch.stack([torch.stack(row, dim=-1) for row in matrix], dim=-2)
    return persp

def construct_mesh(image, depth, cam_int, device):
    """
    画像とデプスからメッシュを構築
    """
    b, c, h, w = image.shape
    image = image.to(device)
    depth = depth.to(device)
    cam_int = cam_int.to(device)
    depth = depth.permute(0, 2, 3, 1)
    image = image.permute(0, 2, 3, 1)
    disparity = 1.0 / (depth + eps)
    disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min() + eps)
    pixel_2d = get_screen_pixel_coord(h, w, device)
    pixel_2d_homo = lift_to_homo(pixel_2d)

    cam_int_inv = torch.inverse(cam_int)
    
    pixel_3d = torch.matmul(cam_int_inv[:, None, None, :, :], pixel_2d_homo[..., None]).squeeze(-1)
    pixel_3d = pixel_3d * depth
    vertice = pixel_3d.reshape(b, h * w, 3)
    
    faces = get_faces(h, w, device).repeat(b, 1, 1).long()
    
    attr_color = image.reshape(b, h * w, 3)
    initial_mask = get_visible_mask(disparity, device=device)
    attr_mask = initial_mask.reshape(b, h * w, 1)
    attr = torch.cat([attr_color, attr_mask], dim=-1)
    
    mesh_dict = {
        "vertice": vertice,
        "faces": faces,
        "attributes": attr,
        "size": [h, w],
    }
    return mesh_dict, disparity, initial_mask

def render_mesh(mesh_dict, cam_int, cam_ext, device):
    """
    メッシュをレンダリング
    """
    vertice = mesh_dict["vertice"]
    faces = mesh_dict["faces"]
    attributes = mesh_dict["attributes"]
    h, w = mesh_dict["size"]
    
    vertice_homo = lift_to_homo(vertice)
    
    vertice_world = torch.matmul(cam_ext.unsqueeze(1), vertice_homo[..., None]).squeeze(-1)
    vertice_depth = vertice_world[..., -1:]
    attributes = torch.cat([attributes, vertice_depth], dim=-1)
    
    vertice_world_homo = lift_to_homo(vertice_world)
    persp = get_perspective_from_intrinsic(cam_int)
    
    vertice_ndc = torch.matmul(persp.unsqueeze(1), vertice_world_homo[..., None]).squeeze(-1)
    vertice_ndc = vertice_ndc[..., :-1] / (vertice_ndc[..., -1:])
    
    vertice_ndc[..., :-1] *= -1 
    vertice_ndc[..., 0] *= w /h
    
    mesh = Meshes(vertice_ndc, faces)
    pix_to_face, _, bary_coords, _ = rasterize_meshes(mesh, (h, w), faces_per_pixel=1, blur_radius=1e-6)
    
    b, nf, _ = faces.size()
    faces = faces.reshape(b, nf * 3, 1).repeat(1, 1, 5)
    face_attributes = torch.gather(attributes, dim=1, index=faces)
    face_attributes = face_attributes.reshape(b * nf, 3, 5)
    
    output = interpolate_face_attributes(pix_to_face, bary_coords, face_attributes)
    output = output.squeeze(-2).permute(0, 3, 1, 2)
    
    render = output[:, :3]
    mask = output[:, 3:4]
    mask[mask>0.5] = 1
    mask[mask<0.5] = 0
    return render, 1 - mask

def warp_image_and_mask(image, depth, K, RT_ref_to_target, device='cuda'):
    """
    画像とエッジ検出マスクの両方をワーピング
    """
    b, c, h, w = image.shape
    
    if isinstance(K, list):
        K = torch.tensor(K, dtype=torch.float32)
    if K.dim() == 2:
        K = K.unsqueeze(0).repeat(b, 1, 1)
    
    if isinstance(RT_ref_to_target, list):
        RT_ref_to_target = torch.tensor(RT_ref_to_target, dtype=torch.float32)
    if RT_ref_to_target.dim() == 2:
        RT_ref_to_target = RT_ref_to_target.unsqueeze(0).repeat(b, 1, 1)
    
    K = K.to(device)
    RT_ref_to_target = RT_ref_to_target.to(device)
    
    K_norm = K.clone()
    K_norm[:, 0, 0] /= w
    K_norm[:, 1, 1] /= h
    K_norm[:, 0, 2] /= w
    K_norm[:, 1, 2] /= h
    mesh_dict, disparity, initial_mask = construct_mesh(image, depth, K_norm, device)
    warped_image, warped_mask = render_mesh(mesh_dict, K_norm, RT_ref_to_target, device)
    
    return warped_image, warped_mask, disparity, initial_mask

def drb_to_rdf(c2w):
    c2w_ = np.zeros_like(c2w)
    c2w_ = np.concatenate([c2w[:, [1]], c2w[:, [0]], -c2w[:, [2]], c2w[:, [3]]], axis=1)
    return c2w_

def rdf_to_luf(c2w):
    c2w_luf = c2w.copy()
    c2w_luf[:, 0] *= -1
    c2w_luf[:, 1] *= -1
    return c2w_luf

def process_single_sample(sample_dir, scene_name, sample_idx):
    """単一サンプルを処理してOCC maskを生成・保存"""
    print(f"Processing {scene_name} - Sample {sample_idx:03d}")
    
    # ファイルパスを構築
    poses_path = sample_dir / "camera_data.json"
    src_img_path = sample_dir / "source.png"
    tgt_img_path = sample_dir / "target.png"
    depth_path = sample_dir / "depth_pred.npy"
    
    # ファイルの存在確認
    if not all(p.exists() for p in [poses_path, src_img_path, tgt_img_path, depth_path]):
        print(f"  Warning: Missing files in {sample_dir}")
        return False
    
    try:
        # データ読み込み load json file
        with open(poses_path, "r") as f:
            warp_data = json.load(f)
        source_K = warp_data["source_K"]
        target_K = warp_data["target_K"]
        source_c2w = warp_data["source_c2w"]
        target_c2w = warp_data["target_c2w"]
        scale = warp_data["precomputed_scale"]
        source_K = torch.tensor(source_K, dtype=torch.float32)
        target_K = torch.tensor(target_K, dtype=torch.float32)
        precomputed_scale = torch.tensor(scale, dtype=torch.float32)
        # source_c2w = rdf_to_luf(np.array(source_c2w))
        source_c2w = torch.tensor(source_c2w, dtype=torch.float32)
        # target_c2w = rdf_to_luf(np.array(target_c2w))
        target_c2w = torch.tensor(target_c2w, dtype=torch.float32)
        dst_to_src_mat = torch.mm(torch.linalg.inv(source_c2w), target_c2w)
        dst_to_src_mat[:3, 3] *= precomputed_scale

        width = source_K[0, 2] * 2
        height = source_K[1, 2] * 2
                
        # 画像とデプス読み込み
        image = Image.open(src_img_path).convert('RGB')
        tgt_image_pil = Image.open(tgt_img_path).convert('RGB')
        depth = np.load(depth_path)  # shape: (1, H, W)
        
        # NumPy配列に変換
        image_array = np.array(image).astype(np.float32) / 255.0        
        # テンソルに変換
        image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0)  # [1,3,h,w]
        depth_tensor = torch.tensor(depth).unsqueeze(0)  # [1,1,h,w]
        
        # カメラパラメータ
        RT_ref_to_target = torch.linalg.inv(dst_to_src_mat)[:3, :].unsqueeze(0)
        
        # ワーピング実行
        warped_image, warped_mask, disparity, initial_mask = warp_image_and_mask(
            image_tensor, depth_tensor, source_K, RT_ref_to_target, device
        )
        
        # マスクを保存
        warped_mask_np = warped_mask.squeeze().cpu().numpy()
        mask_uint8 = (warped_mask_np * 255).astype(np.uint8)
        
        output_path = sample_dir / "occ_mask.png"
        cv2.imwrite(str(output_path), mask_uint8)
        
        print(f"  Saved: {output_path}")

        # tgt_img_cv = cv2.cvtColor(np.array(tgt_image_pil), cv2.COLOR_RGB2BGR)
        # red_mask = np.zeros_like(tgt_img_cv)
        # red_mask[mask_uint8 == 255] = [0, 0, 255] # BGRなので赤は[0,0,255]
        # overlayed_img = cv2.addWeighted(tgt_img_cv, 0.7, red_mask, 0.3, 0)
        # overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)

        # # 可視化関数を呼び出し
        # visualize_results(
        #     src_img=image_tensor,
        #     disparity_map=disparity,
        #     initial_mask=initial_mask,
        #     tgt_img=tgt_image_pil,
        #     warped_mask=warped_mask,
        #     warped_src_img=warped_image,
        #     overlayed_img=overlayed_img,
        #     output_dir=sample_dir / "visualizations", # 保存先ディレクトリ
        #     sample_idx=sample_idx
        # )
        return True
        
    except Exception as e:
        print(f"  Error processing {sample_dir}: {str(e)}")
        return False

def main():
    """メイン処理：全シーン・全サンプルを処理"""
    # データセットルートパス
    dataset_root = Path("/home/rintoyagawa/ssd2/Code/simple-adampi-ueda/data/re10k_eval")
    
    # シーン名のリスト
    scenes = ["test_5_frame"]#, "test_10_frame", "test_random"]
    
    scene_ranges = {
        "test_5_frame": (0, 2),    
        # "test_10_frame": (0, 2),     
        # "test_random": (0, 2),      
    }

    total_processed = 0
    total_success = 0
    
    for scene_name, (start_idx, end_idx) in scene_ranges.items():
        scene_dir = dataset_root / scene_name
        
        if not scene_dir.exists():
            print(f"Warning: Scene directory {scene_dir} does not exist")
            continue
            
        print(f"\n=== Processing Scene: {scene_name} ===")
        test_dir = scene_dir
        
        if not test_dir.exists():
            print(f"Warning: Test directory {test_dir} does not exist")
            continue
        
        # 各シーンの15サンプルを処理
        for sample_idx in range(start_idx, end_idx + 1):
            sample_dir = test_dir / f"{sample_idx:06d}"
            
            if sample_dir.exists():
                total_processed += 1
                if process_single_sample(sample_dir, scene_name, sample_idx):
                    total_success += 1
            else:
                print(f"Warning: Sample directory {sample_dir} does not exist")
    
    print(f"\n=== Processing Complete ===")
    print(f"Total samples processed: {total_processed}")
    print(f"Successful: {total_success}")
    print(f"Failed: {total_processed - total_success}")

if __name__ == "__main__":
    main()