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

eps = 1e-1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def construct_mesh(image, disparity, cam_int, near_depth, far_depth, device):
    """
    画像とデプスからメッシュを構築
    """
    b, c, h, w = image.shape
    image = image.to(device)
    disparity = disparity.to(device)
    cam_int = cam_int.to(device)
    
    disparity = disparity.permute(0, 2, 3, 1)
    image = image.permute(0, 2, 3, 1)

    pixel_2d = get_screen_pixel_coord(h, w, device)
    pixel_2d_homo = lift_to_homo(pixel_2d)

    cam_int_inv = torch.inverse(cam_int)
    
    pixel_3d = torch.matmul(cam_int_inv[:, None, None, :, :], pixel_2d_homo[..., None]).squeeze(-1)
    scaled_disparity = disparity * ((1 / near_depth) - (1 / far_depth)) + (1 / far_depth)
    scaled_depth = torch.reciprocal(scaled_disparity) 

    pixel_3d = pixel_3d * scaled_depth
    vertice = pixel_3d.reshape(b, h * w, 3)
    
    faces = get_faces(h, w, device).repeat(b, 1, 1).long()
    
    attr_color = image.reshape(b, h * w, 3)
    attr_mask = get_visible_mask(disparity, device=device).reshape(b, h * w, 1)
    attr = torch.cat([attr_color, attr_mask], dim=-1)
    
    mesh_dict = {
        "vertice": vertice,
        "faces": faces,
        "attributes": attr,
        "size": [h, w],
    }
    return mesh_dict

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

def warp_image_and_mask(image, depth, K, RT_ref_to_target, near_depth, far_depth, device='cuda'):
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
    
    mesh_dict = construct_mesh(image, depth, K, near_depth, far_depth, device)
    warped_image, warped_mask = render_mesh(mesh_dict, K, RT_ref_to_target, device)
    
    return warped_image, warped_mask

def drb_to_rdf(c2w):
    c2w_ = np.zeros_like(c2w)
    c2w_ = np.concatenate([c2w[:, [1]], c2w[:, [0]], -c2w[:, [2]], c2w[:, [3]]], axis=1)
    return c2w_

def get_cam_intrinsics(poses: Tensor):
    """Get camera intrinsics from poses."""
    intrinsics = []
    for i in range(poses.shape[-1]):
        height = poses[0, -1, i]
        width = poses[1, -1, i] 
        focal_length = poses[2, -1, i]
        f_x = f_y = focal_length
        c_x = width / 2.0
        c_y = height / 2.0
        
        cam_int = torch.tensor([
            [f_x / width, 0, c_x / width],
            [0, f_y / height, c_y / height],
            [0, 0, 1],
        ], dtype=torch.float32)
        intrinsics.append(cam_int)
    
    src_int = intrinsics[0]
    dst_int = intrinsics[1]
    return src_int, dst_int

def get_cam_int_and_cam_ext(poses: torch.Tensor):
    """LLFFフォーマットのposesからカメラパラメータを抽出"""
    bottom = torch.tensor([[0, 0, 0, 1.0]])
    src_int, dst_int = get_cam_intrinsics(poses)
    
    src_cam2wor = torch.cat((poses[:, :-1, 0], bottom), 0)
    dst_cam2wor = torch.cat((poses[:, :-1, 1], bottom), 0)
    
    src_cam2wor = torch.tensor(drb_to_rdf(src_cam2wor.numpy()), dtype=torch.float32)
    dst_cam2wor = torch.tensor(drb_to_rdf(dst_cam2wor.numpy()), dtype=torch.float32)
    
    dst_to_src_mat = torch.mm(torch.linalg.inv(src_cam2wor), dst_cam2wor)
    
    return src_int, dst_int, dst_to_src_mat

def process_single_sample(sample_dir, scene_name, sample_idx):
    """単一サンプルを処理してOCC maskを生成・保存"""
    print(f"Processing {scene_name} - Sample {sample_idx:03d}")
    
    # ファイルパスを構築
    poses_path = sample_dir / "poses_bounds.npy"
    src_img_path = sample_dir / "aif.png"
    tgt_img_path = sample_dir / "aif_tgt.png"
    depth_path = sample_dir / "depth.png"
    
    # ファイルの存在確認
    if not all(p.exists() for p in [poses_path, src_img_path, tgt_img_path, depth_path]):
        print(f"  Warning: Missing files in {sample_dir}")
        return False
    
    try:
        # データ読み込み
        poses_arr = np.load(poses_path)
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        poses = torch.tensor(poses).float()
        bds = poses_arr[:, -2:].reshape([-1,2]).transpose([1, 0])
        bds = torch.tensor(bds).float()
        
        height = poses[0, -1, 0].item()
        width = poses[1, -1, 0].item()
        
        src_int, dst_int, dst_to_src_mat = get_cam_int_and_cam_ext(poses)
        
        # 画像とデプス読み込み
        image = Image.open(src_img_path).convert('RGB')
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        
        # NumPy配列に変換
        image_array = np.array(image).astype(np.float32) / 255.0
        depth_array = np.array(depth_img).astype(np.float32) / 255.0
        
        # テンソルに変換
        image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0)  # [1,3,h,w]
        depth_tensor = torch.tensor(depth_array).unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
        
        # カメラパラメータ
        near_depth, far_depth = bds[0,0], bds[1,0]
        RT_ref_to_target = torch.linalg.inv(dst_to_src_mat)[:3, :].unsqueeze(0)
        
        # ワーピング実行
        warped_image, warped_mask = warp_image_and_mask(
            image_tensor, depth_tensor, src_int, RT_ref_to_target, near_depth, far_depth, device
        )
        
        # マスクを保存
        warped_mask_np = warped_mask.squeeze().cpu().numpy()
        mask_uint8 = (warped_mask_np * 255).astype(np.uint8)
        
        output_path = sample_dir / "occ_mask.png"
        cv2.imwrite(str(output_path), mask_uint8)
        
        print(f"  Saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"  Error processing {sample_dir}: {str(e)}")
        return False

def main():
    """メイン処理：全シーン・全サンプルを処理"""
    # データセットルートパス
    dataset_root = Path("/home/rintoyagawa/ssd2/Code/simple-adampi-ueda/data/llff_eval-fixedsrc")
    
    # シーン名のリスト
    scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
    
    scene_ranges = {
        "fern": (253, 271),    
        "flower": (142, 174),     
        "fortress": (40, 80),  
        "horns": (81, 141),    
        "leaves": (272, 296),  
        "orchids": (229, 252),   
        "room": (0, 39),      
        "trex": (175, 228)       
    }

    total_processed = 0
    total_success = 0
    
    for scene_name, (start_idx, end_idx) in scene_ranges.items():
        scene_dir = dataset_root / scene_name
        
        if not scene_dir.exists():
            print(f"Warning: Scene directory {scene_dir} does not exist")
            continue
            
        print(f"\n=== Processing Scene: {scene_name} ===")
        test_dir = scene_dir / "test"
        
        if not test_dir.exists():
            print(f"Warning: Test directory {test_dir} does not exist")
            continue
        
        # 各シーンの15サンプルを処理
        for sample_idx in range(start_idx, end_idx + 1):
            sample_dir = test_dir / f"{sample_idx:03d}"
            
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