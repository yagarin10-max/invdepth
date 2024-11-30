# src/datasets/multiview_dataset.py

import numpy as np
import cv2
import OpenEXR
import torch
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import math
from .base_dataset import BaseMultiViewDataset

class MultiViewDepthDataset(BaseMultiViewDataset):
    def __init__(
        self,
        data_root: str,
        num_source_views: int = 2,
        img_height: int = 256,
        img_width: int = 256,
        focal_mm: float = 55,
        sensor_width_mm: float = 24,
        z_min: float = 0.4,
        z_max: float = 3.5,
        transform=None
    ):
        """
        マルチビューデプスデータセットの初期化
        Args:
            data_root: データセットのルートディレクトリ
            num_source_views: ソース画像の数
            img_height: 画像の高さ
            img_width: 画像の幅
            focal_mm: 焦点距離(mm)
            sensor_width_mm: センサー幅(mm)
            z_min: デプスの最小値
            z_max: デプスの最大値
            transform: 画像の前処理
        """
        super().__init__(
            data_root=data_root,
            num_source_views=num_source_views,
            img_height=img_height,
            img_width=img_width,
            transform=transform
        )
        
        self.focal_mm = focal_mm
        self.sensor_width_mm = sensor_width_mm
        self.z_min = z_min
        self.z_max = z_max
        
        # 内部・外部パラメータの設定
        self.K = self._compute_intrinsic_matrix()
        self.camera_params = self.load_camera_params()

    def _compute_intrinsic_matrix(self) -> np.ndarray:
        """カメラの内部パラメータ行列を計算"""
        focal_px = (self.img_width * self.focal_mm) / self.sensor_width_mm
        cam_int = np.array([
            [focal_px/ self.img_width, 0, 0.5],
            [0, focal_px/ self.img_width, 0.5],
            [0, 0, 1]
        ])
        # cam_int = np.array([
        #     [focal_px, 0, self.img_width/2],
        #     [0, focal_px, self.img_height/2],
        #     [0, 0, 1]
        # ])
        return  cam_int

    @staticmethod
    def _euler_to_rotation_matrix(euler_angles: List[float]) -> np.ndarray:
        """
        オイラー角（度数）から回転行列を計算
        Args:
            euler_angles: [rx, ry, rz] 形式のオイラー角（度）
        """
        def deg_to_rad(deg):
            return deg * math.pi / 180.0
        
        rx, ry, rz = map(deg_to_rad, euler_angles)
        
        # X軸周りの回転
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        # Y軸周りの回転
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        # Z軸周りの回転
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # 回転の合成（順序：Z → Y → X）
        R = Rx @ Ry @ Rz
        return R

    def load_camera_params(self) -> Dict[str, Any]:
        """カメラパラメータの読み込みと変換行列の計算"""
        params = self._load_json('camera_params.json')
        camera_params = {}
        
        for image_name, param in params.items():
            R = self._euler_to_rotation_matrix(param['rotation'])
            t = np.array(param['location']).reshape(3, 1)
            
            # 変換行列の作成
            transform = np.eye(4)
            transform[:3, :3] = R
            transform[:3, 3] = t.flatten()
            
            camera_params[image_name] = {
                'location': param['location'],
                'rotation': param['rotation'],
                'transform': transform
            }
            
        return camera_params

    def load_depth(self, path: Path) -> np.ndarray:
        """
        EXRファイルからデプスマップを読み込む
        Args:
            path: EXRファイルのパス
        Returns:
            depth: 正規化されたデプス画像
        """
        exr_file = OpenEXR.InputFile(str(path))
        dw = exr_file.header()["dataWindow"]
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        
        channel_keys = sorted(exr_file.header()["channels"].keys())
        
        # RGBまたはVチャンネルからデプスを読み込む
        if channel_keys == sorted(["R", "G", "B"]):
            v, _, _ = exr_file.channels("RGB")
            depth = np.frombuffer(v, dtype=np.float16)
        elif channel_keys == sorted(["V"]):
            v = exr_file.channels("V")[0]
            depth = np.frombuffer(v, dtype=np.float32).astype(np.float16)
        else:
            raise ValueError(f"Unexpected EXR channels: {channel_keys}")

        depth = depth.reshape(size[1], size[0])
        
        # デプス値のクリップと正規化
        if depth.min() < self.z_min or depth.max() > self.z_max:
            print(
                f"Depth values out of range [{self.z_min}, {self.z_max}]: "
                f"[{depth.min()}, {depth.max()}]. Clipping."
            )
        depth = np.clip(depth, self.z_min, self.z_max)
        depth = (depth - self.z_min) / (self.z_max - self.z_min)
        
        # リサイズが必要な場合
        if depth.shape != (self.img_height, self.img_width):
            depth = cv2.resize(
                depth, 
                (self.img_width, self.img_height), 
                interpolation=cv2.INTER_LINEAR
            )
            
        return depth

    def load_image(self, path: Path) -> np.ndarray:
        """
        画像を読み込む
        Args:
            path: 画像ファイルのパス
        Returns:
            image: RGB画像 (H, W, 3)
        """
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image.shape[:2] != (self.img_height, self.img_width):
            image = cv2.resize(
                image,
                (self.img_width, self.img_height),
                interpolation=cv2.INTER_LINEAR
            )
        
        return image

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        データセットからアイテムを取得
        Args:
            idx: インデックス
        Returns:
            data: 以下を含む辞書
                - ref_image: 参照画像 (3, H, W)
                - ref_depth: 正規化された参照画像のデプス (H, W)
                - src_images: ソース画像のリスト (N, 3, H, W)
                - ref_transform: 参照画像の変換行列 (4, 4)
                - src_transforms: ソース画像の変換行列のリスト (N, 4, 4)
                - K: カメラ内部パラメータ行列 (3, 3)
        """
        scene = self.scenes[idx]
        ref_image_name = scene['reference_image']
        
        # 参照画像とデプスの読み込み
        ref_image = self.load_image(self.data_root / 'images' / ref_image_name)
        ref_depth = self.load_depth(
            self.data_root / 'depths' / scene['depth_file']
        )
        
        # ソース画像とその変換行列の読み込み
        src_images = []
        src_transforms = []
        
        for src_img_name in scene['source_images']:
            src_image = self.load_image(self.data_root / 'images' / src_img_name)
            if self.transform:
                src_image = self.transform(src_image)
            src_images.append(src_image)
            src_transforms.append(self.camera_params[src_img_name]['transform'])
        
        # 参照画像の変換
        if self.transform:
            ref_image = self.transform(ref_image)
        
        # numpy配列をtensorに変換
        data = {
            'ref_image': torch.from_numpy(ref_image).float().permute(2, 0, 1) / 255.0,
            'ref_depth': torch.from_numpy(ref_depth).float(),
            'src_images': torch.stack([
                torch.from_numpy(img).float().permute(2, 0, 1) / 255.0 
                for img in src_images
            ]),
            'ref_transform': torch.from_numpy(
                self.camera_params[ref_image_name]['transform']
            ).float(),
            'src_transforms': torch.stack([
                torch.from_numpy(transform).float() 
                for transform in src_transforms
            ]),
            'K': torch.from_numpy(self.K).float()
        }
        
        return data

# 使用例
if __name__ == "__main__":
    import os
    
    # データセットの初期化
    data_root = "/home/rintoyagawa/ssd2/Code/invdepth/data"
    image_dir = Path(data_root) / 'images'
    depth_dir = Path(data_root) / 'depths'
    
    print("Checking dataset structure...")
    print(f"Data root: {data_root}")
    print(f"Image directory: {image_dir}")
    print(f"Depth directory: {depth_dir}")

    # ファイル数の確認
    image_files = sorted(list(image_dir.glob('*.png')))
    depth_file = depth_dir / 'depth.exr'
    
    print(f"\nFound {len(image_files)} images in images directory:")
    for f in image_files:
        print(f"  - {f.name}")
    
    print(f"\nDepth file {'exists' if depth_file.exists() else 'does not exist'}: {depth_file}")
    
    try:
        dataset = MultiViewDepthDataset(
            data_root=data_root,
            num_source_views=4,  # より安全な値に設定
            img_height=256,
            img_width=256,
            focal_mm=55,
            sensor_width_mm=24,
            z_min=0.5,
            z_max=5.0
        )
        
        print(f"\nSuccessfully created dataset with {len(dataset)} scenes")
        
        # サンプルデータの確認
        sample = dataset[0]
        print("\nFirst scene information:")
        print("Reference image:", dataset.scenes[0]['reference_image'])
        print("Source images:", dataset.scenes[0]['source_images'])
        print("Depth file:", dataset.scenes[0]['depth_file'])
        
    except Exception as e:
        print(f"Error initializing dataset: {str(e)}")
        raise