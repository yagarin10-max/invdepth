# src/datasets/base_dataset.py

from torch.utils.data import Dataset
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import json
import abc

class BaseDataset(Dataset):
    """すべてのデータセットの基底クラス"""
    def __init__(
        self,
        data_root: str,
        transform=None,
        **kwargs
    ):
        """
        Args:
            data_root: データセットのルートディレクトリ
            transform: 画像の前処理
        """
        self.data_root = Path(data_root)
        self.transform = transform
        
        # データセットの基本的なバリデーション
        self._validate_dataset_structure()

    @abc.abstractmethod
    def __len__(self) -> int:
        """データセットの長さを返す"""
        pass

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """指定されたインデックスのデータを返す"""
        pass

    def _validate_dataset_structure(self) -> None:
        """データセットの構造を検証"""
        required_dirs = ['images', 'depths']
        # required_files = ['scenes.json']
        
        for dir_name in required_dirs:
            dir_path = self.data_root / dir_name
            if not dir_path.exists():
                raise RuntimeError(f"Required directory '{dir_name}' not found in {self.data_root}")
        
    def _load_json(self, file_name: str) -> dict:
        """JSONファイルを読み込む"""
        file_path = self.data_root / file_name
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON format in {file_path}")

class BaseMultiViewDataset(BaseDataset):
    """マルチビューデータセットの基底クラス"""
    def __init__(
        self,
        data_root: str,
        num_source_views: int = 2,
        img_height: int = 256,
        img_width: int = 256,
        transform=None,
        **kwargs
    ):
        """
        Args:
            data_root: データセットのルートディレクトリ
            num_source_views: ソース画像の数
            img_height: 画像の高さ
            img_width: 画像の幅
            transform: 画像の前処理
        """
        super().__init__(data_root, transform)
        
        self.num_source_views = num_source_views
        self.img_height = img_height
        self.img_width = img_width
        
        # 基本的なデータの読み込み
        self.scenes = self._load_scenes()
        
    def _load_scenes(self) -> List[Dict[str, Any]]:
        """シーン情報を読み込む"""
        # .pngファイルを検索
        image_files = sorted(list((self.data_root / 'images').glob('*.png')))
        image_files = [f.name for f in image_files]
        
        # デプスファイルのパス（固定）
        depth_file = "depth.exr"
        if not (self.data_root / 'depths' / depth_file).exists():
            raise RuntimeError(f"Depth file {depth_file} not found in {self.data_root / 'depths'}")
        
        scenes = []
        required_frames = 1 + self.num_source_views
        
        if len(image_files) >= required_frames:
            # 最初の画像を参照画像として使用
            ref_image = image_files[0]
            source_images = image_files[1:1+self.num_source_views]
            
            scene = {
                'reference_image': ref_image,
                'source_images': source_images,
                'depth_file': depth_file  # 固定のデプスファイル名を使用
            }
            scenes.append(scene)
        
        if not scenes:
            raise RuntimeError(
                f"No valid scenes found. Each scene requires:\n"
                f"- 1 reference image (found: {image_files[0] if image_files else 'none'})\n"
                f"- {self.num_source_views} source images (found: {len(image_files)-1 if len(image_files)>1 else 0})\n"
                f"- depth.exr file (found: {'yes' if (self.data_root / 'depths' / depth_file).exists() else 'no'})"
            )
        
        return scenes
    
    def __len__(self) -> int:
        """データセットの長さを返す"""
        return len(self.scenes)

    @abc.abstractmethod
    def load_depth(self, path: Path) -> np.ndarray:
        """デプス画像を読み込む抽象メソッド"""
        pass

    @abc.abstractmethod
    def load_image(self, path: Path) -> np.ndarray:
        """画像を読み込む抽象メソッド"""
        pass

    @abc.abstractmethod
    def load_camera_params(self) -> Dict[str, Any]:
        """カメラパラメータを読み込む抽象メソッド"""
        pass