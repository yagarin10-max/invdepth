import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from models.siren import SIREN
from datasets.multiview_dataset import MultiViewDepthDataset
from torch.utils.data import DataLoader
from utils.warp import warp_with_inverse_depth_mesh

class InverseDepthInference:
    def __init__(
        self,
        checkpoint_path: str,
        config: dict,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.config = config
        self.device = device
        
        # モデルの初期化と重みの読み込み
        self.net = SIREN(
            in_dim=2,
            hidden_dim=config['hidden_dim'],
            hidden_layers=config['hidden_layers'],
            out_dim=1,
            outermost_linear=True
        ).to(device)
        
        self.net.load_state_dict(torch.load(checkpoint_path))
        self.net.eval()
        
        # データセットとデータローダーの設定
        self.dataset = MultiViewDepthDataset(
            data_root=config['data_root'],
            num_source_views=config['num_source_views'],
            img_height=config['img_height'],
            img_width=config['img_width'],
            focal_mm=config['focal_length_mm'],
            sensor_width_mm=config['sensor_width_mm']
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

    def evaluate_depth(self, pred_inverse_depth, gt_depth, min_depth=0.5, max_depth=6.0, epsilon=1e-4):
        """Inverse DepthとGround Truth Depthの評価を行う"""
        # Ground Truth DepthをInverse Depthに変換
        gt_depth_clipped = torch.clamp(gt_depth, min_depth, max_depth)
        gt_inverse_depth = (1/gt_depth_clipped - 1/max_depth) / (1/min_depth - 1/max_depth)
        
        # 有効なピクセルのマスクを作成
        valid_mask = (gt_depth >= min_depth) & (gt_depth <= max_depth)
        
        metrics = {}
        # MAE (Mean Absolute Error)
        metrics['mae'] = torch.mean(torch.abs(pred_inverse_depth[valid_mask] - gt_inverse_depth[valid_mask]))
        
        # RMSE (Root Mean Square Error)
        metrics['rmse'] = torch.sqrt(torch.mean((pred_inverse_depth[valid_mask] - gt_inverse_depth[valid_mask])**2))
        
        diff = torch.log(pred_inverse_depth[valid_mask].clamp(min=epsilon)) - torch.log(gt_inverse_depth[valid_mask].clamp(min=epsilon))

        # 負の値をsqrtに入れないようにする
        var = torch.mean(diff**2) - torch.mean(diff)**2
        metrics['scale_inv'] = torch.sqrt(torch.clamp(var, min=0.0))
        
        # Accuracy metrics (δ < threshold)
        ratios = torch.max(
            (pred_inverse_depth[valid_mask] / gt_inverse_depth[valid_mask].clamp(min=epsilon)),
            (gt_inverse_depth[valid_mask] / pred_inverse_depth[valid_mask].clamp(min=epsilon))
        )
        
        metrics['delta1'] = torch.mean((ratios < 1.25).float())
        metrics['delta2'] = torch.mean((ratios < 1.25**2).float())
        metrics['delta3'] = torch.mean((ratios < 1.25**3).float())
        
        return metrics

    def infer_and_visualize(self, save_dir='inference_results', idx=None):
        """指定されたインデックスのデータに対して推論と可視化を行う"""
        os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            if idx is not None:
                data = self.dataset[idx]
                # バッチ次元を追加
                for k, v in data.items():
                    if torch.is_tensor(v):
                        data[k] = v.unsqueeze(0)
            else:
                data = next(iter(self.dataloader))
            
            ref_image = data['ref_image'].to(self.device)
            src_images = data['src_images'].to(self.device)
            K = data['K'].to(self.device)
            relative_transforms = data['relative_transforms'].to(self.device)
            
            gt_depth = data['ref_depth'].to(self.device)
            
            b, c, h, w = ref_image.shape
            
            # 座標グリッドの生成
            x_loc, y_loc = torch.meshgrid(
                torch.linspace(-1, 1, steps=w),
                torch.linspace(-1, 1, steps=h),
                indexing="xy"
            )
            coords = torch.cat(
                (x_loc.reshape(-1, 1), y_loc.reshape(-1, 1)),
                dim=1
            ).to(self.device)
            
            # 深度の予測
            pred_inverse_depth = self.net(coords)
            pred_inverse_depth = torch.clamp(pred_inverse_depth, -1, 1)
            pred_inverse_depth = pred_inverse_depth.reshape(b, 1, h, w)
            pred_inverse_depth = (pred_inverse_depth + 1) / 2.0
            
            # Ground Truth Depthが利用可能な場合は評価を実行
            if 'ref_depth' in data:
                metrics = self.evaluate_depth(pred_inverse_depth[0, 0], gt_depth[0])
                
                # 評価結果の保存
                print("\nDepth Estimation Metrics:")
                with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
                    for metric_name, value in metrics.items():
                        print(f"{metric_name}: {value.item():.4f}")
                        f.write(f"{metric_name}: {value.item():.4f}\n")
            
            min_depth = 0.5
            max_depth = 6.0
            
            # 結果の可視化
            plt.figure(figsize=(15, 5))
            
            # Inverse Depthマップの可視化
            plt.subplot(131)
            plt.imshow(pred_inverse_depth[0, 0].cpu().numpy(), cmap='plasma')
            plt.colorbar()
            plt.title('Predicted Inverse Depth')
            plt.axis('off')
            
            plt.subplot(132)
            gt_inverse_depth = (1/torch.clamp(gt_depth, min_depth, max_depth) - 1/max_depth) / (1/min_depth - 1/max_depth)
            plt.imshow(gt_inverse_depth[0].cpu().numpy(), cmap='plasma')
            plt.colorbar()
            plt.title('Ground Truth Inverse Depth')
            plt.axis('off')
            
            # Ground Truth Depthの可視化（利用可能な場合）
            if 'ref_depth' in data:
                plt.subplot(133)
                plt.imshow(gt_depth[0].cpu().numpy(), cmap='plasma')
                plt.colorbar()
                plt.title('Ground Truth Depth')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'depth_visualization.png'))
            plt.close()
            
            return pred_inverse_depth

def main():
    config = {
        'data_root': '/home/rintoyagawa/ssd2/Code/invdepth/data',
        'hidden_dim': 256,
        'hidden_layers': 2,
        'num_source_views': 10,
        'img_height': 256,
        'img_width': 256,
        'focal_length_mm': 55,
        'sensor_width_mm': 24,
    }
    
    checkpoint_path = '/home/rintoyagawa/ssd2/Code/invdepth/checkpoints/best_model.pth'
    save_dir = './invdepth/inference_results'
    
    inferencer = InverseDepthInference(checkpoint_path, config)
    
    # データセットの特定のインデックスで推論を実行
    pred_inverse_depth = inferencer.infer_and_visualize(save_dir=save_dir, idx=0)
    print(f"Results saved to {save_dir}")

if __name__ == "__main__":
    main()