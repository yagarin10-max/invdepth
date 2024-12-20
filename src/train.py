# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import skimage.metrics
import os
from matplotlib import pyplot as plt
from utils.warp import warp_with_inverse_depth_mesh, compute_loss
from utils.metrics import TVLoss
from models.siren import SIREN
from datasets.multiview_dataset import MultiViewDepthDataset

class InverseDepthTrainer:
    def __init__(
        self,
        config: dict,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.config = config
        self.device = device
        
        # ネットワークの初期化
        self.net = SIREN(
            in_dim=2,
            hidden_dim=config['hidden_dim'],
            hidden_layers=config['hidden_layers'],
            out_dim=1,
            outermost_linear=True
        ).to(device)
        
        # 最適化器の設定
        self.optimizer = optim.AdamW(
            self.net.parameters(),
            lr=config['learning_rate']
        )
        
        # ロス関数の設定
        self.tv_loss = TVLoss().to(device)
        
        # データセットとデータローダーの設定
        self.train_dataset = MultiViewDepthDataset(
            data_root=config['data_root'],
            num_source_views=config['num_source_views'],
            img_height=config['img_height'],
            img_width=config['img_width'],
            focal_mm=config['focal_length_mm'],
            sensor_width_mm=config['sensor_width_mm']
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        )

    def train_epoch(self, use_patch_loss: bool = True, patch_size: int = 3):
        """1エポックの学習を実行"""
        self.net.train()
        total_loss = 0
        psnr_values = []

        for batch in tqdm(self.train_loader):
            # バッチデータの取り出し
            ref_image = batch['ref_image'].to(self.device)
            src_images = batch['src_images'].to(self.device)
            K = batch['K'].to(self.device)
            ref_depth = batch['ref_depth'].to(self.device)
            relative_transforms = batch['relative_transforms'].to(self.device)
            # print(src_transforms)
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

            # 逆深度の予測
            pred_inverse_depth = self.net(coords)
            pred_inverse_depth = torch.clamp(pred_inverse_depth, -1, 1)
            pred_inverse_depth = pred_inverse_depth.reshape(b, 1, h, w)
            pred_inverse_depth = (pred_inverse_depth + 1) / 2.0  # 0~1に正規化

            # マルチビューの再構成損失の計算
            total_recon_loss = 0
            batch_psnr = 0
            for i in range(src_images.size(1)):  # ソース画像の数だけループ
                pred_img, warp_inv, mask = warp_with_inverse_depth_mesh(
                    ref_image,
                    pred_inverse_depth,
                    self.device,
                    K,
                    relative_transforms[:, i][:,:3,:],
                    ref_depth
                )
                recon_loss = compute_loss(
                    pred_img,
                    src_images[:, i],
                    mask,
                    use_patch_loss,
                    patch_size
                )
                total_recon_loss += recon_loss
                # PSNRの計算
                with torch.no_grad():
                    psnr = skimage.metrics.peak_signal_noise_ratio(
                        src_images[:, i].cpu().numpy(),
                        pred_img.detach().cpu().numpy()
                    )
                    batch_psnr += psnr
                    
            batch_psnr /= src_images.size(1)
            psnr_values.append(batch_psnr)
            # Total Variation正則化
            tv_loss = self.tv_loss(pred_inverse_depth)
            # 全体の損失
            loss = total_recon_loss + tv_loss * self.config['tv_weight']

            # 最適化ステップ
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()


        return total_loss / len(self.train_loader), sum(psnr_values) / len(psnr_values)

    def train(self):
        """学習の実行"""
        os.makedirs(self.config['save_dir'], exist_ok=True)

        best_psnr = 0
        psnr_history = []
        loss_history = []
        for epoch in range(self.config['num_epochs']):
            loss, psnr = self.train_epoch(
                use_patch_loss=self.config['use_patch_loss'],
                patch_size=self.config['patch_size']
            )

            psnr_history.append(psnr)
            loss_history.append(loss)
            
            print(f'Epoch {epoch + 1}/{self.config["num_epochs"]}')
            print(f'Loss: {loss:.4f}, PSNR: {psnr:.2f}')

            # モデルの保存
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(
                    self.net.state_dict(),
                    f"{self.config['save_dir']}/best_model.pth"
                )
        plt.figure(figsize=(10,5))
        plt.plot(loss_history, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{self.config['results_dir']}/loss_history.png")
        plt.close()
        plt.figure(figsize=(10,5))
        plt.plot(psnr_history, label='PSNR')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.legend()
        plt.savefig(f"{self.config['results_dir']}/psnr_history.png")
        plt.close()


if __name__ == "__main__":
    # 設定
    config = {
        'data_root': '/home/rintoyagawa/ssd2/Code/invdepth/data',
        'hidden_dim': 256,
        'hidden_layers': 2,
        'learning_rate': 1e-4,
        'num_epochs': 3000,
        'batch_size': 1,
        'num_workers': 4,
        'num_source_views': 3,
        'img_height': 256,
        'img_width': 256,
        'focal_length_mm': 55,
        'sensor_width_mm': 24,
        'use_patch_loss': False,
        'patch_size': 1,
        'tv_weight': 15e-3,
        'save_dir': '/home/rintoyagawa/ssd2/Code/invdepth/checkpoints',
        'results_dir': '/home/rintoyagawa/ssd2/Code/invdepth/results'
    }

    trainer = InverseDepthTrainer(config)
    trainer.train()