import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import skimage.metrics
import os
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from utils.metrics import TVLoss
from models.siren import SIREN
from utils.warp import warp_with_inverse_depth_mesh, compute_loss
from datasets.multiview_dataset import MultiViewDepthDataset
import wandb 

class DepthAnythingTrainer:
    def __init__(
        self,
        config: dict,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.config = config
        self.device = device

        wandb.init(
            project="inverse-depth-estimation",
            config=config,
            name=config.get('run_name', 'inverse-depth-run')
        )
        
        self.net = SIREN(
            in_dim=1,
            hidden_dim=config['hidden_dim'],
            hidden_layers=config['hidden_layers'],
            out_dim=1,
            outermost_linear=True
        ).to(device)

        wandb.watch(self.net)

        # 最適化器の設定
        self.optimizer = optim.AdamW(
            self.net.parameters(),
            lr=config['learning_rate']
        )

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
        total_recon_loss = 0
        total_tv_loss = 0
        psnr_values = []

        for batch in tqdm(self.train_loader):
            # 既存のバッチ処理コード...
            ref_image = batch['ref_image'].to(self.device)
            src_images = batch['src_images'].to(self.device)
            K = batch['K'].to(self.device)
            ref_depth = batch['ref_depth'].to(self.device)
            initial_depth = batch['initial_depth'].to(self.device)
            relative_transforms = batch['relative_transforms'].to(self.device)
            
            b, c, h, w = ref_image.shape
            initial_depth_flat = initial_depth.squeeze().reshape(-1, 1)
            pred_inverse_depth = self.net(initial_depth_flat)

            pred_inverse_depth = torch.sigmoid(pred_inverse_depth)
            # pred_inverse_depth = torch.clamp(pred_inverse_depth, -1, 1)
            pred_inverse_depth = pred_inverse_depth.reshape(b, 1, h, w)
            # pred_inverse_depth = (pred_inverse_depth + 1) / 2.0

            # 損失計算
            batch_recon_loss = 0
            batch_psnr = 0
            for i in range(src_images.size(1)):
                pred_img, warp_inv, mask = warp_with_inverse_depth_mesh(
                    ref_image,
                    pred_inverse_depth,
                    self.device,
                    K,
                    relative_transforms[:, i][:,:3,:]
                )
                recon_loss = compute_loss(
                    pred_img,
                    src_images[:, i],
                    mask,
                    use_patch_loss,
                    patch_size
                )
                batch_recon_loss += recon_loss
                
                # PSNR計算
                with torch.no_grad():
                    psnr = skimage.metrics.peak_signal_noise_ratio(
                        src_images[:, i].cpu().numpy(),
                        pred_img.detach().cpu().numpy()
                    )
                    batch_psnr += psnr
            
            batch_psnr /= src_images.size(1)
            psnr_values.append(batch_psnr)
            
            # TV Loss
            tv_loss = self.tv_loss(pred_inverse_depth)
            
            # 全体の損失
            loss = batch_recon_loss + tv_loss * self.config['tv_weight']

            # 最適化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 損失の累積
            total_loss += loss.item()
            total_recon_loss += batch_recon_loss.item()
            total_tv_loss += tv_loss.item()

            # バッチごとのログ
            wandb.log({
                "batch_loss": loss.item(),
                "batch_recon_loss": batch_recon_loss.item(),
                "batch_tv_loss": tv_loss.item(),
                "batch_psnr": batch_psnr
            })

        # エポックの平均を計算
        avg_loss = total_loss / len(self.train_loader)
        avg_recon_loss = total_recon_loss / len(self.train_loader)
        avg_tv_loss = total_tv_loss / len(self.train_loader)
        avg_psnr = sum(psnr_values) / len(psnr_values)

        return avg_loss, avg_psnr, avg_recon_loss, avg_tv_loss
        
    def train(self):
        """学習の実行"""
        os.makedirs(self.config['save_dir'], exist_ok=True)

        best_psnr = 0
        for epoch in range(self.config['num_epochs']):
            loss, psnr, recon_loss, tv_loss = self.train_epoch(

                use_patch_loss=self.config['use_patch_loss'],
                patch_size=self.config['patch_size']
            )
            
            # W&Bにエポックごとの指標をログ
            wandb.log({
                "epoch": epoch,
                "loss": loss,
                "psnr": psnr,
                "reconstruction_loss": recon_loss,
                "tv_loss": tv_loss
            })

            print(f'Epoch {epoch + 1}/{self.config["num_epochs"]}')
            print(f'Loss: {loss:.4f}, PSNR: {psnr:.2f}')

            # 深度マップとワープ画像のサンプルをW&Bにログ
            if (epoch + 1) % 100 == 0:  # 100エポックごとに画像をログ
                with torch.no_grad():
                    # サンプルデータの取得と処理
                    sample_data = next(iter(self.train_loader))
                    ref_image = sample_data['ref_image'].to(self.device)
                    src_images = sample_data['src_images'].to(self.device)
                    K = sample_data['K'].to(self.device)
                    initial_depth = sample_data['initial_depth'].to(self.device)
                    relative_transforms = sample_data['relative_transforms'].to(self.device)
                    b, c, h, w = ref_image.shape

                    initial_depth_flat = initial_depth.squeeze().reshape(-1, 1)
                    pred_inverse_depth = self.net(initial_depth_flat)

                    pred_inverse_depth = torch.sigmoid(pred_inverse_depth)
                    # pred_inverse_depth = torch.clamp(pred_inverse_depth, -1, 1)
                    pred_inverse_depth = pred_inverse_depth.reshape(b, 1, h, w)
                    # pred_inverse_depth = (pred_inverse_depth + 1) / 2.0

                    # 画像をW&Bにログ
                    wandb.log({
                        "depth_map": wandb.Image(pred_inverse_depth[0, 0].cpu().numpy()),
                        "reference_image": wandb.Image(ref_image[0].permute(1, 2, 0).cpu().numpy())
                    })

            # モデルの保存
            if psnr > best_psnr:
                best_psnr = psnr
                model_path = f"{self.config['save_dir']}/best_model.pth"
                torch.save(self.net.state_dict(), model_path)
                wandb.save(model_path)  # W&Bにも保存

        # 学習終了時の処理
        wandb.finish()

if __name__ == "__main__":
    # 設定
    config = {
        'data_root': '/home/rintoyagawa/ssd2/Code/invdepth/data',
        'num_epochs': 100,
        'batch_size': 1,
        'num_workers': 4,
        'num_source_views': 3,
        'img_height': 256,
        'img_width': 256,
        'focal_length_mm': 55,
        'sensor_width_mm': 24,
        'use_patch_loss': False,
        'patch_size': 1,
        'save_dir': 'checkpoints',
        'results_dir': 'results',
        'run_name': 'depth-anything-view10'  # W&B実験の名前
    }

    trainer = DepthAnythingTrainer(config)
    trainer.train()