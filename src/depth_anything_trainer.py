import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import skimage.metrics
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from utils.warp import warp_with_inverse_depth_mesh, compute_loss
from datasets.multiview_dataset import MultiViewDepthDataset

class DepthAnythingTrainer:
    def __init__(
        self,
        config: dict,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.config = config
        self.device = device
        
        # Initialize Depth Anything V2
        self.depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        ).to(device)
        
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

    def estimate_depth(self, image):
        """Depth Anything V2を使用して深度を推定"""
        # 画像の前処理
        inputs = self.depth_processor(images=image, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # 深度推定
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
        # 推定された深度を取得
        predicted_depth = outputs.predicted_depth
        # 入力画像のサイズを取得
        target_size = (image.shape[2], image.shape[3])
        # バッチとチャンネル次元を追加してからリサイズ
        if len(predicted_depth.shape) == 2:
            predicted_depth = predicted_depth.unsqueeze(0)  # バッチ次元を追加
        if len(predicted_depth.shape) == 3:
            predicted_depth = predicted_depth.unsqueeze(1)  # チャンネル次元を追加
        
        # 入力画像のサイズにリサイズ
        predicted_depth = torch.nn.functional.interpolate(
            predicted_depth,
            size=target_size,
            mode='bilinear',
            align_corners=True
        )
        predicted_depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
        return predicted_depth

    def train_epoch(self, use_patch_loss: bool = True, patch_size: int = 3):
        """1エポックの学習を実行"""
        self.depth_model.eval()  # Depth Anythingは学習しない
        total_loss = 0
        psnr_values = []

        for batch in tqdm(self.train_loader):
            # バッチデータの取り出し
            ref_image = batch['ref_image'].to(self.device)
            src_images = batch['src_images'].to(self.device)
            K = batch['K'].to(self.device)
            relative_transforms = batch['relative_transforms'].to(self.device)

            # Depth Anything V2で逆深度を推定
            pred_inverse_depth = self.estimate_depth(ref_image)

            # マルチビューの再構成損失の計算
            total_recon_loss = 0
            batch_psnr = 0
            for i in range(src_images.size(1)):
                pred_img, warp_inv, mask = warp_with_inverse_depth_mesh(
                    ref_image,
                    pred_inverse_depth,
                    self.device,
                    K,
                    relative_transforms[:, i][:,:3,:],
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
            
            total_loss += total_recon_loss.item()

        return total_loss / len(self.train_loader), sum(psnr_values) / len(psnr_values)

    def train(self):
        """学習の実行"""
        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(self.config['results_dir'], exist_ok=True)

        for epoch in range(self.config['num_epochs']):
            loss, psnr = self.train_epoch(
                use_patch_loss=self.config['use_patch_loss'],
                patch_size=self.config['patch_size']
            )
            
            print(f'Epoch {epoch + 1}/{self.config["num_epochs"]}')
            print(f'Loss: {loss:.4f}, PSNR: {psnr:.2f}')

if __name__ == "__main__":
    # 設定
    config = {
        'data_root': '/path/to/data',
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
        'results_dir': 'results'
    }

    trainer = DepthAnythingTrainer(config)
    trainer.train()