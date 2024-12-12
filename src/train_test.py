# train_test.py

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from train import InverseDepthTrainer
from utils.warp import warp_with_inverse_depth_mesh
from utils.camera import compute_relative_transform

def visualize_results(trainer, save_dir='results'):
    """学習結果の可視化"""
    os.makedirs(save_dir, exist_ok=True)
    trainer.net.eval()

    with torch.no_grad():
        # データセットから1セットのデータを取得
        data = next(iter(trainer.train_loader))
        ref_image = data['ref_image'].to(trainer.device)
        src_images = data['src_images'].to(trainer.device)
        K = data['K'].to(trainer.device)
        relative_transforms = data['relative_transforms'].to(trainer.device)
        print(K)
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
        ).to(trainer.device)

        # 逆深度の予測
        pred_inverse_depth = trainer.net(coords)
        pred_inverse_depth = torch.clamp(pred_inverse_depth, -1, 1)
        pred_inverse_depth = pred_inverse_depth.reshape(b, 1, h, w)
        pred_inverse_depth = (pred_inverse_depth + 1) / 2.0

        # 1. Inverse Depthマップの可視化
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(pred_inverse_depth[0, 0].cpu().numpy(), cmap='plasma')
        plt.colorbar()
        plt.title('Predicted Inverse Depth Map')
        plt.axis('off')

        # 2. 参照画像の表示
        plt.subplot(122)
        ref_img_display = ref_image[0].permute(1, 2, 0).cpu().numpy()
        plt.imshow(ref_img_display)
        plt.title('Reference Image')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, 'depth_and_reference.png'))
        plt.close()

        # 3. ワーピング結果の可視化
        num_source_views = src_images.size(1)
        plt.figure(figsize=(20, 5 * num_source_views))
        
        for i in range(num_source_views):
            print(relative_transforms[:, i])
            pred_img, warp_inv, mask = warp_with_inverse_depth_mesh(
                ref_image,
                pred_inverse_depth,
                trainer.device,
                K,
                relative_transforms[:, i][:,:3,:]
            )

            # 真値（ソース画像）の表示
            plt.subplot(num_source_views, 4, 4*i + 1)
            src_img_display = src_images[0, i].permute(1, 2, 0).cpu().numpy()
            src_img_display = np.clip(src_img_display, 0, 1)
            plt.imshow(src_img_display)
            plt.title(f'Source View {i+1} (Ground Truth)')
            plt.axis('off')

            # ワーピング結果の表示
            plt.subplot(num_source_views, 4, 4*i + 2)
            warped_img_display = pred_img[0].permute(1, 2, 0).cpu().numpy()
            warped_img_display = np.clip(warped_img_display, 0, 1)
            plt.imshow(warped_img_display)
            plt.title(f'Source View {i+1} (Warped)')
            plt.axis('off')

            plt.subplot(num_source_views, 4, 4*i + 3)
            mask_display = mask[0, 0].cpu().numpy()
            plt.imshow(mask_display, cmap='gray')
            plt.colorbar()
            plt.title(f'Visibility Mask {i+1}')
            plt.axis('off')

            # 差分の表示
            mask_display_3ch = np.expand_dims(mask_display, axis=2).repeat(3, axis=2)
            plt.subplot(num_source_views, 4, 4*i + 4)
            diff = np.abs(src_img_display - warped_img_display)
            plt.imshow(diff*mask_display_3ch)
            plt.colorbar()
            plt.title('Absolute Difference')
            plt.axis('off')

            # plt.subplot(num_source_views, 4, 4*i + 4)
            # plt.imshow((mask[0, 0]*warp_inv[0, 0]).cpu().numpy(), cmap='plasma')
            # plt.colorbar()
            # plt.title('Warped Inverse Depth Map')
            # plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'warping_comparison_with_masks.png'))
        plt.close()

def main():
    # 設定
    config = {
        'data_root': '/home/rintoyagawa/ssd2/Code/invdepth/data',  # データセットのパスを適切に設定
        'hidden_dim': 256,
        'hidden_layers': 2,
        'learning_rate': 1e-4,
        'num_epochs': 500,  # テスト用に少なめのエポック数
        'batch_size': 1,
        'num_workers': 0,  # デバッグ時は0にする
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

    # トレーナーの初期化と学習
    trainer = InverseDepthTrainer(config)
    print("Starting training...")
    trainer.train()
    print("Training completed!")

    # 結果の可視化
    print("Visualizing results...")
    visualize_results(trainer, save_dir=config['results_dir'])
    print("Visualization completed! Check the 'results' directory for output images.")

if __name__ == "__main__":
    main()