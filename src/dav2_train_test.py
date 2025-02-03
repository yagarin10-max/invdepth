import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from depth_anything_trainer import DepthAnythingTrainer  # 新しいトレーナークラスを使用
from utils.warp import warp_with_inverse_depth_mesh

def visualize_results(trainer, save_dir='results', max_views=4):
    """深度推定結果の可視化
    
    Args:
        trainer: DepthAnythingTrainerのインスタンス
        save_dir: 結果の保存ディレクトリ
        max_views: 可視化する最大ターゲット視点数
    """
    os.makedirs(save_dir, exist_ok=True)
    trainer.depth_model.eval()

    with torch.no_grad():
        # データセットから1セットのデータを取得
        data = next(iter(trainer.train_loader))
        ref_image = data['ref_image'].to(trainer.device)
        src_images = data['src_images'].to(trainer.device)
        K = data['K'].to(trainer.device)
        relative_transforms = data['relative_transforms'].to(trainer.device)
        
        b, c, h, w = ref_image.shape
        num_source_views = src_images.size(1)
        
        # 可視化するビュー数を決定
        views_to_show = min(num_source_views, max_views)
        
        # ビューのインデックスを選択
        if num_source_views > max_views:
            step = num_source_views // max_views
            view_indices = list(range(0, num_source_views, step))[:max_views]
        else:
            view_indices = list(range(num_source_views))

        # Depth Anything V2で深度を推定
        pred_inverse_depth = trainer.estimate_depth(ref_image)

        # Inverse Depthマップと参照画像の可視化
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(pred_inverse_depth[0, 0].cpu().numpy(), cmap='plasma')
        plt.colorbar()
        plt.title('Predicted Inverse Depth Map')
        plt.axis('off')

        plt.subplot(122)
        ref_img_display = ref_image[0].permute(1, 2, 0).cpu().numpy()
        plt.imshow(ref_img_display)
        plt.title('Reference Image')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, 'depth_and_reference.png'))
        plt.close()

        # ワーピング結果の可視化
        plt.figure(figsize=(20, 5 * views_to_show))
        
        for i, view_idx in enumerate(view_indices):
            pred_img, warp_inv, mask = warp_with_inverse_depth_mesh(
                ref_image,
                pred_inverse_depth,
                trainer.device,
                K,
                relative_transforms[:, view_idx][:,:3,:]
            )

            # 真値（ソース画像）の表示
            plt.subplot(views_to_show, 4, 4*i + 1)
            src_img_display = src_images[0, view_idx].permute(1, 2, 0).cpu().numpy()
            src_img_display = np.clip(src_img_display, 0, 1)
            plt.imshow(src_img_display)
            plt.title(f'Source View {view_idx+1} (Ground Truth)')
            plt.axis('off')

            # ワーピング結果の表示
            plt.subplot(views_to_show, 4, 4*i + 2)
            warped_img_display = pred_img[0].permute(1, 2, 0).cpu().numpy()
            warped_img_display = np.clip(warped_img_display, 0, 1)
            plt.imshow(warped_img_display)
            plt.title(f'Source View {view_idx+1} (Warped)')
            plt.axis('off')

            # マスクの表示
            plt.subplot(views_to_show, 4, 4*i + 3)
            mask_display = mask[0, 0].cpu().numpy()
            plt.imshow(mask_display, cmap='gray')
            plt.colorbar()
            plt.title(f'Visibility Mask {view_idx+1}')
            plt.axis('off')

            # 差分の表示
            mask_display_3ch = np.expand_dims(mask_display, axis=2).repeat(3, axis=2)
            plt.subplot(views_to_show, 4, 4*i + 4)
            diff = np.abs(src_img_display - warped_img_display)
            plt.imshow(diff * mask_display_3ch)
            plt.colorbar()
            plt.title('Absolute Difference')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'warping_comparison_with_masks.png'))
        plt.close()

def main():
    # 設定
    config = {
        'data_root': '/home/rintoyagawa/ssd2/Code/invdepth/temp_datas/horizon_5/view20_inter3mm',  # データセットのパスを適切に設定
        'batch_size': 1,
        'num_workers': 0,  # デバッグ時は0にする
        'num_source_views': 20,
        'img_height': 256,
        'img_width': 256,
        'focal_length_mm': 55,
        'sensor_width_mm': 24,
        'use_patch_loss': False,
        'patch_size': 1,
        'tv_weight': 15e-3,
        'save_dir': '/home/rintoyagawa/ssd2/Code/invdepth/checkpoints/horizon_5/view20_inter3mm',
        'results_dir': '/home/rintoyagawa/ssd2/Code/invdepth/results/dav2/s5_v20_e3000'
    }

    # トレーナーの初期化
    trainer = DepthAnythingTrainer(config)
    
    # 結果の可視化
    print("Visualizing results...")
    visualize_results(trainer, save_dir=config['results_dir'])
    print("Visualization completed! Check the 'results' directory for output images.")

if __name__ == "__main__":
    main()