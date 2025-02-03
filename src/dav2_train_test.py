import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from depth_anything_trainer import DepthAnythingTrainer  # 新しいトレーナークラスを使用
from utils.warp import warp_with_inverse_depth_mesh

def evaluate_depth(pred_inverse_depth, gt_depth, min_depth=0.5, max_depth=6.0, epsilon=1e-4):
    """Inverse DepthとGround Truth Depthの評価を行う
    
    Args:
        pred_inverse_depth: 予測されたInverse Depth (0~1の範囲)
        gt_depth: Ground Truth Depth (メートル単位)
        min_depth: 最小深度 (メートル)
        max_depth: 最大深度 (メートル)
    
    Returns:
        dict: 各種評価指標の結果
    """
    # Ground Truth DepthをInverse Depthに変換
    gt_depth_clipped = torch.clamp(gt_depth, min_depth, max_depth)
    gt_inverse_depth = (1/gt_depth_clipped - 1/max_depth) / (1/min_depth - 1/max_depth)
    
    # 有効なピクセルのマスクを作成
    valid_mask = (gt_depth >= min_depth) & (gt_depth <= max_depth)
    
    # 評価指標の計算
    metrics = {}
    
    # MAE (Mean Absolute Error)
    metrics['mae'] = torch.mean(torch.abs(pred_inverse_depth[valid_mask] - gt_inverse_depth[valid_mask]))
    
    # RMSE (Root Mean Square Error)
    metrics['rmse'] = torch.sqrt(torch.mean((pred_inverse_depth[valid_mask] - gt_inverse_depth[valid_mask])**2))
    
    # Scale Invariant Error
    # log(0)を防ぐ
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

def visualize_results(trainer, save_dir='results', max_views=4):
    """深度推定結果の可視化
    
    Args:
        trainer: DepthAnythingTrainerのインスタンス
        save_dir: 結果の保存ディレクトリ
        max_views: 可視化する最大ターゲット視点数
    """
    os.makedirs(save_dir, exist_ok=True)
    trainer.net.eval()

    with torch.no_grad():
        # データセットから1セットのデータを取得
        data = next(iter(trainer.train_loader))
        ref_image = data['ref_image'].to(trainer.device)
        src_images = data['src_images'].to(trainer.device)
        K = data['K'].to(trainer.device)
        ref_depth = data['ref_depth'].to(trainer.device)
        initial_depth = data['initial_depth'].to(trainer.device)
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
        initial_depth_flat = initial_depth.squeeze().reshape(-1, 1)
        pred_inverse_depth = trainer.net(initial_depth_flat)
        pred_inverse_depth = torch.clamp(pred_inverse_depth, -1, 1)
        pred_inverse_depth = pred_inverse_depth.reshape(b, 1, h, w)
        pred_inverse_depth = (pred_inverse_depth + 1) / 2.0

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

        min_depth = 0.5
        max_depth = 6.0
        
        metrics = evaluate_depth(pred_inverse_depth[0, 0], ref_depth[0])
        
        # 評価結果の表示と保存
        print("\nDepth Estimation Metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value.item():.4f}")
        
        # メトリクスをファイルに保存
        with open(os.path.join(save_dir, 'depth_metrics.txt'), 'w') as f:
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value.item():.4f}\n")
        
        # Ground TruthとPredictionの比較可視化
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        # gt_inverse_depth = (1/ref_depth - 1/max_depth) / (1/min_depth - 1/max_depth)
        gt_inverse_depth = (1/torch.clamp(ref_depth, min_depth, max_depth) - 1/max_depth) / (1/min_depth - 1/max_depth)
        plt.imshow(gt_inverse_depth[0].cpu().numpy(), cmap='plasma')
        plt.colorbar()
        plt.title('Ground Truth Inverse Depth')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(initial_depth[0].cpu().numpy(), cmap='plasma')
        plt.colorbar()
        plt.title('Depthanythingv2 result')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(pred_inverse_depth[0, 0].cpu().numpy(), cmap='plasma')
        plt.colorbar()
        plt.title('Updated Inverse Depth')
        plt.axis('off')
        
        plt.savefig(os.path.join(save_dir, 'depth_comparison.png'))

        # ヒストグラムの比較
        plt.figure(figsize=(15, 5))

        # Ground Truth Inverse Depthのヒストグラム
        plt.subplot(131)
        gt_hist = plt.hist(gt_inverse_depth[0].cpu().numpy().flatten(), 
                        bins=100, range=(0, 1), 
                        alpha=0.7, color='blue',
                        density=True)  # density=Trueで確率密度に正規化
        plt.title('Ground Truth Inverse Depth')
        plt.xlabel('Inverse Depth Value')
        plt.ylabel('Density')

        # Depth Anything V2のヒストグラム
        plt.subplot(132)
        initial_hist = plt.hist(initial_depth[0].cpu().numpy().flatten(), 
                            bins=100, range=(0, 1), 
                            alpha=0.7, color='green',
                            density=True)
        plt.title('Depth Anything V2')
        plt.xlabel('Inverse Depth Value')
        plt.ylabel('Density')

        # Updated Inverse Depthのヒストグラム
        plt.subplot(133)
        pred_hist = plt.hist(pred_inverse_depth[0, 0].cpu().numpy().flatten(), 
                            bins=100, range=(0, 1), 
                            alpha=0.7, color='red',
                            density=True)
        plt.title('Updated Inverse Depth')
        plt.xlabel('Inverse Depth Value')
        plt.ylabel('Density')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'depth_histograms.png'))
        plt.close()

        # 3つのヒストグラムを重ねて比較
        plt.figure(figsize=(10, 6))
        plt.hist(gt_inverse_depth[0].cpu().numpy().flatten(), 
                bins=100, range=(0, 1), 
                alpha=0.5, color='blue', 
                label='Ground Truth', density=True)
        plt.hist(initial_depth[0].cpu().numpy().flatten(), 
                bins=100, range=(0, 1), 
                alpha=0.5, color='green', 
                label='Depth Anything V2', density=True)
        plt.hist(pred_inverse_depth[0, 0].cpu().numpy().flatten(), 
                bins=100, range=(0, 1), 
                alpha=0.5, color='red', 
                label='Updated', density=True)

        plt.title('Inverse Depth Distribution Comparison')
        plt.xlabel('Inverse Depth Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'depth_histograms_comparison.png'))
        plt.close()

        # 基本統計量の計算と保存
        with open(os.path.join(save_dir, 'depth_statistics.txt'), 'w') as f:
            for name, data in [('Ground Truth', gt_inverse_depth[0]),
                            ('Depth Anything V2', initial_depth[0]),
                            ('Updated', pred_inverse_depth[0, 0])]:
                data_np = data.cpu().numpy().flatten()
                f.write(f'\n{name} Statistics:\n')
                f.write(f'Mean: {np.mean(data_np):.4f}\n')
                f.write(f'Std: {np.std(data_np):.4f}\n')
                f.write(f'Min: {np.min(data_np):.4f}\n')
                f.write(f'Max: {np.max(data_np):.4f}\n')
                f.write(f'Median: {np.median(data_np):.4f}\n')
        plt.close()

def main():
    # 設定
    config = {
        'data_root': '/home/rintoyagawa/ssd2/Code/invdepth/temp_datas/horizon_5/view20_inter3mm',  # データセットのパスを適切に設定
        'hidden_dim': 256,
        'hidden_layers': 2,
        'learning_rate': 1e-4,
        'num_epochs': 500,  # テスト用に少なめのエポック数
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
    print("Starting training...")
    trainer.train()
    print("Training completed!")
    # 結果の可視化
    print("Visualizing results...")
    visualize_results(trainer, save_dir=config['results_dir'])
    print("Visualization completed! Check the 'results' directory for output images.")

if __name__ == "__main__":
    main()