# sweep.py

import wandb
from train import InverseDepthTrainer

# スイープの設定
sweep_configuration = {
    'method': 'bayes',  # ベイズ最適化を使用
    'metric': {
        'name': 'psnr',
        'goal': 'maximize'
    },
    'parameters': {
        'hidden_dim': {
            'values': [128, 256, 512]
        },
        'hidden_layers': {
            'values': [2, 3, 4]
        },
        'learning_rate': {
            'distribution': 'log_uniform',
            'min': -5,  # 10^-5
            'max': -3   # 10^-3
        },
        'tv_weight': {
            'distribution': 'log_uniform',
            'min': -4,  # 10^-4
            'max': -2   # 10^-2
        },
        'batch_size': {
            'values': [1, 2, 4]
        },
        'use_patch_loss': {
            'values': [True, False]
        },
        'patch_size': {
            'values': [1, 3, 5]
        }
    }
}

def train_sweep():
    # デフォルトの設定
    base_config = {
        'data_root': '/home/rintoyagawa/ssd2/Code/invdepth/data',
        'num_epochs': 3000,
        'num_workers': 4,
        'num_source_views': 10,
        'img_height': 256,
        'img_width': 256,
        'focal_length_mm': 55,
        'sensor_width_mm': 24,
        'save_dir': '/home/rintoyagawa/ssd2/Code/invdepth/checkpoints',
        'results_dir': '/home/rintoyagawa/ssd2/Code/invdepth/results'
    }
    
    # wandb.initの実行（configはwandbが自動的に提供）
    with wandb.init() as run:
        # wandbの設定とbase_configをマージ
        config = {**base_config, **run.config}
        
        # トレーナーの初期化と学習の実行
        trainer = InverseDepthTrainer(config)
        trainer.train()

if __name__ == "__main__":
    # スイープIDの生成
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project="inverse-depth-estimation"
    )
    
    # スイープの実行（100回の試行）
    wandb.agent(sweep_id, train_sweep, count=100)