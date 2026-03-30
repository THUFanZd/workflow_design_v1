import os
import pandas as pd
from pathlib import Path

def merge_csv_files():
    """
    合并 all_tokens 和 last_token_only 目录下的 batch_results_table.csv 文件
    """
    base_path = Path("./steer_playground/output")
    
    # 获取所有 layer-id 目录
    for layer_dir in base_path.iterdir():
        if not layer_dir.is_dir():
            continue
            
        # 检查 layer-id 是否为数字
        try:
            layer_id = int(layer_dir.name)
        except ValueError:
            continue
            
        # 遍历所有 feature-id 目录
        for feature_dir in layer_dir.iterdir():
            if not feature_dir.is_dir():
                continue
                
            # 检查 feature-id 是否为数字
            try:
                feature_id = int(feature_dir.name)
            except ValueError:
                continue
            
            # 定义路径
            all_tokens_path = feature_dir / "all_tokens" / "batch_results_table.csv"
            last_token_only_path = feature_dir / "last_token_only" / "batch_results_table.csv"
            
            # 检查是否存在 all_tokens 和 last_token_only 目录以及对应的 CSV 文件
            if all_tokens_path.exists() and last_token_only_path.exists():
                try:
                    # 读取两个 CSV 文件
                    all_tokens_df = pd.read_csv(all_tokens_path)
                    last_token_only_df = pd.read_csv(last_token_only_path)
                    
                    # 将两个 DataFrame 垂直拼接
                    merged_df = pd.concat([all_tokens_df, last_token_only_df], ignore_index=True)
                    
                    # 保存合并后的文件
                    output_path = feature_dir / "merged_batch_results_table.csv"
                    merged_df.to_csv(output_path, index=False)
                    
                    print(f"已合并 {layer_id}/{feature_id}: {len(all_tokens_df)} + {len(last_token_only_df)} = {len(merged_df)} 行")
                    
                except Exception as e:
                    print(f"处理 {layer_id}/{feature_id} 时出错: {e}")
                    continue
            else:
                # 如果任一文件不存在，则跳过
                continue

if __name__ == "__main__":
    merge_csv_files()