# 划分训练集

import os
import random
import shutil
from pathlib import Path

def setup_dirs(base_path):
    """创建 processed_images 下的标准目录结构"""
    splits = ['train', 'val', 'test']
    subdirs = ['images', 'masks']
    for split in splits:
        for subdir in subdirs:
            (base_path / subdir / split).mkdir(parents=True, exist_ok=True)

def main():
    # 路径设置
    annotated_dir = Path("data/annotated_data")
    processed_dir = Path("data/processed_images")
    
    setup_dirs(processed_dir)
    
    categories = ['Erosion', 'Polyp'] # 目前你标注的类别
    
    train_ratio, val_ratio = 0.7, 0.1
    
    total_train, total_val, total_test = 0, 0, 0

    for cat in categories:
        img_cat_dir = annotated_dir / "images" / cat
        mask_cat_dir = annotated_dir / "masks" / cat
        
        if not img_cat_dir.exists() or not mask_cat_dir.exists():
            print(f"[跳过] 找不到类别 {cat} 的数据目录")
            continue
            
        # 获取该类别下所有有 Mask 的图像文件名
        mask_files = list(mask_cat_dir.glob("*.png"))
        valid_stems = [p.stem for p in mask_files]
        
        # 随机打乱
        random.seed(42) # 固定随机种子，保证每次划分一致
        random.shuffle(valid_stems)
        
        # 计算划分索引
        num_files = len(valid_stems)
        train_idx = int(num_files * train_ratio)
        val_idx = train_idx + int(num_files * val_ratio)
        
        splits_dict = {
            'train': valid_stems[:train_idx],
            'val': valid_stems[train_idx:val_idx],
            'test': valid_stems[val_idx:]
        }
        
        # 复制文件
        for split_name, stems in splits_dict.items():
            for stem in stems:
                # 为了防止不同类别有同名文件，复制时加上类别前缀
                new_name = f"{cat}_{stem}"
                
                # 复制原图 (假设原图可能是 jpg)
                src_img = list(img_cat_dir.glob(f"{stem}.*"))[0] 
                dst_img = processed_dir / "images" / split_name / f"{new_name}{src_img.suffix}"
                shutil.copy(src_img, dst_img)
                
                # 复制 Mask (固定为 png)
                src_mask = mask_cat_dir / f"{stem}.png"
                dst_mask = processed_dir / "masks" / split_name / f"{new_name}.png"
                shutil.copy(src_mask, dst_mask)
                
            if split_name == 'train': total_train += len(stems)
            elif split_name == 'val': total_val += len(stems)
            elif split_name == 'test': total_test += len(stems)

    print(f"数据划分完毕！")
    print(f"Train: {total_train} | Val: {total_val} | Test: {total_test}")

if __name__ == "__main__":
    main()