# 原图对齐

import argparse
import shutil
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="根据生成的 Mask，从原图库中提取对应的原始图像。")
    parser.add_argument("--raw-dir", type=str, default="data/raw_images", help="包含所有类别的原始图像根目录")
    parser.add_argument("--mask-dir", type=str, default="data/annotated_data/masks", help="已经生成 Mask 的根目录")
    parser.add_argument("--out-dir", type=str, default="data/annotated_data/images", help="对齐后原图的输出根目录")
    parser.add_argument("--category", type=str, default="Polyp", help="指定要处理的类别文件夹名，例如 'Polyp'。输入 'all' 则处理所有子文件夹")
    return parser.parse_args()

def align_category(raw_cat_dir: Path, mask_cat_dir: Path, out_cat_dir: Path):
    if not mask_cat_dir.exists():
        print(f"[跳过] Mask 目录不存在: {mask_cat_dir}")
        return
    
    out_cat_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 扫描该类别下所有的原始图像，建立“文件名(不含后缀) -> 完整路径”的映射字典
    # 这样做是为了兼容原图可能是 .jpg, .png 或 .bmp 等不同格式
    raw_images_map = {}
    if raw_cat_dir.exists():
        for p in raw_cat_dir.glob("*"):
            if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                raw_images_map[p.stem] = p
    else:
        print(f"[错误] 原始图像目录不存在: {raw_cat_dir}")
        return

    # 2. 获取已经标注好的 Mask 列表
    mask_files = list(mask_cat_dir.glob("*.png"))
    if not mask_files:
        print(f"[警告] 目录中没有找到 Mask 文件: {mask_cat_dir}")
        return
        
    print(f"\n--- 开始对齐类别: {mask_cat_dir.name} ---")
    copied_count = 0
    missing_count = 0
    exist_count = 0
    
    # 3. 遍历每一个 Mask，去字典里找对应的原图并复制
    for mask_path in mask_files:
        stem = mask_path.stem  # 获取文件名（不带 .png）
        
        if stem in raw_images_map:
            src_img_path = raw_images_map[stem]
            dst_img_path = out_cat_dir / src_img_path.name
            
            if not dst_img_path.exists():
                shutil.copy2(src_img_path, dst_img_path) # copy2 会保留文件的原始元数据
                copied_count += 1
            else:
                exist_count += 1
        else:
            print(f"[警告] 找不到与 Mask 对应的原图: {mask_path.name}")
            missing_count += 1
            
    print(f"处理完成！")
    print(f"成功提取: {copied_count} 张 | 已存在跳过: {exist_count} 张 | 缺失原图: {missing_count} 张 | 总 Mask 数: {len(mask_files)}\n")

def main():
    args = parse_args()
    
    raw_base = Path(args.raw_dir)
    mask_base = Path(args.mask_dir)
    out_img_base = Path(args.out_dir)
    
    if args.category.lower() == 'all':
        # 自动遍历 masks 文件夹下的所有类别子文件夹
        if not mask_base.exists():
            print(f"[错误] 找不到 Mask 根目录: {mask_base}")
            return
        categories = [d.name for d in mask_base.iterdir() if d.is_dir()]
        for cat in categories:
            align_category(raw_base / cat, mask_base / cat, out_img_base / cat)
    else:
        # 只处理指定的类别
        cat = args.category
        align_category(raw_base / cat, mask_base / cat, out_img_base / cat)

if __name__ == "__main__":
    main()