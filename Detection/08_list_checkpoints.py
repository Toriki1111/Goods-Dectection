# Colab cell 8 | Grocerystuffs_Train.ipynb
# Run in order using %run -i in the same Colab runtime (see README.md).

# 8. Xem checkpoint epoch 5-95 truoc khi don
from pathlib import Path
import re
checkpoint_dir = Path('/CV_Object_Detection/drive/MyDrive/YOLO_Grocerystuffs/runs/Grocerystuffs_Seg_20261029/weights')
cleanup_candidates = sorted([p for p in checkpoint_dir.iterdir() if p.is_file() and re.fullmatch(r'epoch(\d+)\.pt', p.name) and 5 <= int(re.fullmatch(r'epoch(\d+)\.pt', p.name)[1]) <= 95], key=lambda p: int(p.stem[5:]))
for p in cleanup_candidates:
    print(p.name, round(p.stat().st_size/1e6, 1), 'MB')
print('COUNT:', len(cleanup_candidates), '| TOTAL_GB:', round(sum(p.stat().st_size for p in cleanup_candidates)/1e9, 3))
print('KEEP:', [p.name for p in checkpoint_dir.iterdir() if p.is_file() and p not in cleanup_candidates])