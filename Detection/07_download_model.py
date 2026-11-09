# Colab cell 7 | Grocerystuffs_Train.ipynb
# Run in order using %run -i in the same Colab runtime (see README.md).

# 7. Tai best.pt Grocerystuffs ve may
from pathlib import Path
from google.colab import drive, files
export_model = Path('/CV_Object_Detection/drive/MyDrive/YOLO_Grocerystuffs/runs/Grocerystuffs_Seg_20261029/weights/best.pt')
try:
    available = export_model.is_file()
except OSError:
    available = False
if not available:
    drive.mount('/CV_Object_Detection/drive', force_remount=True)
assert export_model.is_file(), f'Khong tim thay: {export_model}'
print('Downloading:', export_model.name, '| bytes:', export_model.stat().st_size)
files.download(str(export_model))