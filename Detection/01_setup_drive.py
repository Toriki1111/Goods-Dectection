# Colab cell 1 | Grocerystuffs_Train.ipynb
# Run in order using %run -i in the same Colab runtime (see README.md).

# 1. GPU, Drive va NDJSON goc
from pathlib import Path
from google.colab import drive
import json, hashlib, shutil, torch, ultralytics
assert torch.cuda.is_available(), 'Can ket noi GPU truoc khi train.'
print('GPU:', torch.cuda.get_device_name(0), '| Ultralytics:', ultralytics.__version__)
MOUNT_DIR=Path('/CV_Object_Detection/drive')
MOUNT_DIR.parent.mkdir(parents=True,exist_ok=True)
drive.mount(str(MOUNT_DIR))
DRIVE_DIR=MOUNT_DIR/'MyDrive/YOLO_Grocerystuffs'
WORK_DIR=Path('/content/grocerystuffs')
SNAPSHOT_DIR=DRIVE_DIR/'datasets/grocerystuffs_20261029'
for folder in (DRIVE_DIR,WORK_DIR,SNAPSHOT_DIR): folder.mkdir(parents=True,exist_ok=True)
SOURCE_SHA='65dc7d0a947a98ff0adfbc8654a2d2c0c459ab06d7d6b8f6769b8c147988dac3'
SOURCE_PATH=SNAPSHOT_DIR/'source_export.ndjson'
UPLOAD=Path('/content/grocerystuffs.ndjson')
if not SOURCE_PATH.is_file():
    assert UPLOAD.is_file(), 'Upload grocerystuffs.ndjson vao Files cua Colab.'
    assert hashlib.sha256(UPLOAD.read_bytes()).hexdigest()==SOURCE_SHA
    shutil.copy2(UPLOAD,SOURCE_PATH)
assert hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()==SOURCE_SHA
records=[json.loads(s) for s in SOURCE_PATH.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
metadata=next(r for r in records if r.get('type')=='dataset')
image_records=[r for r in records if r.get('type')=='image']
NAMES={int(k):v for k,v in metadata['class_names'].items()}
assert metadata['task']=='segment' and len(image_records)==918 and len(NAMES)==31
print('Drive:',DRIVE_DIR,'| Anh:',len(image_records),'| Lop:',len(NAMES))
