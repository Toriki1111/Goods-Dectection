# Colab cell 3 | Grocerystuffs_Train.ipynb
# Run in order using %run -i in the same Colab runtime (see README.md).

# 3. Kiem tra du anh/polygon va luu dataset.zip de resume
from PIL import Image,ImageDraw,ImageOps
from IPython.display import display
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from urllib.parse import urlsplit
import os, gc
VALIDATED_SHA=None
info=check_det_dataset(str(DATA_YAML))
assert info['names']==NAMES
split_images={}; audit_splits={}; total_segments=0; total_empty=0
for split in ('train','val','test'):
    ds=YOLODataset(img_path=info[split],data=info,task='segment',imgsz=640,augment=False,cache=False,hyp=get_cfg())
    paths=[Path(p) for p in ds.im_files]
    assert len(paths)==expected_splits[split], f'Missing/corrupt image in {split}'
    for p in paths:
        with Image.open(p) as image: image.verify()
    instances=sum(len(label['cls']) for label in ds.labels)
    polygons=sum(len(label['segments']) for label in ds.labels)
    assert instances==polygons
    empty=sum(len(label['cls'])==0 for label in ds.labels)
    audit_splits[split]=dict(images=len(paths),segments=polygons,empty=empty)
    split_images[split]=paths
    total_segments+=polygons; total_empty+=empty
    print(split,audit_splits[split])
    del ds
expected_unique_segments=sum(len({tuple(s) for s in r.get('annotations',{}).get('segments',[])}) for r in image_records)
assert total_segments==expected_unique_segments and total_empty==expected_empty
print('Exact duplicate polygons normalized:',expected_segments-expected_unique_segments)
# File anh cung bytes khong duoc nam o hai split khac nhau.
content_splits=defaultdict(set)
for split,paths in split_images.items():
    for p in paths: content_splits[hashlib.sha256(p.read_bytes()).hexdigest()].add(split)
assert all(len(v)==1 for v in content_splits.values()), 'Exact duplicate images cross splits; resolve before training.'
# Xem polygon tren 8 anh co nhan; 14 anh thieu nhan duoc giu trong source_export.ndjson.
samples=[]
for cls in NAMES:
    match=next((r for r in image_records if any(int(s[0])==cls for s in r.get('annotations',{}).get('segments',[])) and r not in samples),None)
    if match is not None: samples.append(match)
    if len(samples)==8: break
canvas=Image.new('RGB',(4*280,2*230),'#202020'); draw=ImageDraw.Draw(canvas)
all_paths=[p for paths in split_images.values() for p in paths]
for i,r in enumerate(samples):
    stem=Path(urlsplit(r['url']).path).stem
    matches=[p for p in all_paths if p.stem.startswith(stem+'_')]
    assert len(matches)==1
    with Image.open(matches[0]) as im: image=im.convert('RGB')
    overlay=ImageDraw.Draw(image)
    for seg in r.get('annotations',{}).get('segments',[]):
        pts=[(x*image.width,y*image.height) for x,y in zip(seg[1::2],seg[2::2])]
        overlay.line(pts+[pts[0]],fill='lime',width=4)
    thumb=ImageOps.contain(image,(275,200))
    x=(i%4)*280; y=(i//4)*230
    canvas.paste(thumb,(x,y)); draw.text((x+3,y+203),f"{r['split']} | {r['file'][:26]}",fill='white')
print('POLYGON REVIEW - 14 unannotated product photos excluded, originals retained.')
display(canvas)
audit=dict(source_sha256=SOURCE_SHA,prepared_sha256=PREPARED_SHA,task='segment',names=NAMES,
    grouping='Roboflow original stem; camera photos grouped by capture minute; seed42; all observed classes in train',
    excluded_unannotated=[r['file'] for r in excluded_records],raw_segments=expected_segments,unique_segments=expected_unique_segments,splits=audit_splits,missing_classes=[NAMES[i] for i in NAMES if not active[i]])
(TRAIN_SNAPSHOT_DIR/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
if not DATASET_ZIP.is_file():
    local_zip=Path(shutil.make_archive(str(WORK_DIR/'dataset_backup'),'zip',root_dir=DATASET_ROOT))
    with zipfile.ZipFile(local_zip) as archive: assert archive.testzip() is None
    temporary_zip=DATASET_ZIP.with_suffix('.zip.partial')
    shutil.copy2(local_zip,temporary_zip)
    assert hashlib.sha256(temporary_zip.read_bytes()).hexdigest()==hashlib.sha256(local_zip.read_bytes()).hexdigest()
    os.replace(temporary_zip,DATASET_ZIP)
VALIDATED_SHA=PREPARED_SHA
gc.collect()
print('VALIDATED:',VALIDATED_SHA,'| backup:',DATASET_ZIP)
