# Colab cell 2 | Grocerystuffs_Train.ipynb
# Run in order using %run -i in the same Colab runtime (see README.md).

# 2. Chia theo nhom nguon, chuyen NDJSON thanh YOLO segmentation
from collections import Counter,defaultdict
import re, numpy as np, zipfile, yaml
from ultralytics.data.converter import convert_ndjson_to_yolo

def source_group(record):
    stem=Path(record['file']).stem.lower()
    if '.rf.' in stem: return 'rf:'+stem.split('.rf.')[0]
    match=re.match(r'win_(\d{8})_(\d{2})_(\d{2})_',stem)
    if match: return 'camera:'+''.join(match.groups())
    match=re.match(r'img_(\d{8})_(\d{2})(\d{2})',stem)
    if match: return 'camera:'+''.join(match.groups())
    return 'file:'+stem

image_records=[r for r in records if r.get('type')=='image']
groups=defaultdict(list)
for index,record in enumerate(image_records): groups[source_group(record)].append(index)
keys=sorted(groups)
sizes=np.array([len(groups[k]) for k in keys])
class_counts=np.zeros((len(keys),len(NAMES)),dtype=int)
for j,key in enumerate(keys):
    for index in groups[key]:
        for segment in image_records[index].get('annotations',{}).get('segments',[]):
            assert len(segment)>=7 and (len(segment)-1)%2==0
            assert int(segment[0]) in NAMES and all(0<=v<=1 for v in segment[1:])
            class_counts[j,int(segment[0])]+=1
active=class_counts.sum(axis=0)>0
eligible_eval=(class_counts>0).sum(axis=0)>=3
rng=np.random.default_rng(42)
best_score=float('inf'); assignment=None
target=np.array([0.8,0.1,0.1])
for attempt in range(2000):
    candidate=rng.choice(3,size=len(keys),p=target)
    for cls in np.flatnonzero(active):
        if class_counts[candidate==0,cls].sum()==0:
            possible=np.flatnonzero(class_counts[:,cls]>0)
            candidate[rng.choice(possible)]=0
    totals=np.bincount(candidate,weights=sizes,minlength=3)
    if np.any(totals==0): continue
    split_classes=np.array([class_counts[candidate==s].sum(axis=0) for s in range(3)])
    missing_eval=int(((split_classes[1:]==0)&eligible_eval).sum())
    score=1000*float(((totals/len(image_records)-target)**2).sum())+2*missing_eval
    if score<best_score: best_score=score; assignment=candidate.copy()
assert assignment is not None
split_names=('train','val','test')
new_split={index:split_names[int(assignment[j])] for j,key in enumerate(keys) for index in groups[key]}
prepared=[]; image_index=0
for record in records:
    copy=dict(record)
    if copy.get('type')=='image':
        copy['split']=new_split[image_index]; image_index+=1
    prepared.append(copy)
excluded_records=[r for r in prepared if r.get('type')=='image' and not r.get('annotations',{}).get('segments',[])]
prepared=[r for r in prepared if r.get('type')!='image' or r.get('annotations',{}).get('segments',[])]
assert len(excluded_records)==14
TRAIN_SNAPSHOT_DIR=SNAPSHOT_DIR/'labeled_only'
TRAIN_SNAPSHOT_DIR.mkdir(parents=True,exist_ok=True)
prepared_bytes=('\n'.join(json.dumps(r,ensure_ascii=False,separators=(',',':')) for r in prepared)+'\n').encode('utf-8')
PREPARED_SHA=hashlib.sha256(prepared_bytes).hexdigest()
NDJSON_PATH=TRAIN_SNAPSHOT_DIR/'grouped.ndjson'
if NDJSON_PATH.is_file(): assert NDJSON_PATH.read_bytes()==prepared_bytes, 'Grouped snapshot changed; inspect before reuse.'
else: NDJSON_PATH.write_bytes(prepared_bytes)
image_records=[r for r in prepared if r.get('type')=='image']
expected_splits=Counter(r['split'] for r in image_records)
expected_segments=sum(len(r.get('annotations',{}).get('segments',[])) for r in image_records)
expected_empty=sum(not r.get('annotations',{}).get('segments',[]) for r in image_records)
assert len(image_records)==904 and expected_segments==1827 and expected_empty==0
group_splits=defaultdict(set)
for record in image_records: group_splits[source_group(record)].add(record['split'])
assert all(len(splits)==1 for splits in group_splits.values())
train_classes={int(s[0]) for r in image_records if r['split']=='train' for s in r.get('annotations',{}).get('segments',[])}
assert train_classes==set(np.flatnonzero(active).tolist())
print('GROUPED_SPLITS:',dict(expected_splits),'| polygons:',expected_segments,'| empty:',expected_empty)
print('Absent class:',[NAMES[i] for i in NAMES if not active[i]])
print('Prepared SHA256:',PREPARED_SHA)
DATASET_ZIP=TRAIN_SNAPSHOT_DIR/'dataset.zip'
if DATASET_ZIP.is_file():
    DATASET_ROOT=WORK_DIR/'restored'
    DATASET_ROOT.mkdir(exist_ok=True)
    with zipfile.ZipFile(DATASET_ZIP) as archive:
        for name in archive.namelist(): assert (DATASET_ROOT/name).resolve().is_relative_to(DATASET_ROOT.resolve())
        assert archive.testzip() is None
        archive.extractall(DATASET_ROOT)
    DATA_YAML=DATASET_ROOT/'data.yaml'
else:
    # Colab already has an event loop; run the converter in a worker loop.
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as executor:
        converted = executor.submit(
            asyncio.run,
            convert_ndjson_to_yolo(str(NDJSON_PATH), output_path=str(WORK_DIR/'converted')),
        ).result()
    DATA_YAML=Path(converted)
    DATASET_ROOT=DATA_YAML.parent
assert DATA_YAML.is_file()
config=yaml.safe_load(DATA_YAML.read_text())
config['path']=str(DATASET_ROOT)
config['task']='segment'
for split in split_names: config[split]=f'images/{split}'
DATA_YAML.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
print('DATA_YAML:',DATA_YAML)
