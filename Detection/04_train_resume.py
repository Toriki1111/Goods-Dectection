# Colab cell 4 | Grocerystuffs_Train.ipynb
# Run in order using %run -i in the same Colab runtime (see README.md).

# 4. Train / resume YOLO26l-seg tren Grocerystuffs
from ultralytics import YOLO
import json,hashlib,torch
assert torch.cuda.is_available() and globals().get('VALIDATED_SHA')==PREPARED_SHA
RUN_NAME='Grocerystuffs_Seg_20261029'
RUN_DIR=DRIVE_DIR/'runs'/RUN_NAME
BEST_PT=RUN_DIR/'weights/best.pt'
LAST_PT=RUN_DIR/'weights/last.pt'
PROVENANCE=RUN_DIR/'dataset_provenance.json'
is_resume=LAST_PT.is_file()
if is_resume:
    saved=json.loads(PROVENANCE.read_text())
    assert saved['source_sha256']==SOURCE_SHA and saved['prepared_sha256']==PREPARED_SHA and saved['task']=='segment'
    checkpoint=torch.load(LAST_PT,map_location='cpu',weights_only=False)
    assert checkpoint.get('epoch',-1)>=0 and checkpoint.get('optimizer') is not None, 'Training already finished; do not restart it.'
    print('RESUME from epoch:',checkpoint['epoch']+2)
    del checkpoint
    model=YOLO(str(LAST_PT))
else:
    assert not RUN_DIR.exists(), f'Existing folder without last.pt: {RUN_DIR}; inspect before reuse.'
    model=YOLO('yolo26l-seg.pt')
assert model.task=='segment'
def verify_training(trainer):
    assert trainer.args.task=='segment' and Path(trainer.args.data).resolve()==DATA_YAML.resolve()
    assert trainer.data['names']==NAMES and Path(trainer.save_dir).resolve()==RUN_DIR.resolve()
    if not PROVENANCE.is_file():
        PROVENANCE.write_text(json.dumps(dict(source_sha256=SOURCE_SHA,prepared_sha256=PREPARED_SHA,
            task='segment',pretrained='yolo26l-seg.pt',names=NAMES,splits=audit_splits,
            epochs=100,patience=20,imgsz=640,batch=trainer.batch_size,lr0=0.0005),ensure_ascii=False,indent=2))
    print('VERIFIED_GROCERIES_TRAIN:',trainer.start_epoch+1,'/100 | batch:',trainer.batch_size,'| checkpoint:',trainer.last,flush=True)
def mark_epoch(trainer): trainer._grocery_epoch=trainer.epoch
def save_progress(trainer):
    if trainer.epoch!=getattr(trainer,'_grocery_epoch',None): return
    assert Path(trainer.last).is_file(), 'Checkpoint has not been saved to Drive.'
    if trainer.epoch==trainer.start_epoch:
        print('CHECKPOINT_SAVED:',trainer.last,'| bytes:',Path(trainer.last).stat().st_size,flush=True)
    (RUN_DIR/'progress.json').write_text(json.dumps(dict(epoch=trainer.epoch+1,batch=trainer.batch_size,
        metrics={k:float(v) for k,v in trainer.metrics.items()},last=str(trainer.last)),indent=2))
model.add_callback('on_train_start',verify_training)
model.add_callback('on_train_epoch_end',mark_epoch)
model.add_callback('on_fit_epoch_end',save_progress)
if is_resume:
    results=model.train(resume=True,data=str(DATA_YAML),device=0,workers=2,save_dir=str(RUN_DIR))
else:
    results=model.train(data=str(DATA_YAML),epochs=100,patience=20,imgsz=640,
        batch=0.7,device=0,workers=2,cache=False,amp=True,freeze=None,nbs=64,
        optimizer='AdamW',lr0=0.0005,lrf=0.1,cos_lr=True,weight_decay=0.0005,warmup_epochs=5.0,
        hsv_h=0.015,hsv_s=0.4,hsv_v=0.3,degrees=3.0,translate=0.1,scale=0.3,
        fliplr=0.5,flipud=0.0,mosaic=0.3,close_mosaic=10,mixup=0.0,
        seed=42,deterministic=True,val=True,plots=True,save=True,save_period=5,
        project=str(DRIVE_DIR/'runs'),name=RUN_NAME,exist_ok=False)
RUN_DIR=Path(model.trainer.save_dir)
BEST_PT=RUN_DIR/'weights/best.pt'; LAST_PT=RUN_DIR/'weights/last.pt'
print('TRAIN_FINISHED:',RUN_DIR,'| Best:',BEST_PT,'| Last:',LAST_PT)
