# Colab cell 6 | Grocerystuffs_Train.ipynb
# Run in order using %run -i in the same Colab runtime (see README.md).

# 6. Danh gia TEST sau khi chot model - ca bounding box va mask
from ultralytics import YOLO
assert BEST_PT.is_file() and DATA_YAML.is_file()
best_model=YOLO(str(BEST_PT))
test_metrics=best_model.val(data=str(DATA_YAML),split='test',imgsz=640,batch=4,device=0,workers=2,
    plots=True,project=str(RUN_DIR),name='test_evaluation',exist_ok=True)
test_summary=dict(dataset_sha256=PREPARED_SHA,checkpoint_sha256=hashlib.sha256(BEST_PT.read_bytes()).hexdigest(),
    box=dict(precision=float(test_metrics.box.mp),recall=float(test_metrics.box.mr),map50=float(test_metrics.box.map50),map5095=float(test_metrics.box.map)),
    mask=dict(precision=float(test_metrics.seg.mp),recall=float(test_metrics.seg.mr),map50=float(test_metrics.seg.map50),map5095=float(test_metrics.seg.map)))
(RUN_DIR/'test_summary.json').write_text(json.dumps(test_summary,indent=2))
print(json.dumps(test_summary,indent=2))
print('MODEL MOI:',BEST_PT)
