# Colab cell 5 | Grocerystuffs_Train.ipynb
# Run in order using %run -i in the same Colab runtime (see README.md).

# 5. Xem bieu do sau khi train xong
from IPython.display import Image as DisplayImage,display
assert BEST_PT.is_file()
for name in ('results.png','BoxPR_curve.png','MaskPR_curve.png','confusion_matrix_normalized.png','val_batch0_labels.jpg','val_batch0_pred.jpg'):
    path=RUN_DIR/name
    if path.is_file():
        print(name); display(DisplayImage(filename=str(path),width=1000))
