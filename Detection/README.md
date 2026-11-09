# Các phần code từ Colab

Tách từ notebook `Grocerystuffs_Train.ipynb` đang mở trên Colab ngày 09/11/2026.
Mỗi file tương ứng một ô code. Thư mục tên Detection; cấu hình bên trong vẫn là
**YOLO26l-seg**, 31 lớp Grocerystuffs, như notebook gốc.

| File | Phần tương ứng |
|---|---|
| `00_install_dependencies.py` | Cài thư viện từ requirements.txt ở thư mục gốc |
| `01_setup_drive.py` | Kiểm tra GPU, mount Drive, đọc NDJSON |
| `02_prepare_dataset.py` | Chia nhóm nguồn và chuyển dataset segmentation |
| `03_check_dataset.py` | Kiểm tra ảnh/polygon, xem nhãn và sao lưu dataset |
| `04_train_resume.py` | Train 100 epoch hoặc tiếp tục checkpoint bị gián đoạn |
| `05_show_results.py` | Xem biểu đồ và ảnh validation |
| `06_test_model.py` | Đánh giá best.pt trên tập test |
| `07_download_model.py` | Tải best.pt về máy |
| `08_list_checkpoints.py` | Liệt kê checkpoint epoch 5–95 và dung lượng |

## Cách dùng trong Colab

Đưa thư mục Detection vào `/content/Detection` và file requirements.txt ở gốc
dự án vào `/content/requirements.txt` trên Colab. Mọi thư viện pip được khai báo
tại file requirements.txt này; phần 00 chỉ gọi `pip install -r`.
Chạy từng lệnh trong
một ô code, theo thứ tự, trong cùng runtime:

```python
%run -i /content/Detection/00_install_dependencies.py
```

```python
%run -i /content/Detection/01_setup_drive.py
```

Tiếp tục tương tự với 02, 03 và 04. Sau khi train xong mới chạy 05–07.
`-i` giữ các biến trong cùng không gian làm việc giữa các phần.
File 08 chỉ liệt kê checkpoint; không xoá file.

Khi Colab mất runtime, chạy lại 00–03 rồi 04 để khôi phục dataset và resume.
Nếu lượt train đã hoàn tất, phần 04 sẽ dừng với thông báo; không tự train lại.

## Dữ liệu và đường dẫn hiện tại

- Mount: `/CV_Object_Detection/drive`.
- Dự án Drive: `MyDrive/YOLO_Grocerystuffs`.
- Lượt train: `Grocerystuffs_Seg_20261029`.
- Snapshot: dataset Grocerystuffs gốc 918 ảnh, sau lọc còn 904 ảnh.
- Giữ nguyên hash, kiểm tra số lớp, số ảnh và split của notebook.

Các file không tự dùng bộ ảnh MoreData mới. Khi thay dataset cần cập nhật
snapshot và các kiểm tra liên quan trước khi train.
Chúng cần môi trường Google Colab và GPU; không chạy tuần tự bằng các tiến trình
`python file.py` riêng trên Windows vì các phần chia sẻ biến và dùng Google Drive.

Chuyển phần cài thư viện sang đọc requirements.txt và xử lý lời gọi async converter
để các file `.py` hợp lệ. Chưa chạy lại train/test khi tách file.
