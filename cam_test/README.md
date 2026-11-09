# Thu camera Grocerystuffs

Dung model segmentation `../best_grocery.pt` (31 lop san pham) va moi truong `../.venv` da cai trong du an.
Camera hien bounding box va mask; bang dem cap nhat moi 10 giay.

Gia demo luu trong `prices.json`, don vi VND, gan theo ten lop chinh xac.
Gia hien tai duoc doc tu JSON; khong con co dinh theo chu ky 10.000-50.000.
Moi 10 giay bang hien cac mon dang duoc nhan dien, so luong, don gia,
thanh tien va tong tien. Ket qua moi thay the ket qua cu, khong cong don.
Lop truoc Coca_Lon van duoc dem, hien NO PRICE va khong cong vao tong tien;
dong Unpriced cho biet so vat chua co gia. Khung hinh dau tien hien ngay.
Sua `prices.json` roi khoi dong lai de thay gia; co the chon file khac bang `--prices`.

Mo `run.cmd` de bat webcam mac dinh. Nhan Q hoac Escape trong cua so anh de thoat.
Chuong trinh khong ghi hinh hay luu frame.

Mac dinh model xu ly anh voi `imgsz=480` de tang FPS tren CPU.
Kich thuoc cua so van la 1280 x 720. Dau vao nho hon co the bo sot vat nho.
Toc do thuc te phu thuoc canh, may va ung dung dang chay.

Bang ben phai hien so luong tung loai vat va tong so vat model nhan dien duoc.
Bang hien ngay o khung hinh dau tien, sau do cap nhat moi 10 giay va co dem nguoc.
So dem lay tu khung hinh tai thoi diem cap nhat, khong cong don cac khung hinh.
Neu khung hinh luc cap nhat khong co vat nao duoc nhan dien, tat ca so dem ve 0.
Bounding box tren camera van cap nhat lien tuc; nguong `--conf` ap dung ca cho so dem.

Tu PowerShell tai D:\CV_AI:

```powershell
.\cam_test\run.cmd
```

Model YOLO26l-seg tu luot train 100 epoch ngay 29/10/2026 da dat tai
`D:\CV_AI\best_grocery.pt` va la mac dinh cua `run.cmd`.
Day la best checkpoint cua luot Grocerystuffs_Seg_20261029.
Muon dung lai model CommonThings, chay `\.venv\Scripts\python.exe cam_test\webcam.py --model best100.pt`.

Model cu `D:\CV_AI\best120.pt` duoc giu nguyen canh model moi.
De so sanh bang model cu, chay:

```powershell
.\cam_test\run_best120.cmd
```

Lenh `run_best120.cmd` dung rieng model cu `best120.pt`; neu chua co file thi dung va bao loi.
File nay la best checkpoint o epoch 96 cua luot train toi da 120 epochs,
dung som o epoch 116. Ten `best120.pt` la ten luu theo yeu cau.

Camera khac:

```powershell
.\cam_test\run.cmd --camera 1
```

Dien thoai Android qua WiFi, khong can OBS hoac camera ao Windows:
giu app DroidCam tren dien thoai mo; dong DroidCam Client tren PC de tranh
`DroidCam is Busy`. Hai thiet bi cung mang. Dung IP hien tren app dien thoai:

```powershell
.\cam_test\run_phone.cmd PHONE_IP
```

IP co the doi khi doi WiFi. Neu app dung port khac 4747, truyen URL day du:
`run.cmd --camera "http://PHONE_IP:PORT/video"`.

Che do dien thoai nhan video trong luong rieng va chi giu frame moi nhat,
bo qua frame cu khi YOLO xu ly cham hon camera. Tat cua so cam test cu bang Q
va chay lai de ap dung; FPS model khong nhat thiet tang nhung hinh giam bi tre
do hang doi. Neu van tre, kiem tra WiFi va thu video 640x480/30 FPS tren app.

CPU cham thi thu dau vao 320 (co the thay doi do chinh xac):

```powershell
.\cam_test\run.cmd --imgsz 320
```

Can uu tien vat nho/chi tiet thi dung lai dau vao 640:

```powershell
.\cam_test\run.cmd --imgsz 640
```

Dieu chinh nguong du doan:

```powershell
.\cam_test\run.cmd --conf 0.4
```

Mac dinh chay CPU vi PyTorch hien tai la ban CPU. Chi dung `--device 0` sau khi da cai PyTorch CUDA va kiem tra GPU hoat dong.

Neu khong mo duoc camera, dong cac ung dung dang su dung camera va kiem tra quyen camera cho desktop apps trong Windows Settings.
