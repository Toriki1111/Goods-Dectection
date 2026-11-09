"""Run the project's YOLO checkpoint on a local webcam."""

import argparse
import json
import math
import threading
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import cv2
from ultralytics import YOLO


def parse_camera_source(value):
    """Accept a local camera index or an HTTP(S) phone video stream."""
    if value.isascii() and value.isdecimal():
        return int(value)
    url = urlsplit(value)
    if url.scheme in ('http', 'https') and url.hostname:
        return value
    raise argparse.ArgumentTypeError('Use a camera index (0, 1...) or an http(s) video URL.')


class LatestFrameCamera:
    """Drain a phone stream continuously; replace stale frames instead of queuing them."""

    def __init__(self, camera):
        self.camera = camera
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.frame = None
        self.sequence = self.consumed = 0
        self.finished = False
        self.worker = threading.Thread(target=self._capture, daemon=True)
        self.worker.start()

    def _capture(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.camera.read()
                if not ok:
                    break
                with self.condition:
                    self.frame = frame
                    self.sequence += 1
                    self.condition.notify_all()
        finally:
            self.camera.release()
            with self.condition:
                self.finished = True
                self.condition.notify_all()

    def read(self):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence > self.consumed or self.finished or self.stop.is_set(),
                timeout=6,
            )
            if self.sequence <= self.consumed:
                return False, None
            self.consumed = self.sequence
            return True, self.frame

    def release(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        # HTTP reads have a five-second timeout. The worker owns capture.release().
        self.worker.join(timeout=6)


class DetectionCounts:
    """Keep a snapshot of detections, refreshed every ten seconds."""

    def __init__(self):
        self.counts = Counter()
        self.updated_at = None
        self.interval = 10.0

    def update(self, class_ids, now):
        if self.updated_at is None or now - self.updated_at >= self.interval:
            self.counts = Counter(int(class_id) for class_id in class_ids)
            self.updated_at = now

    def seconds_remaining(self, now):
        if self.updated_at is None:
            return 0
        return max(0, math.ceil(self.interval - (now - self.updated_at)))


def load_prices(path):
    """Read exact class names and nonnegative integer VND prices."""
    prices = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(prices, dict) or any(
        not isinstance(name, str) or type(price) is not int or price < 0
        for name, price in prices.items()
    ):
        raise ValueError('Prices must map class names to nonnegative integer VND amounts.')
    return prices


def calculate_bill(counts, names, prices):
    """Bill one snapshot, keeping unpriced objects separate from the subtotal."""
    rows = []
    total = unpriced = 0
    for class_id, count in sorted(counts.items()):
        if count <= 0:
            continue
        name = names[class_id]
        unit_price = prices.get(name)
        amount = count * unit_price if unit_price is not None else None
        rows.append((name, count, unit_price, amount))
        if amount is None:
            unpriced += count
        else:
            total += amount
    return rows, total, unpriced


def draw_counts(preview, names, snapshot, now, prices=None):
    """Show a receipt for the current ten-second detection snapshot."""
    rows, total, unpriced = calculate_bill(snapshot.counts, names, prices or {})
    panel_width = 570
    panel_height = 245 + 27 * len(rows)
    height, width = preview.shape[:2]
    preview = cv2.copyMakeBorder(
        preview, 0, max(0, panel_height - height), 0, panel_width,
        cv2.BORDER_CONSTANT, value=(30, 30, 30),
    )
    x = width + 15

    def text(label, y, color=(230, 230, 230), scale=0.55, offset=0):
        cv2.putText(preview, label, (x + offset, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    text('GROCERY SCAN - DEMO PRICES', 30, (0, 255, 255), 0.65)
    text(f'Refresh in: {snapshot.seconds_remaining(now)}s', 58)
    text('Current snapshot / not cumulative', 82)
    text(f'Objects: {sum(snapshot.counts.values())}', 111)
    text(f'TOTAL: {total:,} VND', 146, (0, 255, 0), 0.85)
    text(f'Unpriced: {unpriced} item(s), excluded from total', 174,
         (0, 190, 255) if unpriced else (160, 160, 160), 0.49)
    text('Item', 207)
    text('Qty', 207, offset=215)
    text('Unit VND', 207, offset=275)
    text('Amount VND', 207, offset=405)
    for row, (name, count, unit_price, amount) in enumerate(rows):
        y = 237 + row * 27
        text(name.replace('_', ' ')[:22], y, scale=0.48)
        text(str(count), y, scale=0.48, offset=215)
        text(f'{unit_price:,}' if unit_price is not None else 'NO PRICE', y,
             scale=0.48, offset=275)
        text(f'{amount:,}' if amount is not None else '--', y,
             scale=0.48, offset=405)
    if not rows:
        text('No objects detected', 237, (160, 160, 160))
    return preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=parse_camera_source, default=0,
                        help='Local camera index or phone stream URL, e.g. http://PHONE_IP:4747/video')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=480,
                        help='Model input size (default: 480). Try 320 for speed or 640 for detail.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--prices', type=Path, default=Path(__file__).with_name('prices.json'))
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[1] / 'best_grocery.pt')
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f'Model not found: {args.model}')
    if not 0 <= args.conf <= 1 or args.imgsz <= 0:
        parser.error('conf must be in [0, 1] and imgsz must be positive')
    try:
        prices = load_prices(args.prices)
    except (OSError, ValueError) as exc:
        parser.error(f'Cannot load prices: {exc}')

    model = YOLO(str(args.model))
    snapshot = DetectionCounts()
    if isinstance(args.camera, str):
        camera = cv2.VideoCapture(args.camera, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
        ])
    else:
        camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(args.camera)
    try:
        if not camera.isOpened():
            raise RuntimeError('Cannot open camera. For a phone URL, keep the phone app open and close DroidCam Client on the PC to free the stream. Check phone IP and WiFi. For a local webcam, check camera permissions and index.')
        if isinstance(args.camera, int):
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            camera = LatestFrameCamera(camera)
            print('Phone mode: continuously receiving video, processing newest frames only.')
        print('Camera started. Press Q or Escape in the preview window to exit.')
        print(f'Model: {args.model} | device: {args.device} | imgsz: {args.imgsz} | conf: {args.conf}')
        print(f'Camera source: {args.camera}')
        print('Object counts refresh every 10 seconds from the current frame (not cumulative).')
        print(f'Demo prices: {args.prices} | unpriced objects are excluded from total.')

        cv2.namedWindow('Grocerystuffs - Webcam', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Grocerystuffs - Webcam', 1280, 720)

        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError('Cannot read a frame from the camera.')
            started = time.perf_counter()
            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
            now = time.monotonic()
            if snapshot.updated_at is None or now - snapshot.updated_at >= snapshot.interval:
                class_ids = result.boxes.cls.cpu().tolist() if result.boxes is not None else []
                snapshot.update(class_ids, now)
            preview = result.plot()
            fps = 1 / max(time.perf_counter() - started, 1e-9)
            cv2.putText(preview, f'{fps:.1f} inference FPS | Q: quit', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            preview = draw_counts(preview, result.names, snapshot, now, prices)
            cv2.imshow('Grocerystuffs - Webcam', preview)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                break
            if cv2.getWindowProperty('Grocerystuffs - Webcam', cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
