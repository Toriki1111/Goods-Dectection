import concurrent.futures
import io
import json
import time
from collections import Counter
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'datasets/commonthings_v1'


def main():
    rows = [json.loads(s) for s in (ROOT / 'data/commonthings.ndjson').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    header = next(r for r in rows if r['type'] == 'dataset')
    names = header['class_names']
    records = [r for r in rows if r['type'] == 'image']
    for split in ('train', 'val', 'test'):
        for folder in ('images', 'labels'):
            (OUT / folder / split).mkdir(parents=True, exist_ok=True)

    def download(r):
        name, split = r['file'], r['split']
        if Path(name).name != name or split not in ('train', 'val', 'test'):
            return {'file': name, 'error': 'Unsafe filename or split'}
        dest = OUT / 'images' / split / name
        error = ''
        for attempt in range(3):
            try:
                if dest.exists():
                    data = dest.read_bytes()
                else:
                    response = requests.get(r['url'], timeout=45)
                    response.raise_for_status()
                    data = response.content
                with Image.open(io.BytesIO(data)) as im:
                    im.load()
                    if im.size != (r['width'], r['height']):
                        raise ValueError(f'Dimension mismatch: {im.size}')
                if not dest.exists():
                    dest.write_bytes(data)
                labels = []
                for c, x, y, w, h in r.get('annotations', {}).get('boxes', []):
                    if str(c) not in names or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        raise ValueError('Invalid box')
                    labels.append(f'{c} {x:.8f} {y:.8f} {w:.8f} {h:.8f}')
                (OUT / 'labels' / split / Path(name).with_suffix('.txt')).write_text('\n'.join(labels) + ('\n' if labels else ''), encoding='utf-8')
                return {'file': name, 'split': split, 'boxes': len(labels)}
            except Exception as exc:
                error = str(exc)
                time.sleep(attempt + 1)
        return {'file': name, 'error': error}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for result in pool.map(download, records):
            results.append(result)
            if len(results) % 200 == 0:
                print(f'Processed {len(results)}/{len(records)}; failures {sum("error" in r for r in results)}', flush=True)
    failures = [r for r in results if 'error' in r]
    good = {r['file'] for r in results if 'error' not in r}
    report = {'expected_images': len(records), 'downloaded_verified': len(good), 'boxes': sum(r.get('boxes', 0) for r in results), 'splits': dict(Counter(r['split'] for r in results if 'error' not in r)), 'empty_labels': sum(r.get('boxes') == 0 for r in results if 'error' not in r), 'failures': failures}
    (OUT / 'preparation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if failures:
        print(json.dumps(report, indent=2), flush=True)
        raise SystemExit('Incomplete dataset: resolve failures before training. No data.yaml generated.')
    yaml = 'path: ' + OUT.as_posix() + '\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'
    yaml += ''.join(f'  {k}: {json.dumps(v)}\n' for k, v in names.items())
    (OUT / 'data.yaml').write_text(yaml, encoding='utf-8')
    previews = OUT / 'review'
    previews.mkdir(exist_ok=True)
    selected = []
    for c in names:
        match = next((r for r in records if r['file'] in good and any(str(b[0]) == c for b in r.get('annotations', {}).get('boxes', [])) and r not in selected), None)
        if match:
            selected.append(match)
    for page, group in [('labeled_samples', selected), ('empty_label_samples', [r for r in records if not r.get('annotations', {}).get('boxes', [])][:12])]:
        canvas = Image.new('RGB', (4 * 360, 3 * 390), '#202020')
        for i, r in enumerate(group):
            im = Image.open(OUT / 'images' / r['split'] / r['file']).convert('RGB')
            draw = ImageDraw.Draw(im)
            for c, x, y, w, h in r.get('annotations', {}).get('boxes', []):
                coords = ((x-w/2)*im.width, (y-h/2)*im.height, (x+w/2)*im.width, (y+h/2)*im.height)
                draw.rectangle(coords, outline='#ff4040', width=2)
                draw.text((max(0, coords[0]), max(0, coords[1])), names[str(c)], fill='white', stroke_width=1, stroke_fill='black')
            im = ImageOps.contain(im, (350, 350))
            ox, oy = (i % 4)*360, (i // 4)*390
            canvas.paste(im, (ox, oy))
            ImageDraw.Draw(canvas).text((ox+4, oy+352), f'{r["split"]} | {r["file"][:40]}', fill='white')
        canvas.save(previews / f'{page}.jpg', quality=90)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
