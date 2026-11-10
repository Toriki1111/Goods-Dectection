"""Named Windows camera discovery; do not open devices during enumeration."""

from collections import Counter


def list_devices():
    import cv2
    from cv2_enumerate_cameras import enumerate_cameras
    return [dict(name=info.name, index=info.index, backend=info.backend,
                 device_id=info.path or 'name:' + info.name)
            for info in enumerate_cameras(cv2.CAP_DSHOW)]


def resolve_device(device_id, devices=None):
    matches = [device for device in (list_devices() if devices is None else devices)
               if device['device_id'] == device_id]
    if not matches:
        raise ValueError('Selected camera is no longer available. Refresh and select a camera again.')
    if len(matches) > 1:
        raise ValueError('Windows cannot distinguish these cameras by a unique identifier. Disconnect the duplicate device and click Refresh.')
    return matches[0]


def device_labels(devices):
    counts = Counter(device['name'] for device in devices)
    identifiers = Counter(device['device_id'] for device in devices)
    seen = Counter()
    labels = []
    for device in devices:
        name = device['name']
        seen[name] += 1
        label = name if counts[name] == 1 else f'{name} ({seen[name]})'
        if identifiers[device['device_id']] > 1:
            label += ' - unavailable: duplicate device identity'
        labels.append(label)
    return labels
