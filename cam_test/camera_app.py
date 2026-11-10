"""Small Windows camera source picker; video runs in the existing OpenCV window."""

import ipaddress
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from receipts import SNAPSHOT_PREFIX, freeze_bill, new_invoice, write_pdf
from camera_devices import list_devices, resolve_device, device_labels

ROOT = Path(__file__).resolve().parent
SETTINGS = ROOT / 'camera_settings.json'
MODES = ('Device', 'Wi-Fi')


def build_source(mode, index, ip, port):
    if mode == 'Device':
        if not index.isascii() or not index.isdecimal():
            raise ValueError('Camera index must be a nonnegative number, e.g. 0 or 1.')
        return str(int(index))
    if mode not in MODES:
        raise ValueError('Select a connection mode.')
    try:
        address = ipaddress.ip_address(ip.strip())
        number = int(port)
        if not 1 <= number <= 65535:
            raise ValueError()
    except ValueError:
        raise ValueError('Enter a valid phone IP address and port (1–65535).') from None
    host = f'[{address}]' if address.version == 6 else str(address)
    return f'http://{host}:{number}/video'


def load_settings(path=SETTINGS):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        settings = {k: v for k, v in data.items()
                    if k in ('mode', 'index', 'ip', 'port', 'device_id') and isinstance(v, str)}
        if settings.get('mode') == 'Hotspot':
            settings['mode'] = 'Wi-Fi'
        return settings
    except (OSError, ValueError, AttributeError):
        return {}


def save_settings(path, settings):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    temporary.replace(path)


def launch_command(source, preview=False, device_id=None):
    command = [sys.executable, '-u', str(ROOT / 'webcam.py'), '--camera', source]
    if preview:
        command.append('--preview')
    if device_id is not None:
        command.extend(['--camera-id', device_id])
    return command


class CameraApp:
    def __init__(self, window):
        self.window = window
        self.process = None
        self.events = queue.Queue()
        self.device_events = queue.Queue()
        self.devices = []
        self.scanning = False
        self.tail = []
        self.receipt = None
        self.stopping = self.closing = False
        settings = load_settings()
        self.selected_device_id = settings.get('device_id', '')
        self.device_name = tk.StringVar()
        self.mode = tk.StringVar(value=settings.get('mode') if settings.get('mode') in MODES else 'Device')
        self.index = tk.StringVar(value=settings.get('index', '0'))
        self.ip = tk.StringVar(value=settings.get('ip', ''))
        self.port = tk.StringVar(value=settings.get('port', '4747'))
        self.status = tk.StringVar(value='Ready. Select a source, then preview or start YOLO.')
        self.hint = tk.StringVar()
        window.title('Grocerystuffs — Camera connection')
        window.geometry('720x530')
        window.minsize(620, 440)
        panel = ttk.Frame(window, padding=22)
        panel.pack(fill='both', expand=True)
        ttk.Label(panel, text='Connect your camera', font=('Segoe UI', 18, 'bold')).pack(anchor='w')
        modes = ttk.Frame(panel)
        modes.pack(fill='x', pady=16)
        for value, label in zip(MODES, ('Device camera', 'Phone camera / IP (DroidCam)')):
            ttk.Radiobutton(modes, text=label, value=value, variable=self.mode,
                            command=self.update_mode).pack(side='left', padx=(0, 18))
        fields = ttk.Frame(panel)
        fields.pack(fill='x')
        self.inputs = []
        for row, (label, variable) in enumerate((('Camera device', self.device_name), ('Phone IP', self.ip), ('Port', self.port))):
            ttk.Label(fields, text=label).grid(row=row, column=0, sticky='w', pady=6)
            entry = (ttk.Combobox(fields, textvariable=variable, state='readonly', width=33)
                     if row == 0 else ttk.Entry(fields, textvariable=variable, width=35))
            entry.grid(row=row, column=1, sticky='ew', padx=16, pady=6)
            self.inputs.append(entry)
        self.inputs[0].bind('<<ComboboxSelected>>', self.select_device)
        self.refresh_button = ttk.Button(fields, text='Refresh', command=self.refresh_devices)
        self.refresh_button.grid(row=0, column=2, padx=(0, 4))
        fields.columnconfigure(1, weight=1)
        ttk.Label(panel, textvariable=self.hint, wraplength=610).pack(fill='x', pady=16)
        buttons = ttk.Frame(panel)
        buttons.pack(fill='x', pady=8)
        self.preview_button = ttk.Button(buttons, text='Preview camera', command=lambda: self.start(True))
        self.start_button = ttk.Button(buttons, text='Start YOLO', command=self.start)
        self.stop_button = ttk.Button(buttons, text='Stop', command=self.stop, state='disabled')
        self.export_button = ttk.Button(buttons, text='Export invoice PDF', command=self.export_pdf, state='disabled')
        for button in (self.preview_button, self.start_button, self.stop_button, self.export_button):
            button.pack(side='left', padx=(0, 12))
        ttk.Separator(panel).pack(fill='x', pady=15)
        ttk.Label(panel, textvariable=self.status, wraplength=610).pack(fill='x')
        self.bill_status = tk.StringVar(value='No scan available for export.')
        ttk.Label(panel, textvariable=self.bill_status, wraplength=610).pack(fill='x', pady=8)
        ttk.Label(panel, text='Close the video window or press Q / Escape to return.\n'
                  'Preview does not load YOLO. Latest-frame capture is used in every mode.',
                  wraplength=610).pack(fill='x', pady=12)
        self.update_mode()
        self.refresh_devices()
        window.protocol('WM_DELETE_WINDOW', self.close)
        window.after(100, self.poll)

    def update_mode(self):
        device = self.mode.get() == 'Device'
        for entry, enabled in zip(self.inputs, (device, not device, not device)):
            entry.configure(state='normal' if enabled else 'disabled')
        self.inputs[0].configure(state='readonly' if device and not self.scanning and self.devices else 'disabled')
        self.refresh_button.configure(state='normal' if device and not self.scanning and self.process is None else 'disabled')
        self.hint.set({
            'Device': 'Select a system camera by name. Click Refresh after connecting or disconnecting a camera. Preview confirms that the selected device can open.',
            'Wi-Fi': 'Use the same Wi-Fi network, or connect the PC to the phone hotspot. Open DroidCam on the phone and enter its IP/port. Exit DroidCam Client on the PC.',
        }[self.mode.get()])

    def select_device(self, event=None):
        position = self.inputs[0].current()
        if 0 <= position < len(self.devices):
            self.selected_device_id = self.devices[position]['device_id']

    def refresh_devices(self):
        if self.scanning or self.process is not None:
            return
        self.scanning = True
        self.status.set('Finding system cameras...')
        self.update_mode()

        def discover():
            try:
                self.device_events.put((list_devices(), None))
            except Exception as exc:
                self.device_events.put(([], str(exc)))
        threading.Thread(target=discover, daemon=True).start()

    def start(self, preview=False):
        if self.process is not None:
            return
        try:
            device_id = None
            if self.mode.get() == 'Device':
                if self.scanning or not self.selected_device_id:
                    raise ValueError('Select a camera from the device list. Click Refresh if no camera is shown.')
                selected = resolve_device(self.selected_device_id, self.devices)
                source, device_id = str(selected['index']), selected['device_id']
            else:
                source = build_source(self.mode.get(), self.index.get(), self.ip.get(), self.port.get())
            settings = {k: v.get() for k, v in (
                ('mode', self.mode), ('index', self.index), ('ip', self.ip), ('port', self.port))}
            settings['device_id'] = self.selected_device_id
            save_settings(SETTINGS, settings)
            self.process = subprocess.Popen(
                launch_command(source, preview, device_id), cwd=ROOT.parent,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace',
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except (ValueError, OSError) as exc:
            messagebox.showerror('Cannot start camera', str(exc), parent=self.window)
            return
        self.tail = []
        self.receipt = None
        self.export_button.configure(state='disabled')
        self.bill_status.set('Waiting for a YOLO scan...' if not preview else 'Preview only - no invoice.')
        self.stopping = False
        self.preview_button.configure(state='disabled')
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.update_mode()
        self.status.set('Opening camera...' if preview else 'Loading model and opening camera...')
        threading.Thread(target=self.read_output, args=(self.process,), daemon=True).start()

    def read_output(self, process):
        with process.stdout:
            for line in process.stdout:
                self.events.put((process, line.strip()))
        process.wait()

    def export_pdf(self):
        try:
            if self.receipt is None:
                raise ValueError('Wait for a YOLO scan before exporting.')
            # Freeze before opening the save dialog; subsequent scans cannot alter it.
            invoice = new_invoice(self.receipt)
            folder = ROOT.parent / 'exports' / 'receipts'
            folder.mkdir(parents=True, exist_ok=True)
            path = filedialog.asksaveasfilename(parent=self.window, title='Save invoice PDF',
                initialdir=str(folder), initialfile=invoice['invoice_id'] + '.pdf',
                defaultextension='.pdf', filetypes=[('PDF document', '*.pdf')])
            if not path:
                return
            write_pdf(invoice, path)
            messagebox.showinfo('Invoice saved', str(path), parent=self.window)
        except (OSError, ValueError, KeyError, ImportError) as exc:
            messagebox.showerror('Cannot export invoice', str(exc), parent=self.window)

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.stopping = True
            self.status.set('Stopping camera...')
            self.process.terminate()
            process = self.process
            self.window.after(2000, lambda: process.kill() if process.poll() is None else None)

    def close(self):
        self.closing = True
        self.stop()
        if self.process is None:
            self.window.destroy()

    def poll(self):
        while True:
            try:
                devices, error = self.device_events.get_nowait()
            except queue.Empty:
                break
            self.scanning = False
            self.devices = devices
            self.inputs[0].configure(values=device_labels(devices))
            position = next((i for i, device in enumerate(devices)
                             if device['device_id'] == self.selected_device_id), None)
            if position is None and not self.selected_device_id and devices:
                position = 0
            if position is None:
                self.device_name.set('')
            else:
                self.inputs[0].current(position)
                self.select_device()
            self.update_mode()
            if error:
                self.status.set('Cannot list cameras: ' + error)
            elif not devices:
                self.status.set('No system cameras found. Connect a camera and click Refresh, or use Phone camera / IP.')
            elif position is None:
                self.status.set('Previous camera is disconnected. Select another device or reconnect it and Refresh.')
            else:
                self.status.set(f'{len(devices)} camera(s) found. Select a device and click Preview camera.')
        while True:
            try:
                process, line = self.events.get_nowait()
            except queue.Empty:
                break
            if process is not self.process:
                continue
            if line:
                if line.startswith(SNAPSHOT_PREFIX):
                    try:
                        data = json.loads(line[len(SNAPSHOT_PREFIX):])
                        self.receipt = freeze_bill(data['rows'], data['total'], data['unpriced'], data['scanned_at'])
                        self.export_button.configure(state='normal' if self.receipt['rows'] else 'disabled')
                        self.bill_status.set(f"Scan: {data['scanned_at']} | Total: {data['total']:,} VND | Unpriced: {data['unpriced']}")
                    except (ValueError, KeyError, TypeError):
                        self.receipt = None
                        self.export_button.configure(state='disabled')
                    continue
                self.tail = (self.tail + [line])[-10:]
                if not self.stopping:
                    self.status.set(line[-300:])
        if self.process is not None and self.process.poll() is not None:
            code = self.process.returncode
            self.process = None
            self.preview_button.configure(state='normal')
            self.start_button.configure(state='normal')
            self.stop_button.configure(state='disabled')
            self.update_mode()
            self.status.set('Stopped.' if self.stopping else 'Camera closed.' if code == 0 else 'Connection or model error. Check the source and try again.')
            if code and not self.stopping and not self.closing:
                messagebox.showerror('Camera failed', '\n'.join(self.tail[-5:]) or 'Cannot open camera.', parent=self.window)
        if self.closing and self.process is None:
            self.window.destroy()
            return
        self.window.after(100, self.poll)


def main():
    window = tk.Tk()
    CameraApp(window)
    window.mainloop()


if __name__ == '__main__':
    main()
