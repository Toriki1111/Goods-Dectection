# Colab cell 0 | Grocerystuffs_Train.ipynb
# Run in order using %run -i in the same Colab runtime (see README.md).

# 0. Cai dung phien ban thu vien
import subprocess
import sys
from pathlib import Path

requirements_path = Path(__file__).resolve().parents[1] / 'requirements.txt'
assert requirements_path.is_file(), f'Khong tim thay requirements.txt: {requirements_path}'
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-r', str(requirements_path),
])
