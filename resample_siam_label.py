import torchio as tio
from pathlib import Path
import shutil

# Source and destination directories
src_dir = Path('test_data/siam_label/v4/vascular')
dst_dir = Path('test_data/siam_label_resampled/v4/vascular')
dst_dir.mkdir(exist_ok=True)

# First, resample and save all NIfTI files
for nii_path in src_dir.rglob('*.nii*'):
    print("path", nii_path)
    img = tio.ScalarImage(nii_path)
    resampled = tio.Resample((0.5, 0.5, 0.5))(img)
    out_path = dst_dir / nii_path.relative_to(src_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resampled.save(out_path)
    print(f"Resampled and saved: {out_path}")

# Then, copy all other files (non-NIfTI) preserving directory structure
for file_path in src_dir.rglob('*'):
    if file_path.is_file() and not file_path.name.endswith(('.nii', '.nii.gz')):
        out_path = dst_dir / file_path.relative_to(src_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, out_path)
        print(f"Copied: {out_path}")
