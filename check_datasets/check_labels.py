#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
import torchio as tio

def check_labelmap(path: Path, expected_max: int = 201) -> bool:
    label_map = tio.LabelMap(path)
    tensor = label_map.data.int()
    unique_vals = torch.unique(tensor)
    expected = torch.arange(0, expected_max + 1, dtype=torch.int)

    if not torch.equal(unique_vals.cpu(), expected):
        missing = set(expected.tolist()) - set(unique_vals.tolist())
        extra   = set(unique_vals.tolist()) - set(expected.tolist())
        print(f"[FAIL] {path.relative_to(root)}: {len(unique_vals)} labels")
        if missing:
            print(f"   Manquants: {sorted(missing)}")
        if extra:
            print(f"   Supplémentaires: {sorted(extra)}")
        return False
    else:
        print(f"[OK]   {path.relative_to(root)}: 0..{expected_max} ({len(unique_vals)} labels)")
        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: check_labels.py <root_dir>")
        sys.exit(1)

    root = Path(sys.argv[1])
    files = sorted(root.glob("gen*/Lab*"))

    if not files:
        print("Aucun fichier 'Lab*' trouvé dans gen*/")
        sys.exit(0)

    count_ok = 0
    nb_files = 0
    for f in files:
        ok = check_labelmap(f)
        if ok:
            count_ok +=1
        nb_files += 1
    
    print(f"=== fichiers {count_ok} / {nb_files} ===")
