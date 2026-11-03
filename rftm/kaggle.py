"""Finding inputs on Kaggle (unchanged from notebooks 05 / 05b)."""
import glob
import os
import zipfile


def find_input(name):
    """Locate a file from an attached dataset or notebook output, wherever Kaggle mounted it."""
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    assert hits, f"{name} not found under /kaggle/input: attach the dataset that holds it (Add Input)"
    if len(hits) > 1:
        print(f"note: {len(hits)} copies of {name}, using {hits[0]}")
    return hits[0]


def resolve_checkpoint(out="/kaggle/working/dncnn_last.pt"):
    """Find the DnCNN checkpoint (dataset shourjabanerjee/dcnn-last). A .pt file is a zip archive, and Kaggle
    unzips zip files uploaded as a dataset, so it can arrive as a folder holding data.pkl: rebuild the file then."""
    pts = sorted(glob.glob("/kaggle/input/**/dncnn*.pt", recursive=True)) or \
          sorted(glob.glob("/kaggle/input/**/*.pt", recursive=True) + glob.glob("/kaggle/input/**/*.pth", recursive=True))
    if pts:
        return pts[0]
    pkls = sorted(glob.glob("/kaggle/input/**/data.pkl", recursive=True))
    assert pkls, "no DnCNN checkpoint under /kaggle/input: Add Input -> Datasets -> shourjabanerjee/dcnn-last"
    root = os.path.dirname(pkls[0])
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:      # torch expects uncompressed records under one folder
        for dirpath, _, files in os.walk(root):
            for f in sorted(files):
                full = os.path.join(dirpath, f)
                z.write(full, os.path.join("archive", os.path.relpath(full, root)))
    print(f"rebuilt {out} from the unpacked folder {root}")
    return out
