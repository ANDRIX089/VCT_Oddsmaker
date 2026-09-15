"""Descarga el dataset VCT de Kaggle y conserva solo los años 2024-2026 en data/raw/."""
import shutil
from pathlib import Path

import kagglehub

DATASET = "ryanluong1/valorant-champion-tour-2021-2023-data"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
KEEP = {"all_ids", "vct_2024", "vct_2025", "vct_2026"}

if __name__ == "__main__":
    print(f"Descargando {DATASET} ...")
    cache_path = Path(kagglehub.dataset_download(DATASET))
    print(f"Descargado en cache: {cache_path}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for item in cache_path.iterdir():
        if item.name not in KEEP:
            continue
        dest = RAW_DIR / item.name
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        shutil.move(str(item), str(dest))
    print(f"Dataset ({', '.join(sorted(KEEP))}) listo en: {RAW_DIR}")
