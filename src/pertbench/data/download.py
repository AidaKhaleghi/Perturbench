import urllib.request
from pathlib import Path
import anndata as ad


DATASETS = {
    "norman2019": "https://zenodo.org/records/13350497/files/NormanWeissman2019_filtered.h5ad?download=1"
}
RAW_DIR = Path("data/raw")


def download(name: str, force: bool = False) -> Path:
    target_path = RAW_DIR / f"{name}.h5ad"
    if target_path.exists() and not force:
        print(f"Dataset {name} already exists at {target_path}. Skipping download.")
        return target_path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATASETS[name], target_path)
    print(f"Downloaded dataset {name} to {target_path}.")


if __name__ == "__main__":
    download("norman2019")

    adata = ad.read_h5ad(RAW_DIR / "norman2019.h5ad")
    print(adata)
