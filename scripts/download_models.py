from pathlib import Path
from huggingface_hub import snapshot_download


BASE_DIR = Path(__file__).resolve().parent.parent / "models"


MODELS = {
    "bge": {
        "repo": "BAAI/bge-m3",
        "path": BASE_DIR / "embeddings" / "bge-m3",
    },
    "reranker": {
        "repo": "BAAI/bge-reranker-v2-m3",
        "path": BASE_DIR / "reranker" / "bge-reranker-v2-m3",
    },
}


def download_model(name, repo, path):
    if path.exists() and any(path.iterdir()):
        print(f"{name}: already exists")
        return

    print(f"Downloading {name}...")

    snapshot_download(
        repo_id=repo,
        local_dir=str(path),
        local_dir_use_symlinks=False,
    )

    print(f"{name}: done")


if __name__ == "__main__":
    for name, cfg in MODELS.items():
        cfg["path"].mkdir(parents=True, exist_ok=True)

        download_model(
            name,
            cfg["repo"],
            cfg["path"]
        )