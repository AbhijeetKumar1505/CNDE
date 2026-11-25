"""Single source of truth for paths and tunable settings (shared by main.ipynb and app.py).

Storage root resolution order:
  1. CNDE_ROOT environment variable, if set.
  2. /content/drive/MyDrive/CNDE_Hackathon when Google Drive is mounted (persistent).
  3. <repo>/workspace locally (git-ignored).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
DRIVE_ROOT = Path("/content/drive/MyDrive/CNDE_Hackathon")


def _default_root() -> Path:
    if os.environ.get("CNDE_ROOT"):
        return Path(os.environ["CNDE_ROOT"])
    if DRIVE_ROOT.parent.exists():
        return DRIVE_ROOT
    return REPO_DIR / "workspace"


@dataclass
class Config:
    root: Path = field(default_factory=_default_root)

    # Models
    llm_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    llm_fallback_id: str = "Qwen/Qwen2.5-1.5B-Instruct"  # only if the 4B model fails on the T4
    embed_id: str = "BAAI/bge-small-en-v1.5"
    embed_device: str = "auto"  # CUDA if available: bge-small is tiny, and CPU embedding measured ~6 chunks/s locally

    # Chunking (counted in BGE tokens; BGE silently truncates anything above 512)
    chunk_tokens: int = 350
    chunk_overlap: int = 50

    # Retrieval
    top_k: int = 5
    min_retrieval_score: float = 0.65  # abstention gate. Provisional: probes scored 0.78-0.83 in-domain vs 0.55 out-of-scope; tune on benchmark

    # Generation
    max_input_tokens: int = 4096
    max_new_tokens: int = 512

    @property
    def docs_dir(self) -> Path:
        return self.root / "documents" / "raw"

    @property
    def index_dir(self) -> Path:
        return self.root / "indexes"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def benchmark_file(self) -> Path:
        return REPO_DIR / "benchmarks" / "questions.jsonl"

    def make_dirs(self) -> None:
        for d in (self.docs_dir, self.index_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)
