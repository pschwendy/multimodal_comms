from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 1_000_000
LARGE_RESOURCE_ALLOWLIST = {
    Path("src/multimodal_comms/apps/collab_overcooked/llama_tokenizer/tokenizer.json"),
    Path("src/multimodal_comms/apps/collab_overcooked/overcooked_ai_js/overcooked.js"),
    Path("src/multimodal_comms/apps/collab_overcooked/overcooked_ai_js/overcooked-window.js"),
    Path("src/multimodal_comms/apps/collab_overcooked/overcooked_ai_py/data/testing/lossless_state_featurization.pickle"),
    Path("src/multimodal_comms/apps/collab_overcooked/overcooked_ai_py/data/testing/state_featurization.pickle"),
    Path("src/multimodal_comms/apps/comma/modules/samples/wire.json"),
    Path("src/multimodal_comms/apps/comma/modules/ARIAL.TTF"),
    Path("src/multimodal_comms/apps/iagents/static/github_cover.svg"),
}
IGNORED = {
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "checkpoints",
    "data",
    "logs",
    "outputs",
    "reports",
    "wandb",
}
SECRET = re.compile(r"(?i)(api[_-]?key|secret|token)\s*[=:]\s*['\"][A-Za-z0-9_\-]{16,}")
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def tracked_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not any(part in IGNORED for part in path.relative_to(ROOT).parts):
            yield path


def main() -> int:
    failures = []
    for path in ROOT.rglob("*"):
        if path.is_symlink():
            try:
                path.resolve(strict=True).relative_to(ROOT)
            except (FileNotFoundError, ValueError):
                failures.append(f"symlink escapes or is broken: {path.relative_to(ROOT)}")
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_BYTES and relative not in LARGE_RESOURCE_ALLOWLIST:
            failures.append(f"oversized file: {relative} ({path.stat().st_size} bytes)")
        if path.suffix in {
            ".py", ".md", ".yaml", ".yml", ".toml", ".json", ".txt", ".sh",
            ".html", ".htm", ".css", ".xml", ".csv", ".ini", ".cfg", ".conf",
            ".ps1", ".ts", ".tsx", ".jsx",
        } or path.name in {"Dockerfile", "Makefile"}:
            text = path.read_text(errors="replace")
            secret_match = SECRET.search(text)
            if secret_match and not any(
                marker in secret_match.group(0).lower()
                for marker in ("your-", "your_", "local-")
            ):
                failures.append(f"possible secret: {relative}")
            if "methods" in relative.parts and re.search(
                r"(?:from|import) multimodal_comms\.benchmarks", text
            ):
                failures.append(f"method imports benchmark: {relative}")
            if CJK.search(text):
                failures.append(f"CJK text is not permitted: {relative}")
    method_ids = {
        match.group(1)
        for match in re.finditer(
            r'^\s+"([a-z0-9_]+)": \(',
            (ROOT / "src/multimodal_comms/registry.py").read_text(),
            re.MULTILINE,
        )
    }
    documented_methods = {
        path.parent.name
        for path in (ROOT / "src/multimodal_comms/methods").rglob("README.md")
    }
    for method_id in method_ids:
        if method_id not in documented_methods:
            failures.append(f"missing colocated method README: {method_id}")
    experiment_families = (
        "autoencoders",
        "selectors",
        "gkd",
        "packing",
        "mwnot",
        "crypt_ae",
        "compressed_sensing",
        "hiddenbench",
        "comma",
        "collab_overcooked",
        "iagents",
    )
    for family in experiment_families:
        directory = ROOT / "experiments" / family
        if not (directory / "README.md").is_file():
            failures.append(f"missing experiment README: {family}")
        if not (directory / "run_full.sh").is_file():
            failures.append(f"missing full trajectory launcher: {family}")
    if not (ROOT / "training" / "README.md").is_file():
        failures.append("missing training/README.md")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("repository policy checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
