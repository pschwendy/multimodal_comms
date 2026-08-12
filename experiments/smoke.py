"""Repository-wide experiment smoke harness.

Core mode is dependency-light and is the default local acceptance gate. Full mode also
launches ``--help`` for safe model-facing CLIs when their optional dependencies
are installed; it never contacts a service or database.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROGRAM_ROOTS = (ROOT / "experiments", ROOT / "training")

SAFE_FULL_HELP = (
    ("training.programs.pretrain_autoencoder", ("torch", "transformers")),
    ("training.programs.pretrain_mwnot_autoencoder", ("torch", "transformers")),
    ("training.programs.pretrain_packed", ("torch", "transformers")),
    ("training.programs.pretrain_superpose", ("torch", "transformers")),
    ("training.programs.train_selector", ()),
    ("training.programs.train_counterfactual_scorer", ()),
    ("training.programs.train_repmatch_selector", ()),
    ("training.benchmark_adaptation.comma.train_repmatch_selector", ()),
    ("training.benchmark_adaptation.iagents.train_repmatch_selector", ()),
    ("training.programs.train_rewriter_grpo", ("datasets", "openai")),
    ("training.programs.train_tokenfilter_pg", ("torch",)),
    ("experiments.packing.programs.eval_packing", ("torch",)),
    ("experiments.packing.programs.eval_multimodal", ("torch",)),
    ("experiments.iagents.programs.run_offline_eval", ("google.generativeai",)),
    ("experiments.iagents.programs.regrade", ()),
    ("experiments.crypt_ae.programs.eval_cryptae", ("torch", "transformers")),
    ("experiments.compressed_sensing.programs.demo_cs", ("skimage",)),
    ("experiments.compressed_sensing.programs.demo_cs_embedding", ("torch", "skimage")),
    ("experiments.compressed_sensing.programs.demo_cs_vlm", ("torch", "skimage")),
)


def _python_programs() -> list[Path]:
    return sorted(
        path
        for root in PROGRAM_ROOTS
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def check_sources() -> dict[str, int]:
    python_files = _python_programs()
    source_modules = sorted((ROOT / "src").rglob("*.py"))
    shell_files = sorted(path for root in PROGRAM_ROOTS for path in root.rglob("*.sh"))
    configs = sorted(path for root in PROGRAM_ROOTS for path in root.rglob("*.yaml"))
    configs += sorted(path for root in PROGRAM_ROOTS for path in root.rglob("*.yml"))

    for path in source_modules + python_files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        compile(tree, str(path), "exec")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module.startswith(("multimodal_comms", "experiments", "training")):
                    module_path = Path(*node.module.split("."))
                    candidates = (
                        ROOT / "src" / module_path.with_suffix(".py"),
                        ROOT / "src" / module_path / "__init__.py",
                        ROOT / module_path.with_suffix(".py"),
                        ROOT / module_path / "__init__.py",
                    )
                    if not any(candidate.exists() for candidate in candidates):
                        raise RuntimeError(f"unresolved internal import {node.module} in {path}")

    for path in shell_files:
        subprocess.run(["bash", "-n", str(path)], check=True)

    for path in configs:
        yaml.safe_load(path.read_text(encoding="utf-8"))
    for path in sorted((ROOT / "experiments" / "hiddenbench" / "configs").glob("*.json")):
        json.loads(path.read_text(encoding="utf-8"))

    return {
        "source_modules": len(source_modules),
        "python_programs": len(python_files),
        "shell_programs": len(shell_files),
        "configs": len(configs),
    }


def check_real_kernels() -> None:
    import numpy as np

    from multimodal_comms.benchmarks.hiddenbench.runtime.channel import build_channel
    from multimodal_comms.benchmarks.hiddenbench.runtime.task import DiscussionMessage
    from multimodal_comms.methods.packing import BlockPacker, RotorPacker
    from multimodal_comms.methods.sensing import CompressedSensingCodec
    from multimodal_comms.apps.collab_overcooked.overcooked_ai_py.mdp.actions import Action
    from multimodal_comms.apps.collab_overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    from multimodal_comms.apps.collab_overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

    history = [DiscussionMessage(agent_id=0, round_num=0, content="meet at noon")]
    channel = build_channel("full_history", "identity")
    assert channel.discussion_view(history, receiver_id=1, seen_count=0)[0]["content"] == "meet at noon"

    codes = {0: np.array([1.0, -2.0]), 1: np.array([3.0, 4.0])}
    for packer in (BlockPacker(4, 2), RotorPacker(4, 2, layout_seed=7)):
        packet = packer.pack(codes)
        for slot, code in codes.items():
            np.testing.assert_allclose(packer.unpack(packet, slot), code, atol=1e-7)

    codec = CompressedSensingCodec(np.eye(4), np.eye(4))
    matrix = np.eye(4)
    np.testing.assert_allclose(codec.decode(codec.encode(matrix, seed=3), seed=3), matrix)

    mdp = OvercookedGridworld.from_layout_name("new_env")
    environment = OvercookedEnv(mdp, horizon=2)
    _, reward, done, _ = environment.step((Action.STAY, Action.STAY))
    assert reward == 0 and not done


def check_behavioral_validation(full: bool = False) -> None:
    from experiments.validation.sensing_behavior import run_validation as run_sensing
    from experiments.validation.training_dynamics import run_validation as run_training

    sensing = run_sensing(require_natural_image=full)
    assert sensing["dense_embeddings"]["omp_relative_l2"] > 0.9
    training = run_training(include_torch=full)
    assert training["linear_supervised"]["reduction_fraction"] > 0.9


def check_resources() -> None:
    required = (
        "src/multimodal_comms/benchmarks/hiddenbench/data/hiddenbench_official",
        "src/multimodal_comms/apps/comma/config/prompts.json",
        "src/multimodal_comms/apps/collab_overcooked/prompts",
        "src/multimodal_comms/apps/collab_overcooked/overcooked_ai_py/data/layouts/new_env.layout",
        "src/multimodal_comms/apps/iagents/templates",
        "src/multimodal_comms/apps/iagents/static",
    )
    missing = [item for item in required if not (ROOT / item).exists()]
    if missing:
        raise RuntimeError(f"missing application resources: {missing}")


def check_full_cli_help() -> dict[str, int]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")))
    checked = skipped = 0
    for module, dependencies in SAFE_FULL_HELP:
        if any(importlib.util.find_spec(dependency) is None for dependency in dependencies):
            skipped += 1
            continue
        subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=30,
            check=True,
        )
        checked += 1
    return {"full_help_checked": checked, "full_help_skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("core", "full"), default="core")
    args = parser.parse_args()
    counts = check_sources()
    check_resources()
    check_real_kernels()
    check_behavioral_validation(full=args.profile == "full")
    optional = {}
    if args.profile == "full":
        optional = check_full_cli_help()
    print(
        json.dumps(
            {"profile": args.profile, **counts, **optional, "status": "passed"},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
