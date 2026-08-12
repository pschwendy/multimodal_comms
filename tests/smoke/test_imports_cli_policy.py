import importlib
import pkgutil
import subprocess
import sys

import multimodal_comms
from multimodal_comms.cli.main import main


def test_import_every_public_module():
    for module in pkgutil.walk_packages(multimodal_comms.__path__, multimodal_comms.__name__ + "."):
        if module.name.endswith(("_test", "run_tests_fast", "run_tests_full")):
            continue
        if ".apps." in module.name and module.name != "multimodal_comms.apps.headless":
            # Full applications have explicitly documented optional
            # GUI/model/database dependencies. Their source and resources are
            # covered by experiments.smoke; the base import gate stays headless.
            continue
        importlib.import_module(module.name)


def test_cli_help_and_roundtrip(capsys):
    assert main(["roundtrip", "identity", "hello"]) == 0
    assert capsys.readouterr().out.strip() == "hello"


def test_repository_policy_script():
    result = subprocess.run(
        [sys.executable, "scripts/check_repository.py"], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
