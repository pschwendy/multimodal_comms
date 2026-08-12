from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.resources import files


@dataclass(frozen=True, slots=True)
class AppInfo:
    name: str
    database_required: bool = False
    network_required: bool = False


def _create(name: str):
    """Create a DB/network-independent Flask app without import-time side effects."""
    try:
        from flask import Flask, jsonify, render_template
    except ImportError as error:
        raise RuntimeError("application factories require the 'apps' extra (Flask)") from error
    template_root = files("multimodal_comms.apps.resources").joinpath("templates")
    app = Flask(name, template_folder=str(template_root))
    app.config.update(TESTING=True, DATABASE_ENABLED=False, EXTERNAL_APIS_ENABLED=False)

    @app.get("/health")
    def health():
        return jsonify(asdict(AppInfo(name)))

    @app.get("/")
    def index():
        return render_template("index.html", benchmark=name)

    return app


def create_hiddenbench_app():
    return _create("hiddenbench")


def create_comma_app():
    return _create("comma")


def create_overcooked_app():
    return _create("collab_overcooked")


def create_iagents_app():
    return _create("iagents")
