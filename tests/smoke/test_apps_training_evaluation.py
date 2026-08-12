import numpy as np
from multimodal_comms.apps import (
    create_comma_app,
    create_hiddenbench_app,
    create_iagents_app,
    create_overcooked_app,
)
from multimodal_comms.evaluation import load_experiment, run_experiment
from multimodal_comms.training import LinearTrainer, TrainingBatch


def test_headless_application_factories_load_routes_and_templates():
    for factory in (
        create_hiddenbench_app,
        create_comma_app,
        create_overcooked_app,
        create_iagents_app,
    ):
        client = factory().test_client()
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 200


def test_training_one_step_and_checkpoint(tmp_path):
    trainer = LinearTrainer(2, 1, seed=3)
    batch = TrainingBatch(np.array([[1.0, 2.0]]), np.array([[1.0]]))
    assert trainer.step(batch) >= 0
    path = tmp_path / "tiny.json"
    trainer.save(path)
    loaded = LinearTrainer.load(path)
    assert np.array_equal(loaded.weights, trainer.weights)


def test_evaluation_spec_schema_and_run():
    spec = load_experiment("examples/smoke.yaml")
    result = run_experiment(spec)
    assert set(result) == {
        "method",
        "messages",
        "raw_bytes",
        "wire_bytes",
        "byte_ratio",
        "exact_messages",
    }
    assert result["exact_messages"] == 2
