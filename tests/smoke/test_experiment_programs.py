from experiments.smoke import check_real_kernels, check_resources, check_sources


def test_every_experiment_program_has_static_smoke_coverage():
    counts = check_sources()
    assert counts["python_programs"] >= 100
    assert counts["shell_programs"] >= 20


def test_resources_and_real_kernels():
    check_resources()
    check_real_kernels()
