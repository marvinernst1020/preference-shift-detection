import numpy as np

from preference_shift.config import PrimaryConfig
from preference_shift.experiment import run_primary_experiment, save_primary_run


def test_primary_experiment_runs_end_to_end(tmp_path) -> None:
    config = PrimaryConfig(
        dimension=3,
        training_variances=(1.0, 0.3, 0.1),
        true_weights=(0.8, -0.4, 0.2),
        n_fit=250,
        n_calibration=150,
        horizon=80,
        change_point=35,
        transition_width=0,
        scenario="covariate",
        covariate_alignment_degrees=0.0,
        preference_alignment_degrees=45.0,
        mean_shift_magnitude=0.0,
        covariance_shift_magnitude=0.4,
        preference_shift_magnitude=0.5,
        prior_precision=1.0,
        quadrature_order=12,
        cusum_allowance=0.25,
        false_alarm_probability=0.1,
        null_calibration_runs=20,
        rolling_accuracy_window=15,
        risk_evaluation_samples=250,
        failure_excess_risk_increase=0.001,
        seed=101,
    )

    run = run_primary_experiment(config)
    assert run.time_series["time"].shape == (config.horizon,)
    assert run.arrays["differences"].shape == (config.horizon, config.dimension)
    assert np.isfinite(run.time_series["oracle_excess_risk"]).all()

    destination = save_primary_run(run, config, tmp_path / "result")
    assert (destination / "resolved_config.yaml").is_file()
    assert (destination / "summary.json").is_file()
    assert (destination / "time_series.csv").is_file()
    assert (destination / "run_arrays.npz").is_file()
