import json

import pytest

from location_classifier.cli import main


@pytest.fixture()
def workspace(tmp_path):
    """A tmp workspace with a small dataset and a trained model already in it."""
    data = tmp_path / "locations.csv"
    model = tmp_path / "model.joblib"
    assert (
        main(
            [
                "--seed",
                "5",
                "make-data",
                "--sites",
                "metros",
                "--samples-per-class",
                "25",
                "--output",
                str(data),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--seed",
                "5",
                "--output-dir",
                str(tmp_path / "out"),
                "train",
                "--data",
                str(data),
                "--model",
                "knn",
                "--model-path",
                str(model),
                "--n-anchors",
                "5",
                "--no-cv",
            ]
        )
        == 0
    )
    return {"root": tmp_path, "data": data, "model": model, "out": tmp_path / "out"}


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_make_data_writes_a_csv(tmp_path, capsys):
    target = tmp_path / "zones.csv"
    code = main(
        ["make-data", "--sites", "zones", "--samples-per-class", "10", "--output", str(target)]
    )
    assert code == 0
    assert target.exists()
    output = capsys.readouterr().out
    assert "wrote 80 rows" in output
    assert "Colaba" in output


def test_make_data_is_reproducible(tmp_path):
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    main(["--seed", "3", "make-data", "--samples-per-class", "5", "--output", str(first)])
    main(["--seed", "3", "make-data", "--samples-per-class", "5", "--output", str(second)])
    assert first.read_text() == second.read_text()


def test_make_data_rejects_an_unknown_site_collection(capsys):
    with pytest.raises(SystemExit):
        main(["make-data", "--sites", "atlantis"])


def test_info_summarises_the_dataset(workspace, capsys):
    assert main(["info", "--data", str(workspace["data"])]) == 0
    output = capsys.readouterr().out
    assert "rows: 200" in output
    assert "classes: 8" in output


def test_info_on_a_missing_file_returns_one(tmp_path, capsys):
    assert main(["info", "--data", str(tmp_path / "gone.csv")]) == 1
    assert "error:" in capsys.readouterr().err


def test_train_saves_a_model_and_metadata(workspace):
    assert workspace["model"].exists()
    metadata = json.loads(workspace["model"].with_suffix(".json").read_text())
    assert metadata["model_name"] == "knn"
    assert len(metadata["classes"]) == 8


def test_train_rejects_an_unknown_model(workspace, capsys):
    code = main(["train", "--data", str(workspace["data"]), "--model", "nope"])
    assert code == 1
    assert "unknown model" in capsys.readouterr().err


def test_train_writes_plots_when_asked(tmp_path, workspace):
    out = tmp_path / "figures"
    code = main(
        [
            "--output-dir",
            str(out),
            "train",
            "--data",
            str(workspace["data"]),
            "--model-path",
            str(tmp_path / "m2.joblib"),
            "--n-anchors",
            "4",
            "--no-cv",
            "--plots",
        ]
    )
    assert code == 0
    assert (out / "dataset.png").exists()
    assert (out / "confusion_matrix.png").exists()


def test_train_runs_cross_validation_by_default(tmp_path, workspace, capsys):
    code = main(
        [
            "train",
            "--data",
            str(workspace["data"]),
            "--model-path",
            str(tmp_path / "m3.joblib"),
            "--n-anchors",
            "4",
            "--cv-folds",
            "3",
        ]
    )
    assert code == 0
    assert "cv accuracy" in capsys.readouterr().out


def test_evaluate_prints_metrics(workspace, capsys):
    code = main(
        [
            "evaluate",
            "--data",
            str(workspace["data"]),
            "--model-path",
            str(workspace["model"]),
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "accuracy" in output
    assert "model: knn" in output


def test_evaluate_can_emit_json(workspace, capsys):
    code = main(
        [
            "evaluate",
            "--data",
            str(workspace["data"]),
            "--model-path",
            str(workspace["model"]),
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_samples"] == 200
    assert payload["accuracy"] > 0.9


def test_evaluate_without_a_model_returns_one(tmp_path, workspace, capsys):
    code = main(
        [
            "evaluate",
            "--data",
            str(workspace["data"]),
            "--model-path",
            str(tmp_path / "absent.joblib"),
        ]
    )
    assert code == 1
    assert "model not found" in capsys.readouterr().err


def test_predict_single_point(workspace, capsys):
    code = main(
        [
            "predict",
            "--model-path",
            str(workspace["model"]),
            "--lat",
            "19.076",
            "--lon",
            "72.877",
            "--top-k",
            "2",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "Mumbai" in output
    assert "rank_2_location" in output


def test_predict_requires_both_coordinates(workspace, capsys):
    code = main(["predict", "--model-path", str(workspace["model"]), "--lat", "19.0"])
    assert code == 1
    assert "must be given together" in capsys.readouterr().err


def test_predict_from_csv_writes_output(workspace, tmp_path):
    target = tmp_path / "predictions.csv"
    code = main(
        [
            "predict",
            "--model-path",
            str(workspace["model"]),
            "--data",
            str(workspace["data"]),
            "--output",
            str(target),
        ]
    )
    assert code == 0
    assert target.exists()
    assert "predicted_location" in target.read_text().splitlines()[0]


def test_cluster_reports_a_summary(workspace, capsys):
    code = main(
        ["cluster", "--data", str(workspace["data"]), "--n-clusters", "8"]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "clusters: 8" in output
    assert "dominant_label" in output


def test_cluster_dbscan_and_plot(workspace, tmp_path, capsys):
    out = tmp_path / "cl"
    code = main(
        [
            "--output-dir",
            str(out),
            "cluster",
            "--data",
            str(workspace["data"]),
            "--method",
            "dbscan",
            "--eps-km",
            "20",
            "--min-samples",
            "5",
            "--plot",
        ]
    )
    assert code == 0
    assert (out / "clusters.png").exists()
    assert "method: dbscan" in capsys.readouterr().out


def test_compare_lists_every_model(workspace, capsys):
    code = main(["compare", "--data", str(workspace["data"])])
    assert code == 0
    output = capsys.readouterr().out
    for name in ("knn", "random_forest", "logistic_regression"):
        assert name in output


def test_invalid_test_size_is_reported(workspace, tmp_path, capsys):
    code = main(
        [
            "train",
            "--data",
            str(workspace["data"]),
            "--model-path",
            str(tmp_path / "m4.joblib"),
            "--test-size",
            "1.5",
        ]
    )
    assert code == 1
    assert "test_size" in capsys.readouterr().err


def test_environment_variables_configure_paths(workspace, monkeypatch, capsys):
    monkeypatch.setenv("LOC_CLF_DATA_PATH", str(workspace["data"]))
    assert main(["info"]) == 0
    assert "rows: 200" in capsys.readouterr().out
