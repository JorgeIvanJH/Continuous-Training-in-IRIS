import importlib
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


# Ensure imports like `python_utils.utils` resolve when tests are run from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_utils_with_fakes():
    """Import python_utils.utils with fake iris/mlflow modules preloaded."""
    logs = []

    class FakeSystem:
        @staticmethod
        def WriteToConsoleLog(message, prefix, severity):
            logs.append((message, prefix, severity))

    fake_mlflow = types.SimpleNamespace(
        sklearn=types.SimpleNamespace(
            load_model=lambda path: {"loaded_from": path},
            save_model=lambda model, path: None,
        ),
        set_tracking_uri=lambda uri: None,
    )

    fake_iris = types.SimpleNamespace(
        _SYS=types.SimpleNamespace(System=FakeSystem()),
        MLpipeline=types.SimpleNamespace(
            FeatureStore=types.SimpleNamespace(_New=lambda: None)
        ),
        cls=lambda name: types.SimpleNamespace(_New=lambda rs: None),
    )

    sys.modules["mlflow"] = fake_mlflow
    sys.modules["iris"] = fake_iris

    if "python_utils.utils" in sys.modules:
        del sys.modules["python_utils.utils"]

    import python_utils.utils as utils

    importlib.reload(utils)
    return utils, fake_mlflow, fake_iris, logs


def test_measure_time_decorator_returns_result_and_elapsed_time():
    utils, _, _, _ = _load_utils_with_fakes()

    @utils.measure_time_decorator
    def add(a, b):
        return a + b

    result, elapsed = add(2, 3)

    assert result == 5
    assert elapsed >= 0
    assert add.__name__ == "add"


def test_safe_model_load_success_first_try():
    utils, fake_mlflow, _, _ = _load_utils_with_fakes()

    sentinel_model = object()
    fake_mlflow.sklearn.load_model = lambda path: sentinel_model

    model = utils.safe_model_load("/models/run123")

    assert model is sentinel_model


def test_safe_model_load_retries_after_failure_and_uses_run_id(monkeypatch):
    utils, fake_mlflow, _, _ = _load_utils_with_fakes()

    sentinel_model = object()
    calls = {"count": 0, "run_id": None}

    def flaky_load(_path):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("first load failed")
        return sentinel_model

    def fake_resave(run_id):
        calls["run_id"] = run_id
        return True

    fake_mlflow.sklearn.load_model = flaky_load
    monkeypatch.setattr(utils, "save_mlflow_model", fake_resave)

    model = utils.safe_model_load("/tmp/some_run_id")

    assert model is sentinel_model
    assert calls["count"] == 2
    assert calls["run_id"] == "some_run_id"


def test_safe_model_load_returns_none_if_resave_fails(monkeypatch):
    utils, fake_mlflow, _, _ = _load_utils_with_fakes()

    fake_mlflow.sklearn.load_model = lambda _path: (_ for _ in ()).throw(RuntimeError("failed"))
    monkeypatch.setattr(utils, "save_mlflow_model", lambda _run_id: False)

    model = utils.safe_model_load("/tmp/failed_run")

    assert model is None


def test_iris_dbquery_rejects_invalid_identifiers():
    utils, _, _, _ = _load_utils_with_fakes()

    with pytest.raises(ValueError):
        utils.IRIS_DBQuery("bad-schema", "PointSample")


def test_iris_dbquery_returns_dataframe_from_feature_store():
    utils, _, fake_iris, _ = _load_utils_with_fakes()

    expected_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    class FakeFeatureStore:
        def DataExtraction(self, schema, tablename, columns, filters):
            assert schema == "SQLUser"
            assert tablename == "PointSample"
            assert columns == "*"
            assert filters == ""
            return "result_set"

    class FakeSQLResultSetClass:
        @staticmethod
        def _New(os_result_set):
            assert os_result_set == "result_set"
            return types.SimpleNamespace(dataframe=lambda: expected_df)

    fake_iris.MLpipeline = types.SimpleNamespace(
        FeatureStore=types.SimpleNamespace(_New=lambda: FakeFeatureStore())
    )
    fake_iris.cls = lambda name: FakeSQLResultSetClass()

    df = utils.IRIS_DBQuery("SQLUser", "PointSample")

    assert df.equals(expected_df)


def test_iris_dbquery_returns_empty_dataframe_when_data_extraction_fails():
    utils, _, fake_iris, _ = _load_utils_with_fakes()

    class BrokenFeatureStore:
        def DataExtraction(self, schema, tablename, columns, filters):
            raise RuntimeError("db unavailable")

    fake_iris.MLpipeline = types.SimpleNamespace(
        FeatureStore=types.SimpleNamespace(_New=lambda: BrokenFeatureStore())
    )

    df = utils.IRIS_DBQuery("SQLUser", "PointSample")

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_iris_dbquery_logs_warning_for_empty_dataframe():
    utils, _, fake_iris, logs = _load_utils_with_fakes()

    class EmptyFeatureStore:
        def DataExtraction(self, schema, tablename, columns, filters):
            return "empty_result_set"

    class FakeSQLResultSetClass:
        @staticmethod
        def _New(os_result_set):
            assert os_result_set == "empty_result_set"
            return types.SimpleNamespace(dataframe=lambda: pd.DataFrame())

    fake_iris.MLpipeline = types.SimpleNamespace(
        FeatureStore=types.SimpleNamespace(_New=lambda: EmptyFeatureStore())
    )
    fake_iris.cls = lambda name: FakeSQLResultSetClass()

    df = utils.IRIS_DBQuery("SQLUser", "PointSample")

    assert df.empty
    assert any("empty result set" in msg for msg, _, _ in logs)


def test_save_mlflow_model_success(monkeypatch):
    utils, fake_mlflow, fake_iris, _ = _load_utils_with_fakes()
    calls = {"tracking_uri": None, "load_uri": None, "save_path": None}

    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda: None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI_IRIS", "http://mlflow:5000")

    fake_mlflow.set_tracking_uri = lambda uri: calls.__setitem__("tracking_uri", uri)
    fake_mlflow.sklearn.load_model = lambda uri: calls.__setitem__("load_uri", uri) or "model"
    fake_mlflow.sklearn.save_model = (
        lambda model, path: calls.__setitem__("save_path", path)
    )

    fake_iris.cls = lambda name: types.SimpleNamespace(_GetParameter=lambda key: "/dur/models")

    ok = utils.save_mlflow_model("abc123")

    assert ok is True
    assert calls["tracking_uri"] == "http://mlflow:5000"
    assert calls["load_uri"] == "runs:/abc123/model"
    assert calls["save_path"].replace("\\", "/").endswith("/dur/models/abc123")


def test_save_mlflow_model_returns_false_on_failure():
    utils, fake_mlflow, _, _ = _load_utils_with_fakes()

    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda: None)
    fake_mlflow.sklearn.load_model = lambda _uri: (_ for _ in ()).throw(RuntimeError("broken"))

    ok = utils.save_mlflow_model("abc123")

    assert ok is False


def test_plot_inference_logs_error_when_model_load_fails():
    utils, fake_mlflow, _, logs = _load_utils_with_fakes()

    fake_mlflow.sklearn.load_model = lambda _path: (_ for _ in ()).throw(RuntimeError("cannot load"))

    dummy_run = types.SimpleNamespace(
        data=types.SimpleNamespace(tags={"mlflow.runName": "rname"}),
        info=types.SimpleNamespace(run_id="runid"),
    )
    dummy_self = types.SimpleNamespace(_GetParameter=lambda key: "'/dur/models'")

    utils.plot_inference(
        dummy_self,
        pd.Series([1, 2]),
        pd.Series([1, 2]),
        pd.Series([1, 2]),
        pd.Series([1, 2]),
        dummy_run,
        dummy_run,
    )

    assert any("Error in plot_inference" in msg and severity == 2 for msg, _, severity in logs)
