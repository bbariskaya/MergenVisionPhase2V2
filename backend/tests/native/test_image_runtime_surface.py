import json
import pathlib

import pytest

image_runtime = pytest.importorskip("image_runtime")

REPO_ROOT = pathlib.Path(__file__).parents[3]
PROFILE_PATH = REPO_ROOT / "backend" / "config" / "model_profiles" / "retinaface_r50_glintr100_v1.example.json"


def _load_profile():
    with PROFILE_PATH.open() as fh:
        return json.load(fh)


def test_image_runtime_module_is_importable():
    assert hasattr(image_runtime, "ImageRuntime")
    assert hasattr(image_runtime.ImageRuntime, "infer_jpeg")


def test_image_runtime_has_bound_constructor_signature():
    sig = image_runtime.ImageRuntime.__init__.__doc__ or ""
    assert "model_profile" in sig
    assert "retinaface_engine_path" in sig
    assert "glintr100_engine_path" in sig
    assert "device_id" in sig
    assert "context_pool_size" in sig


def test_image_runtime_rejects_profile_missing_required_fields():
    with pytest.raises(RuntimeError):
        image_runtime.ImageRuntime({}, "", "", 0, 1)


def test_image_runtime_rejects_profile_with_wrong_tensor_shape():
    profile = _load_profile()
    profile["detector"]["input_shape"] = [1, 3, 320]
    with pytest.raises(RuntimeError):
        image_runtime.ImageRuntime(profile, "", "", 0, 1)


def test_image_runtime_parses_alignment_crop_size_as_list() -> None:
    """crop_size is [h, w] in the canonical profile, not a scalar int."""
    profile = _load_profile()
    assert profile["alignment"]["crop_size"] == [112, 112]
    # Passing the valid profile must fail on engine path, not on profile parsing.
    with pytest.raises(RuntimeError) as exc_info:
        image_runtime.ImageRuntime(profile, "", "", 0, 1)
    assert "engine" in str(exc_info.value).lower() or "load" in str(exc_info.value).lower()
