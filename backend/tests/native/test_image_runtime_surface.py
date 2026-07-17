import pytest

image_runtime = pytest.importorskip("image_runtime")


def test_image_runtime_module_is_importable():
    assert hasattr(image_runtime, "ImageRuntime")
    assert hasattr(image_runtime.ImageRuntime, "infer_jpeg")


def test_image_runtime_has_bound_constructor_signature():
    sig = image_runtime.ImageRuntime.__init__.__doc__ or ""
    assert "model_profile_path" in sig
    assert "retinaface_engine_path" in sig
    assert "glintr100_engine_path" in sig
    assert "device_id" in sig
    assert "context_pool_size" in sig
