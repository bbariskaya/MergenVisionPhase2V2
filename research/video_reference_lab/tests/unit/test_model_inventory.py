"""Unit tests for model inventory and provider selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from mergenvision_video_lab.errors import ModelArtifactError
from mergenvision_video_lab.model_inventory import (
    ModelInventory,
    available_providers,
    build_model_contract,
    resolve_model_root,
    select_providers,
)


def test_resolve_model_root_defaults_to_backend_artifacts() -> None:
    root = resolve_model_root(None)
    assert root.name == "models"
    assert root.parent.name == "artifacts"


def test_resolve_model_root_relative() -> None:
    root = resolve_model_root("backend/artifacts/models")
    assert root.name == "models"


def test_inventory_discovers_existing_models() -> None:
    inventory = ModelInventory(
        model_root="backend/artifacts/models",
        model_pack=None,
    )
    assert inventory.detector_path.name == "retinaface_r50_dynamic.onnx"
    assert inventory.recognizer_path.name == "glintr100.onnx"
    assert inventory.detector_contract.size_bytes > 0
    assert inventory.recognizer_contract.size_bytes > 0
    assert len(inventory.detector_contract.sha256) == 64
    assert len(inventory.recognizer_contract.inputs) >= 1
    assert len(inventory.recognizer_contract.outputs) >= 1


def test_inventory_missing_detector() -> None:
    with pytest.raises(ModelArtifactError) as exc:
        ModelInventory(
            model_root="backend/artifacts/models",
            model_pack=None,
            detector_basenames=["not_there.onnx"],
        )
    assert "detector" in str(exc.value).lower()


def test_inventory_missing_recognizer() -> None:
    with pytest.raises(ModelArtifactError) as exc:
        ModelInventory(
            model_root="backend/artifacts/models",
            model_pack=None,
            recognizer_basenames=["not_there.onnx"],
        )
    assert "recognizer" in str(exc.value).lower()


def test_select_providers_cpu() -> None:
    providers = select_providers("cpu", ["CPUExecutionProvider"], allow_cpu_fallback=False)
    assert providers == ["CPUExecutionProvider"]


def test_select_providers_cuda_available() -> None:
    providers = select_providers(
        "cuda",
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        allow_cpu_fallback=False,
    )
    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_select_providers_cuda_unavailable_no_fallback() -> None:
    with pytest.raises(ModelArtifactError):
        select_providers(
            "cuda",
            ["CPUExecutionProvider"],
            allow_cpu_fallback=False,
        )


def test_select_providers_cuda_unavailable_with_fallback() -> None:
    providers = select_providers(
        "cuda",
        ["CPUExecutionProvider"],
        allow_cpu_fallback=True,
    )
    assert providers == ["CPUExecutionProvider"]


def test_available_providers_includes_cpu() -> None:
    providers = available_providers()
    assert "CPUExecutionProvider" in providers


def test_build_model_contract(tmp_path: Path) -> None:
    # Minimal ONNX model: single Add op.
    import onnx
    from onnx import TensorProto, helper

    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1])
    c = helper.make_tensor_value_info("c", TensorProto.FLOAT, [1])
    node = helper.make_node("Add", ["a", "b"], ["c"])
    graph = helper.make_graph([node], "test", [a, b], [c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = tmp_path / "add.onnx"
    onnx.save(model, str(path))

    contract = build_model_contract(path)
    assert contract.basename == "add.onnx"
    assert contract.size_bytes == path.stat().st_size
    assert len(contract.inputs) == 2
    assert len(contract.outputs) == 1
