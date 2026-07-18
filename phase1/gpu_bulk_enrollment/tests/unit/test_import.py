"""Static import/version checks for Phase 1 package."""

import mv_phase1_bulk
from mv_phase1_bulk import cli, config, manifest, pipeline


def test_version() -> None:
    assert mv_phase1_bulk.__version__ == "0.1.0"


def test_cli_app_exists() -> None:
    assert cli.app is not None


def test_settings_load() -> None:
    assert config.settings.model_version == "retinaface_r50_glintr100_v1"


def test_manifest_class() -> None:
    assert manifest.EnrollmentManifest is not None


def test_pipeline_class() -> None:
    assert pipeline.GpuFacePipeline is not None
