"""Unit tests for gallery decision semantics."""

from __future__ import annotations

import numpy as np

from mergenvision_video_lab.gallery import match_cluster_to_gallery


def _emb(seed: int) -> np.ndarray:
    """Deterministic unit embedding from a small integer seed."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_single_identity_gallery_is_insufficient_for_strict_margin() -> None:
    """One-identity gallery cannot satisfy top1/top2 margin semantics."""
    cluster_template = _emb(1)
    gallery = {
        "identities": {
            "Rachel": {
                "template": _emb(1),
                "samples": ["Rachel/1.jpg"],
                "selected_samples": ["Rachel/1.jpg"],
                "rejected_samples": [],
                "valid_sample_count": 1,
            }
        },
        "rejected_images": [],
        "valid_identity_count": 1,
        "strict": False,
    }

    result = match_cluster_to_gallery(
        cluster_template,
        gallery,
        match_threshold=0.45,
        margin_threshold=0.05,
        min_identity_count_for_strict=2,
        min_samples_per_identity=2,
    )

    assert result is not None
    assert result["known"] is False
    assert result["top1_label"] == "Rachel"
    assert result["strict"] is True


def test_known_requires_margin_and_competitors() -> None:
    """Known is true only with threshold, margin and enough competitors."""
    cluster_template = _emb(1)
    gallery = {
        "identities": {
            "Rachel": {
                "template": _emb(1),
                "samples": ["Rachel/1.jpg", "Rachel/2.jpg"],
                "selected_samples": ["Rachel/1.jpg", "Rachel/2.jpg"],
                "rejected_samples": [],
                "valid_sample_count": 2,
            },
            "Monica": {
                "template": _emb(2),
                "samples": ["Monica/1.jpg", "Monica/2.jpg"],
                "selected_samples": ["Monica/1.jpg", "Monica/2.jpg"],
                "rejected_samples": [],
                "valid_sample_count": 2,
            },
        },
        "rejected_images": [],
        "valid_identity_count": 2,
        "strict": False,
    }

    result = match_cluster_to_gallery(
        cluster_template,
        gallery,
        match_threshold=0.45,
        margin_threshold=0.05,
        min_identity_count_for_strict=2,
        min_samples_per_identity=2,
    )

    assert result is not None
    assert result["known"] is True
    assert result["top1_label"] == "Rachel"
    assert result["margin"] >= 0.05


def test_below_threshold_is_unknown() -> None:
    """Low cosine yields unknown even with competitors."""
    cluster_template = _emb(3)
    gallery = {
        "identities": {
            "Rachel": {
                "template": _emb(1),
                "samples": ["Rachel/1.jpg", "Rachel/2.jpg"],
                "selected_samples": ["Rachel/1.jpg", "Rachel/2.jpg"],
                "rejected_samples": [],
                "valid_sample_count": 2,
            },
            "Monica": {
                "template": _emb(2),
                "samples": ["Monica/1.jpg", "Monica/2.jpg"],
                "selected_samples": ["Monica/1.jpg", "Monica/2.jpg"],
                "rejected_samples": [],
                "valid_sample_count": 2,
            },
        },
        "rejected_images": [],
        "valid_identity_count": 2,
        "strict": False,
    }

    result = match_cluster_to_gallery(
        cluster_template,
        gallery,
        match_threshold=0.45,
        margin_threshold=0.05,
        min_identity_count_for_strict=2,
        min_samples_per_identity=2,
    )

    assert result is not None
    assert result["known"] is False


def test_empty_gallery_returns_none() -> None:
    """No identities yields None."""
    cluster_template = _emb(1)
    gallery = {"identities": {}, "rejected_images": [], "valid_identity_count": 0, "strict": False}
    result = match_cluster_to_gallery(
        cluster_template,
        gallery,
        match_threshold=0.45,
        margin_threshold=0.05,
        min_identity_count_for_strict=2,
        min_samples_per_identity=2,
    )
    assert result is None
