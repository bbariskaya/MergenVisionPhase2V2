# Third-Party Notices

This lab uses the following open-source components. Code licenses are separate from pretrained model weight licenses.

## Software

| Package | Version installed | License | Use |
|---|---|---|---|
| NumPy | (lock file) | BSD-3-Clause | Array math |
| SciPy | (lock file) | BSD-3-Clause | Linear sum assignment |
| PyAV | (lock file) | BSD-2-Clause | Video decode |
| OpenCV (headless) | (lock file) | Apache-2.0 | Image I/O, drawing, warpAffine |
| scikit-image | (lock file) | BSD-3-Clause | SimilarityTransform, metrics |
| ONNX | (lock file) | Apache-2.0 | Model graph inspection |
| ONNX Runtime | (lock file) | MIT | Inference engine |
| InsightFace | 0.7.3 | MIT | Reference detector/recognizer/alignment oracle |
| Pydantic | (lock file) | MIT | Data contracts |
| PyYAML | (lock file) | MIT | Config files |
| Typer | (lock file) | MIT | CLI |
| orjson | (lock file) | Apache-2.0 | Fast JSON |
| pandas | (lock file) | BSD-3-Clause | Diagnostics tables |
| matplotlib | (lock file) | PSF-based | Plots and histograms |
| Pillow | (lock file) | HPND | Contact sheets |
| psutil | (lock file) | BSD-3-Clause | RSS measurement |
| rich | (lock file) | MIT | CLI output |
| pytest | (lock file) | MIT | Testing |

## Reference implementations

- **InsightFace**: <https://github.com/deepinsight/insightface> — RetinaFace detection, ArcFace/GlintR100 recognition, `face_align.py` five-point alignment. Used only as an offline oracle.
- **FoundationVision/ByteTrack**: <https://github.com/FoundationVision/ByteTrack> — Kalman filter, track lifecycle, two-stage association. Adapted for face metadata tracking.

## Models / pretrained weights

The lab expects these local artifacts to be provided by the user and never downloads them:

- `backend/artifacts/models/retinaface_r50_dynamic.onnx` — RetinaFace detector.
- `backend/artifacts/models/glintr100.onnx` — Glint360K R100 ArcFace-style recognizer.

Pretrained model weights are **not** covered by the open-source code licenses above. Their use is restricted to this offline reference experiment; production use requires separate license verification.

## Dependency range adjustments

The spec originally required:

- `Pillow>=10,<12`
- `opencv-python-headless>=4.10,<5`

Installing `insightface==0.7.3` on Python 3.12.3 resolved to:

- `Pillow==12.3.0`
- `opencv-python-headless==5.0.0.93`

Therefore the lab `pyproject.toml` uses the relaxed ranges `Pillow>=10,<13` and `opencv-python-headless>=4.10,<6`. The exact installed versions are captured in `requirements.lock`.
