"""Bounded input validation for uploaded images.

Validates byte limits, container magic, and header-reported dimensions *before*
any CPU full-frame decode or GPU cudaMalloc.  The parser only inspects JPEG
marker headers.
"""

from __future__ import annotations

from app.domain.errors import InvalidMediaError, PayloadTooLargeError, UnsupportedMediaTypeError

_JPEG_SOI = b"\xff\xd8"
# SOF markers that carry frame dimensions.
_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _read_jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Parse width/height from the first SOF marker without decoding scan data."""
    if len(data) < 2 or data[:2] != _JPEG_SOI:
        raise ValueError("not a JPEG stream")

    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xD9:  # EOI
            break
        if marker == 0x00 or (0xD0 <= marker <= 0xD7):  # padding / restart
            i += 2
            continue
        if i + 3 >= len(data):
            break
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        if marker in _SOF_MARKERS:
            sof_start = i + 4
            if sof_start + 7 > len(data):
                raise ValueError("truncated SOF marker")
            # precision = data[sof_start]
            height = int.from_bytes(data[sof_start + 1 : sof_start + 3], "big")
            width = int.from_bytes(data[sof_start + 3 : sof_start + 5], "big")
            if width == 0 or height == 0:
                raise ValueError("invalid JPEG dimensions")
            return width, height
        # Skip this segment.
        i += 2 + length

    raise ValueError("JPEG dimensions not found")


class ImageValidator:
    """Validates a complete image byte buffer."""

    def __init__(
        self,
        max_image_bytes: int,
        max_image_width: int = 8192,
        max_image_height: int = 8192,
        max_image_pixels: int = 67_108_864,
    ) -> None:
        self._max_image_bytes = max_image_bytes
        self._max_image_width = max_image_width
        self._max_image_height = max_image_height
        self._max_image_pixels = max_image_pixels

    def validate(self, data: bytes) -> tuple[int, int]:
        """Validate image bytes and return (width, height).

        Raises:
            InvalidMediaError: empty, corrupt, or dimension/pixel violations.
            PayloadTooLargeError: byte limit exceeded.
            UnsupportedMediaTypeError: not a JPEG stream.
        """
        if not isinstance(data, bytes):
            raise InvalidMediaError("image data must be bytes")
        if len(data) == 0:
            raise InvalidMediaError("image data is empty")
        if len(data) > self._max_image_bytes:
            raise PayloadTooLargeError(
                f"image exceeds maximum size of {self._max_image_bytes} bytes"
            )
        if data[:2] != _JPEG_SOI:
            raise UnsupportedMediaTypeError("only JPEG images are supported")
        try:
            width, height = _read_jpeg_dimensions(data)
        except ValueError as exc:
            raise InvalidMediaError(f"invalid or unsupported JPEG: {exc}") from exc

        if width > self._max_image_width or height > self._max_image_height:
            raise InvalidMediaError(
                f"image dimensions {width}x{height} exceed the maximum allowed "
                f"{self._max_image_width}x{self._max_image_height}"
            )
        pixels = width * height
        if pixels > self._max_image_pixels:
            raise InvalidMediaError(
                f"image pixel count {pixels} exceeds the maximum {self._max_image_pixels}"
            )
        return width, height
