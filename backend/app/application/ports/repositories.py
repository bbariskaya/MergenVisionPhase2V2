"""Repository ports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.domain.entities.face_identity import FaceIdentity
from app.domain.entities.face_sample import FaceSample
from app.domain.entities.process_record import ProcessRecord
from app.domain.entities.recognition_result import RecognitionResult
from app.domain.value_objects import FaceId, ProcessId, SampleId


class FaceIdentityRepository(ABC):
    @abstractmethod
    async def add(self, identity: FaceIdentity) -> None: ...

    @abstractmethod
    async def get_by_id(self, face_id: FaceId) -> FaceIdentity | None: ...

    @abstractmethod
    async def update(self, identity: FaceIdentity) -> None: ...

    @abstractmethod
    async def list_all(self) -> Sequence[FaceIdentity]: ...


class FaceSampleRepository(ABC):
    @abstractmethod
    async def add(self, sample: FaceSample) -> None: ...

    @abstractmethod
    async def get_by_id(self, sample_id: SampleId) -> FaceSample | None: ...

    @abstractmethod
    async def list_active_by_face_id(self, face_id: FaceId) -> Sequence[FaceSample]: ...

    @abstractmethod
    async def update(self, sample: FaceSample) -> None: ...


class ProcessRepository(ABC):
    @abstractmethod
    async def add(self, process: ProcessRecord) -> None: ...

    @abstractmethod
    async def get_by_id(self, process_id: ProcessId) -> ProcessRecord | None: ...

    @abstractmethod
    async def update(self, process: ProcessRecord) -> None: ...

    @abstractmethod
    async def list_by_status(self, status: str) -> Sequence[ProcessRecord]: ...


class RecognitionResultRepository(ABC):
    @abstractmethod
    async def add(self, result: RecognitionResult) -> None: ...

    @abstractmethod
    async def list_by_process_id(self, process_id: ProcessId) -> Sequence[RecognitionResult]: ...
