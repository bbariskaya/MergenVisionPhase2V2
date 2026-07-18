"""HTTP-facing controllers for the Person directory API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from app.api.schemas import (
    IdentitySummary,
    PeopleBatchCreateRequest,
    PeopleBatchCreateResponse,
    PersonDetailResponse,
    PersonListResponse,
    PersonSummary,
)
from app.application.services.person_management_service import PersonManagementService
from app.domain.entities.face_identity import FaceIdentity
from app.domain.entities.person import Person
from app.domain.errors import ValidationError
from app.domain.value_objects import PersonId


@dataclass(frozen=True)
class CreatePersonRequestData:
    display_name: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class UpdatePersonRequestData:
    display_name: str | None = None
    metadata: dict[str, Any] | None = None


class PersonController:
    def __init__(self, service: PersonManagementService) -> None:
        self._service = service

    async def list_people(
        self,
        request_id: str,
        query: str | None = None,
    ) -> PersonListResponse:
        people = await self._service.list_people(query=query)
        return PersonListResponse(
            request_id=request_id,
            count=len(people),
            people=[self._to_summary(p) for p in people],
        )

    async def get_person(
        self,
        request_id: str,
        person_id_str: str,
    ) -> PersonDetailResponse | None:
        person_id = self._parse_person_id(person_id_str)
        person, faces = await self._service.get_person_with_faces(person_id)
        if person is None:
            return None
        return PersonDetailResponse(
            request_id=request_id,
            person_id=str(person.person_id),
            display_name=person.display_name,
            is_active=person.is_active,
            metadata=dict(person.person_metadata),
            face_count=len(faces),
            faces=[self._to_identity_summary(f) for f in faces],
            created_at=self._format_dt(person.created_at),
            updated_at=self._format_dt(person.updated_at),
        )

    async def create_person(
        self,
        request_id: str,
        data: CreatePersonRequestData,
    ) -> PersonSummary:
        person = await self._service.create_person(
            display_name=data.display_name,
            metadata=data.metadata,
        )
        return self._to_summary(person)

    async def create_people_batch(
        self,
        request_id: str,
        data: PeopleBatchCreateRequest,
    ) -> PeopleBatchCreateResponse:
        people = await self._service.create_people_batch(
            items=[
                {"display_name": item.display_name, "metadata": item.metadata}
                for item in data.people
            ],
        )
        return PeopleBatchCreateResponse(
            request_id=request_id,
            count=len(people),
            people=[self._to_summary(p) for p in people],
        )

    async def update_person(
        self,
        request_id: str,
        person_id_str: str,
        data: UpdatePersonRequestData,
    ) -> PersonSummary:
        person_id = self._parse_person_id(person_id_str)
        person = await self._service.update_person(
            person_id=person_id,
            display_name=data.display_name,
            metadata=data.metadata,
        )
        return self._to_summary(person)

    async def deactivate_person(
        self,
        request_id: str,
        person_id_str: str,
    ) -> None:
        person_id = self._parse_person_id(person_id_str)
        await self._service.deactivate_person(person_id)

    @staticmethod
    def _parse_person_id(value: str) -> PersonId:
        try:
            return PersonId(UUID(value))
        except ValueError as exc:
            raise ValidationError("person_id must be a valid UUID") from exc

    def _to_summary(self, person: Person) -> PersonSummary:
        return PersonSummary(
            person_id=str(person.person_id),
            display_name=person.display_name,
            is_active=person.is_active,
            created_at=self._format_dt(person.created_at),
            updated_at=self._format_dt(person.updated_at),
        )

    def _to_identity_summary(self, identity: FaceIdentity) -> IdentitySummary:
        return IdentitySummary(
            face_id=str(identity.face_id),
            status=identity.status,
            name=identity.display_name,
            metadata=dict(identity.identity_metadata),
            created_at=self._format_dt(identity.created_at),
            updated_at=self._format_dt(identity.updated_at),
        )

    @staticmethod
    def _format_dt(value: datetime | None) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
