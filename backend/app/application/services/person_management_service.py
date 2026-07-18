"""Person management application service.

Read and maintain the global Person directory. Faces remain under the face
identity lifecycle service; this service is concerned with the Person aggregate
and its linked face identities.
"""

from __future__ import annotations

from typing import Any

from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.domain.entities.face_identity import FaceIdentity
from app.domain.entities.person import Person
from app.domain.errors import ValidationError
from app.domain.value_objects import PersonId


class PersonManagementService:
    def __init__(self, unit_of_work_factory: UnitOfWorkFactory) -> None:
        self._unit_of_work_factory = unit_of_work_factory

    async def list_people(
        self,
        query: str | None = None,
        include_inactive: bool = False,
    ) -> list[Person]:
        async with self._unit_of_work_factory() as uow:
            people = await uow.people.search(
                query=query,
                is_active=not include_inactive,
            )
        return list(people)

    async def get_person(self, person_id: PersonId) -> Person | None:
        async with self._unit_of_work_factory() as uow:
            return await uow.people.get_by_id(person_id)

    async def get_person_with_faces(
        self,
        person_id: PersonId,
    ) -> tuple[Person | None, list[FaceIdentity]]:
        async with self._unit_of_work_factory() as uow:
            person = await uow.people.get_by_id(person_id)
            if person is None:
                return None, []
            faces = await uow.face_identities.list_by_person_id(person_id)
        return person, list(faces)

    async def create_person(
        self,
        display_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Person:
        if not display_name or not display_name.strip():
            raise ValidationError("display_name is required")
        person = Person.create(
            display_name=display_name.strip(),
            person_metadata=metadata or {},
        )
        async with self._unit_of_work_factory() as uow:
            await uow.people.add(person)
            await uow.commit()
        return person

    async def create_people_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[Person]:
        if not items:
            raise ValidationError("batch cannot be empty")
        if len(items) > 1000:
            raise ValidationError("batch size cannot exceed 1000")
        people: list[Person] = []
        for idx, item in enumerate(items):
            display_name = (item.get("display_name") or "").strip()
            if not display_name:
                raise ValidationError(f"display_name is required at index {idx}")
            metadata = item.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise ValidationError(f"metadata must be an object at index {idx}")
            people.append(
                Person.create(
                    display_name=display_name,
                    person_metadata=metadata,
                )
            )
        async with self._unit_of_work_factory() as uow:
            await uow.people.add_many(people)
            await uow.commit()
        return people

    async def update_person(
        self,
        person_id: PersonId,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Person:
        async with self._unit_of_work_factory() as uow:
            person = await uow.people.get_by_id(person_id)
            if person is None:
                raise ValidationError("Person not found")
            if display_name is not None:
                person.rename(display_name.strip())
            if metadata is not None:
                person.update_metadata(metadata)
            await uow.people.update(person)
            await uow.commit()
        return person

    async def deactivate_person(self, person_id: PersonId) -> None:
        async with self._unit_of_work_factory() as uow:
            person = await uow.people.get_by_id(person_id)
            if person is None:
                raise ValidationError("Person not found")
            person.deactivate()
            await uow.people.update(person)
            await uow.commit()
