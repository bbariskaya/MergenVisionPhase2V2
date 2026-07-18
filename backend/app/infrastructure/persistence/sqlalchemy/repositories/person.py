"""SQLAlchemy person repository."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities.person import Person
from app.domain.errors import ConcurrentUpdateError
from app.domain.value_objects import PersonId
from app.infrastructure.persistence.sqlalchemy.models.person import PersonOrm


def _to_domain(orm: PersonOrm) -> Person:
    return Person(
        person_id=PersonId(orm.person_id),
        display_name=orm.display_name,
        person_metadata=dict(orm.person_metadata or {}),
        is_active=orm.is_active,
        version=orm.version,
        created_at=orm.created_at,
        updated_at=orm.updated_at,
        deleted_at=orm.deleted_at,
    )


class SqlAlchemyPersonRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, person: Person) -> None:
        orm = PersonOrm(
            person_id=person.person_id,
            display_name=person.display_name,
            person_metadata=person.person_metadata,
            is_active=person.is_active,
            version=person.version,
            created_at=person.created_at,
            updated_at=person.updated_at,
            deleted_at=person.deleted_at,
        )
        self._session.add(orm)

    async def add_many(self, people: Sequence[Person]) -> None:
        orms = [
            PersonOrm(
                person_id=person.person_id,
                display_name=person.display_name,
                person_metadata=person.person_metadata,
                is_active=person.is_active,
                version=person.version,
                created_at=person.created_at,
                updated_at=person.updated_at,
                deleted_at=person.deleted_at,
            )
            for person in people
        ]
        self._session.add_all(orms)

    async def get_by_id(self, person_id: PersonId) -> Person | None:
        result = await self._session.execute(
            select(PersonOrm).where(PersonOrm.person_id == person_id)
        )
        orm = result.scalar_one_or_none()
        return _to_domain(orm) if orm else None

    async def get_active_by_id(self, person_id: PersonId) -> Person | None:
        result = await self._session.execute(
            select(PersonOrm)
            .where(PersonOrm.person_id == person_id)
            .where(PersonOrm.is_active.is_(True))
        )
        orm = result.scalar_one_or_none()
        return _to_domain(orm) if orm else None

    async def update(self, person: Person) -> None:
        result = await self._session.execute(
            select(PersonOrm).where(PersonOrm.person_id == person.person_id)
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            raise ValueError(f"Person {person.person_id} not found")
        orm.display_name = person.display_name
        orm.person_metadata = person.person_metadata
        orm.is_active = person.is_active
        orm.version = person.version
        orm.updated_at = person.updated_at
        orm.deleted_at = person.deleted_at

    async def update_with_expected_version(
        self,
        person: Person,
        expected_version: int,
    ) -> Person:
        result = await self._session.execute(
            update(PersonOrm)
            .where(
                PersonOrm.person_id == person.person_id,
                PersonOrm.version == expected_version,
            )
            .values(
                display_name=person.display_name,
                person_metadata=person.person_metadata,
                is_active=person.is_active,
                version=PersonOrm.version + 1,
                updated_at=person.updated_at,
                deleted_at=person.deleted_at,
            )
            .returning(PersonOrm.version)
        )
        new_version = result.scalar_one_or_none()
        if new_version is None:
            raise ConcurrentUpdateError(
                f"Person {person.person_id} was modified concurrently"
            )
        return Person(
            person_id=PersonId(person.person_id),
            display_name=person.display_name,
            person_metadata=dict(person.person_metadata),
            is_active=person.is_active,
            version=new_version,
            created_at=person.created_at,
            updated_at=person.updated_at,
            deleted_at=person.deleted_at,
        )

    async def list_all(self) -> list[Person]:
        result = await self._session.execute(
            select(PersonOrm).order_by(desc(PersonOrm.created_at))
        )
        return [_to_domain(orm) for orm in result.scalars().all()]

    async def search(
        self,
        query: str | None = None,
        is_active: bool = True,
    ) -> list[Person]:
        stmt = select(PersonOrm)
        if is_active:
            stmt = stmt.where(PersonOrm.is_active.is_(True))
        if query:
            stmt = stmt.where(
                func.lower(PersonOrm.display_name).contains(query.lower())
            )
        stmt = stmt.order_by(desc(PersonOrm.created_at))
        result = await self._session.execute(stmt)
        return [_to_domain(orm) for orm in result.scalars().all()]
