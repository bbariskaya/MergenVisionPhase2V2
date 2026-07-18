"""Integration tests for PersonManagementService."""

from __future__ import annotations

import pytest

from app.application.services.person_management_service import PersonManagementService
from app.domain.errors import ValidationError

pytestmark = pytest.mark.asyncio


@pytest.fixture
def service(unit_of_work_factory: object) -> PersonManagementService:
    return PersonManagementService(unit_of_work_factory=unit_of_work_factory)


async def test_create_people_batch_persists_multiple_people(service: PersonManagementService) -> None:
    people = await service.create_people_batch(
        items=[
            {"display_name": "Alpha", "metadata": {"dept": "eng"}},
            {"display_name": "Beta"},
            {"display_name": " Gamma "},
        ]
    )
    assert len(people) == 3
    assert [p.display_name for p in people] == ["Alpha", "Beta", "Gamma"]

    listed = await service.list_people()
    assert len(listed) == 3


async def test_create_people_batch_rejects_empty_list(service: PersonManagementService) -> None:
    with pytest.raises(ValidationError, match="batch cannot be empty"):
        await service.create_people_batch(items=[])


async def test_create_people_batch_rejects_missing_display_name(service: PersonManagementService) -> None:
    with pytest.raises(ValidationError, match="display_name is required"):
        await service.create_people_batch(
            items=[{"display_name": "Valid"}, {"metadata": {}}]
        )


async def test_create_people_batch_enforces_max_size(service: PersonManagementService) -> None:
    with pytest.raises(ValidationError, match="batch size cannot exceed 1000"):
        await service.create_people_batch(
            items=[{"display_name": f"Person {i}"} for i in range(1001)]
        )
