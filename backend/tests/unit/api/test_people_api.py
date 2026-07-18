"""Unit tests for the Person directory HTTP API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.api.controllers.person_controller import PersonController
from app.api.main import create_app
from app.api.routes.dependencies import get_person_controller
from app.domain.value_objects import PersonId


@dataclass
class _FakePerson:
    person_id: PersonId
    display_name: str
    person_metadata: dict[str, Any]
    is_active: bool = True
    created_at: Any = None
    updated_at: Any = None


@dataclass
class _FakeFace:
    face_id: str
    status: str
    display_name: str | None
    identity_metadata: dict[str, Any]
    created_at: Any = None
    updated_at: Any = None


class _FakePersonManagementService:
    def __init__(self) -> None:
        self._people: dict[PersonId, _FakePerson] = {}
        self._faces: dict[PersonId, list[_FakeFace]] = {}

    async def list_people(self, query: str | None = None) -> list[_FakePerson]:
        people = list(self._people.values())
        if query:
            people = [p for p in people if query.lower() in p.display_name.lower()]
        return people

    async def get_person(self, person_id: PersonId) -> _FakePerson | None:
        return self._people.get(person_id)

    async def get_person_with_faces(
        self,
        person_id: PersonId,
    ) -> tuple[_FakePerson | None, list[_FakeFace]]:
        person = self._people.get(person_id)
        if person is None:
            return None, []
        return person, self._faces.get(person_id, [])

    async def create_person(
        self,
        display_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> _FakePerson:
        person = _FakePerson(
            person_id=PersonId(UUID("018f0000-0000-7000-8000-000000000001")),
            display_name=display_name,
            person_metadata=metadata or {},
        )
        self._people[person.person_id] = person
        self._faces[person.person_id] = []
        return person

    async def create_people_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[_FakePerson]:
        people: list[_FakePerson] = []
        for idx, item in enumerate(items):
            person_id = PersonId(
                UUID(f"018f0000-0000-7000-8000-{idx + 1:012x}")
            )
            person = _FakePerson(
                person_id=person_id,
                display_name=item["display_name"].strip(),
                person_metadata=item.get("metadata") or {},
            )
            self._people[person_id] = person
            self._faces[person_id] = []
            people.append(person)
        return people

    async def update_person(
        self,
        person_id: PersonId,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> _FakePerson:
        person = self._people[person_id]
        if display_name is not None:
            person.display_name = display_name
        if metadata is not None:
            person.person_metadata = metadata
        return person

    async def deactivate_person(self, person_id: PersonId) -> None:
        person = self._people[person_id]
        person.is_active = False


@pytest.fixture
def fake_controller() -> PersonController:
    return PersonController(_FakePersonManagementService())


@pytest.fixture
def client(fake_controller: PersonController) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_person_controller] = lambda: fake_controller
    app.state.person_controller = fake_controller
    return TestClient(app)


def test_list_people_returns_created_people(client: TestClient) -> None:
    client.post("/api/v1/people", json={"display_name": "Alice", "metadata": {"dept": "eng"}})
    response = client.get("/api/v1/people")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["people"][0]["displayName"] == "Alice"
    assert data["people"][0]["isActive"] is True


def test_create_person_returns_summary(client: TestClient) -> None:
    response = client.post("/api/v1/people", json={"display_name": "Bob"})
    assert response.status_code == 201
    data = response.json()
    assert data["displayName"] == "Bob"
    assert "personId" in data


def test_get_person_detail_includes_faces(client: TestClient) -> None:
    created = client.post("/api/v1/people", json={"display_name": "Carol"})
    person_id = created.json()["personId"]

    response = client.get(f"/api/v1/people/{person_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["personId"] == person_id
    assert data["displayName"] == "Carol"
    assert data["faceCount"] == 0
    assert data["faces"] == []


def test_update_person_changes_name(client: TestClient) -> None:
    created = client.post("/api/v1/people", json={"display_name": "Dave"})
    person_id = created.json()["personId"]

    response = client.patch(f"/api/v1/people/{person_id}", json={"display_name": "David"})
    assert response.status_code == 200
    assert response.json()["displayName"] == "David"


def test_delete_person_deactivates(client: TestClient) -> None:
    created = client.post("/api/v1/people", json={"display_name": "Eve"})
    person_id = created.json()["personId"]

    response = client.delete(f"/api/v1/people/{person_id}")
    assert response.status_code == 200

    detail = client.get(f"/api/v1/people/{person_id}")
    assert detail.json()["isActive"] is False


def test_get_missing_person_returns_not_found(client: TestClient) -> None:
    response = client.get("/api/v1/people/018f0000-0000-7000-8000-0000000000ff")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "PERSON_NOT_FOUND"


def test_create_people_batch_returns_summaries(client: TestClient) -> None:
    response = client.post(
        "/api/v1/people/batch",
        json={
            "people": [
                {"display_name": "Batch Alice", "metadata": {"dept": "eng"}},
                {"display_name": "Batch Bob"},
            ]
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["count"] == 2
    assert {p["displayName"] for p in data["people"]} == {"Batch Alice", "Batch Bob"}
    assert data["people"][0]["isActive"] is True


def test_create_people_batch_empty_returns_validation_error(client: TestClient) -> None:
    response = client.post("/api/v1/people/batch", json={"people": []})
    assert response.status_code == 422


def test_create_people_batch_missing_name_returns_validation_error(client: TestClient) -> None:
    response = client.post(
        "/api/v1/people/batch",
        json={"people": [{"display_name": "Valid"}, {"metadata": {}}]},
    )
    assert response.status_code == 422
