"""SQLAlchemy repository adapters."""

from app.infrastructure.persistence.sqlalchemy.repositories.face_identity import (
    SqlAlchemyFaceIdentityRepository,
)
from app.infrastructure.persistence.sqlalchemy.repositories.face_sample import (
    SqlAlchemyFaceSampleRepository,
)
from app.infrastructure.persistence.sqlalchemy.repositories.person import (
    SqlAlchemyPersonRepository,
)
from app.infrastructure.persistence.sqlalchemy.repositories.process_record import (
    SqlAlchemyProcessRepository,
)
from app.infrastructure.persistence.sqlalchemy.repositories.recognition_result import (
    SqlAlchemyRecognitionResultRepository,
)

__all__ = [
    "SqlAlchemyFaceIdentityRepository",
    "SqlAlchemyFaceSampleRepository",
    "SqlAlchemyPersonRepository",
    "SqlAlchemyProcessRepository",
    "SqlAlchemyRecognitionResultRepository",
]
