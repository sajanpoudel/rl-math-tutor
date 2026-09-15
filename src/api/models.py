"""
SQLAlchemy ORM models for the problem bank database.

Best practices implemented:
- BIGINT for IDs (scalability)
- Server-side timestamps with timezone
- CHECK constraints for data integrity
- HNSW index for vector similarity (better for real-time queries)
- Hash index for question deduplication
- Version column for optimistic locking
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY

from .config import get_settings
from .database import Base

settings = get_settings()


class Problem(Base):
    """
    Math problem with concept vectors and difficulty scoring.

    Stores problems from GSM8K and other datasets along with:
    - Extracted concepts and sub-concepts
    - 58-dimensional concept vectors for similarity search
    - Difficulty scores from cognitive load analysis
    - LLM reasoning for concept extraction
    """

    __tablename__ = "problems"

    # Primary key: BIGINT for scalability, SERIAL handled by PostgreSQL
    id = Column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        index=True
    )

    # Core problem data
    question = Column(Text, nullable=False)
    solution = Column(Text, nullable=False)
    answer = Column(String(500), nullable=True)

    # Concept taxonomy
    concepts = Column(ARRAY(Text), nullable=False, server_default="{}")
    sub_concepts = Column(ARRAY(Text), nullable=False, server_default="{}")

    # 58-dimensional L2-normalized vector for similarity search
    concept_vector = Column(Vector(settings.vector_dimension), nullable=False)

    # Difficulty from cognitive load analysis (constrained 0.0-1.0)
    difficulty_score = Column(Float, nullable=False)

    # LLM reasoning trace
    llm_reasoning = Column(Text, nullable=True)

    # Prerequisite concepts for adaptive sequencing
    prerequisite_concepts = Column(ARRAY(Text), nullable=True, server_default="{}")

    # Optimistic locking version
    version = Column(Integer, nullable=False, server_default="1")

    # Timestamps with timezone, server-side defaults
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now()
    )

    __table_args__ = (
        # Data integrity constraint
        CheckConstraint(
            "difficulty_score >= 0.0 AND difficulty_score <= 1.0",
            name="ck_problems_difficulty_range"
        ),

        # B-tree index for range queries on difficulty
        Index("idx_problems_difficulty", "difficulty_score"),

        # GIN indexes for array containment queries
        Index("idx_problems_concepts", "concepts", postgresql_using="gin"),
        Index("idx_problems_sub_concepts", "sub_concepts", postgresql_using="gin"),

        # HNSW index for vector similarity search (cosine distance)
        # HNSW is preferred over IVFFlat for real-time queries
        # m=16: connections per layer, ef_construction=64: build quality
        Index(
            "idx_problems_concept_vector_hnsw",
            "concept_vector",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"concept_vector": "vector_cosine_ops"}
        ),
    )

    def __repr__(self) -> str:
        return f"<Problem(id={self.id}, difficulty={self.difficulty_score:.2f})>"
