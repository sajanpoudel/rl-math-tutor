"""
Pydantic schemas for request/response validation.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from .config import get_settings

settings = get_settings()


class ProblemBase(BaseModel):
    """Base schema with common problem fields."""

    question: str = Field(..., min_length=1, description="The math problem text")
    solution: str = Field(..., min_length=1, description="Step-by-step solution")
    answer: Optional[str] = Field(None, max_length=500, description="Final answer")
    concepts: List[str] = Field(
        default_factory=list,
        description="High-level mathematical concepts"
    )
    sub_concepts: List[str] = Field(
        default_factory=list,
        description="Fine-grained sub-concepts from cognitive load analysis"
    )
    concept_vector: List[float] = Field(
        ...,
        description="58-dimensional normalized concept vector"
    )
    difficulty_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Difficulty score from 0.0 (easy) to 1.0 (hard)"
    )
    llm_reasoning: Optional[str] = Field(
        None,
        description="LLM explanation for concept extraction"
    )
    prerequisite_concepts: Optional[List[str]] = Field(
        default_factory=list,
        description="Prerequisite concepts required"
    )

    @field_validator("concept_vector")
    @classmethod
    def validate_vector_dimension(cls, v: List[float]) -> List[float]:
        if len(v) != settings.vector_dimension:
            raise ValueError(
                f"concept_vector must have {settings.vector_dimension} dimensions, "
                f"got {len(v)}"
            )
        return v


class ProblemCreate(ProblemBase):
    """Schema for creating a new problem."""
    pass


class ProblemUpdate(BaseModel):
    """Schema for updating an existing problem. All fields optional."""

    question: Optional[str] = Field(None, min_length=1)
    solution: Optional[str] = Field(None, min_length=1)
    answer: Optional[str] = Field(None, max_length=500)
    concepts: Optional[List[str]] = None
    sub_concepts: Optional[List[str]] = None
    concept_vector: Optional[List[float]] = None
    difficulty_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    llm_reasoning: Optional[str] = None
    prerequisite_concepts: Optional[List[str]] = None

    @field_validator("concept_vector")
    @classmethod
    def validate_vector_dimension(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None and len(v) != settings.vector_dimension:
            raise ValueError(
                f"concept_vector must have {settings.vector_dimension} dimensions, "
                f"got {len(v)}"
            )
        return v


class ProblemResponse(ProblemBase):
    """Schema for problem responses."""

    id: int
    version: int = Field(default=1, description="Optimistic locking version")
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProblemListResponse(BaseModel):
    """Paginated list of problems."""

    items: List[ProblemResponse]
    total: int
    page: int
    page_size: int
    pages: int


class SimilarProblemResponse(BaseModel):
    """Problem with similarity score."""

    problem: ProblemResponse
    similarity: float = Field(..., description="Cosine similarity score (0-1)")


class SimilarProblemsResponse(BaseModel):
    """List of similar problems."""

    query_vector: List[float]
    results: List[SimilarProblemResponse]


class BulkCreateResponse(BaseModel):
    """Response for bulk create operations."""

    created: int
    failed: int
    errors: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    database: bool
    version: str
