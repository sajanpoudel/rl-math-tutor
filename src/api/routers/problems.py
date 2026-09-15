"""
API endpoints for problem management.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db
from .. import crud
from ..schemas import (
    BulkCreateResponse,
    ProblemCreate,
    ProblemListResponse,
    ProblemResponse,
    ProblemUpdate,
    SimilarProblemResponse,
    SimilarProblemsResponse,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/problems", tags=["problems"])


@router.post("", response_model=ProblemResponse, status_code=201)
def create_problem(
    problem: ProblemCreate,
    db: Session = Depends(get_db)
) -> ProblemResponse:
    """Create a new problem."""
    db_problem = crud.create_problem(db, problem)
    return ProblemResponse.model_validate(db_problem)


@router.post("/bulk", response_model=BulkCreateResponse, status_code=201)
def create_problems_bulk(
    problems: List[ProblemCreate],
    db: Session = Depends(get_db)
) -> BulkCreateResponse:
    """Bulk create multiple problems."""
    if len(problems) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 1000 problems per bulk request"
        )

    created, failed, errors = crud.create_problems_bulk(db, problems)
    return BulkCreateResponse(created=created, failed=failed, errors=errors)


@router.get("", response_model=ProblemListResponse)
def list_problems(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=500, description="Items per page"),
    min_difficulty: Optional[float] = Query(None, ge=0.0, le=1.0),
    max_difficulty: Optional[float] = Query(None, ge=0.0, le=1.0),
    concepts: Optional[str] = Query(
        None,
        description="Comma-separated list of concepts to filter by"
    ),
    sub_concepts: Optional[str] = Query(
        None,
        description="Comma-separated list of sub-concepts to filter by"
    ),
    db: Session = Depends(get_db)
) -> ProblemListResponse:
    """Get paginated list of problems with optional filters."""
    skip = (page - 1) * page_size

    concept_list = concepts.split(",") if concepts else None
    sub_concept_list = sub_concepts.split(",") if sub_concepts else None

    problems, total = crud.get_problems(
        db,
        skip=skip,
        limit=page_size,
        min_difficulty=min_difficulty,
        max_difficulty=max_difficulty,
        concepts=concept_list,
        sub_concepts=sub_concept_list,
    )

    pages = (total + page_size - 1) // page_size

    return ProblemListResponse(
        items=[ProblemResponse.model_validate(p) for p in problems],
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


@router.get("/similar", response_model=SimilarProblemsResponse)
def find_similar_problems(
    vector: str = Query(
        ...,
        description="Comma-separated 58-dimensional concept vector"
    ),
    limit: int = Query(10, ge=1, le=100, description="Number of results"),
    exclude_id: Optional[int] = Query(None, description="Problem ID to exclude"),
    db: Session = Depends(get_db)
) -> SimilarProblemsResponse:
    """Find problems with similar concept vectors using cosine similarity."""
    try:
        concept_vector = [float(x.strip()) for x in vector.split(",")]
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid vector format. Expected comma-separated floats."
        )

    if len(concept_vector) != settings.vector_dimension:
        raise HTTPException(
            status_code=400,
            detail=f"Vector must have {settings.vector_dimension} dimensions"
        )

    results = crud.get_similar_problems(
        db,
        concept_vector=concept_vector,
        limit=limit,
        exclude_id=exclude_id,
    )

    return SimilarProblemsResponse(
        query_vector=concept_vector,
        results=[
            SimilarProblemResponse(
                problem=ProblemResponse.model_validate(problem),
                similarity=similarity
            )
            for problem, similarity in results
        ],
    )


@router.get("/by-difficulty", response_model=List[ProblemResponse])
def get_problems_by_difficulty(
    min_difficulty: float = Query(..., ge=0.0, le=1.0),
    max_difficulty: float = Query(..., ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
) -> List[ProblemResponse]:
    """Get problems within a difficulty range."""
    if min_difficulty > max_difficulty:
        raise HTTPException(
            status_code=400,
            detail="min_difficulty must be less than or equal to max_difficulty"
        )

    problems = crud.get_problems_by_difficulty_range(
        db, min_difficulty, max_difficulty, limit
    )
    return [ProblemResponse.model_validate(p) for p in problems]


@router.get("/by-concepts", response_model=List[ProblemResponse])
def get_problems_by_concepts(
    concepts: str = Query(
        ...,
        description="Comma-separated list of concepts"
    ),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
) -> List[ProblemResponse]:
    """Get problems that contain any of the specified concepts."""
    concept_list = [c.strip() for c in concepts.split(",") if c.strip()]
    if not concept_list:
        raise HTTPException(status_code=400, detail="At least one concept required")

    problems = crud.get_problems_by_concepts(db, concept_list, limit)
    return [ProblemResponse.model_validate(p) for p in problems]


@router.get("/stats")
def get_statistics(db: Session = Depends(get_db)) -> dict:
    """Get statistics about the problem bank."""
    count = crud.get_problem_count(db)
    difficulty_stats = crud.get_difficulty_statistics(db)

    return {
        "total_problems": count,
        "difficulty": difficulty_stats,
    }


@router.get("/{problem_id}", response_model=ProblemResponse)
def get_problem(
    problem_id: int,
    db: Session = Depends(get_db)
) -> ProblemResponse:
    """Get a problem by ID."""
    db_problem = crud.get_problem(db, problem_id)
    if not db_problem:
        raise HTTPException(status_code=404, detail="Problem not found")
    return ProblemResponse.model_validate(db_problem)


@router.put("/{problem_id}", response_model=ProblemResponse)
def update_problem(
    problem_id: int,
    problem: ProblemUpdate,
    db: Session = Depends(get_db)
) -> ProblemResponse:
    """Update an existing problem."""
    db_problem = crud.update_problem(db, problem_id, problem)
    if not db_problem:
        raise HTTPException(status_code=404, detail="Problem not found")
    return ProblemResponse.model_validate(db_problem)


@router.delete("/{problem_id}", status_code=204)
def delete_problem(
    problem_id: int,
    db: Session = Depends(get_db)
) -> None:
    """Delete a problem by ID."""
    if not crud.delete_problem(db, problem_id):
        raise HTTPException(status_code=404, detail="Problem not found")
