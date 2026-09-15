"""
CRUD operations for the problem bank database.
"""

import logging
from typing import List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import Problem
from .schemas import ProblemCreate, ProblemUpdate

logger = logging.getLogger(__name__)


def create_problem(db: Session, problem: ProblemCreate) -> Problem:
    """Create a new problem in the database."""
    db_problem = Problem(
        question=problem.question,
        solution=problem.solution,
        answer=problem.answer,
        concepts=problem.concepts,
        sub_concepts=problem.sub_concepts,
        concept_vector=problem.concept_vector,
        difficulty_score=problem.difficulty_score,
        llm_reasoning=problem.llm_reasoning,
        prerequisite_concepts=problem.prerequisite_concepts or [],
    )
    db.add(db_problem)
    db.commit()
    db.refresh(db_problem)
    return db_problem


def create_problems_bulk(
    db: Session,
    problems: List[ProblemCreate]
) -> Tuple[int, int, List[str]]:
    """
    Bulk create problems. Returns (created_count, failed_count, errors).
    """
    created = 0
    failed = 0
    errors = []

    for i, problem in enumerate(problems):
        try:
            db_problem = Problem(
                question=problem.question,
                solution=problem.solution,
                answer=problem.answer,
                concepts=problem.concepts,
                sub_concepts=problem.sub_concepts,
                concept_vector=problem.concept_vector,
                difficulty_score=problem.difficulty_score,
                llm_reasoning=problem.llm_reasoning,
                prerequisite_concepts=problem.prerequisite_concepts or [],
            )
            db.add(db_problem)
            created += 1
        except Exception as e:
            failed += 1
            errors.append(f"Row {i}: {str(e)}")
            if len(errors) > 100:
                errors.append("... (truncated)")
                break

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Bulk insert failed: {e}")
        return 0, len(problems), [f"Commit failed: {str(e)}"]

    return created, failed, errors


def get_problem(db: Session, problem_id: int) -> Optional[Problem]:
    """Get a problem by ID."""
    return db.query(Problem).filter(Problem.id == problem_id).first()


def get_problems(
    db: Session,
    skip: int = 0,
    limit: int = 50,
    min_difficulty: Optional[float] = None,
    max_difficulty: Optional[float] = None,
    concepts: Optional[List[str]] = None,
    sub_concepts: Optional[List[str]] = None,
) -> Tuple[List[Problem], int]:
    """
    Get paginated list of problems with optional filters.
    Returns (problems, total_count).
    """
    query = db.query(Problem)

    if min_difficulty is not None:
        query = query.filter(Problem.difficulty_score >= min_difficulty)
    if max_difficulty is not None:
        query = query.filter(Problem.difficulty_score <= max_difficulty)

    if concepts:
        query = query.filter(Problem.concepts.overlap(concepts))
    if sub_concepts:
        query = query.filter(Problem.sub_concepts.overlap(sub_concepts))

    total = query.count()
    problems = query.offset(skip).limit(limit).all()

    return problems, total


def get_similar_problems(
    db: Session,
    concept_vector: List[float],
    limit: int = 10,
    exclude_id: Optional[int] = None,
) -> List[Tuple[Problem, float]]:
    """
    Find problems with similar concept vectors using cosine similarity.
    Returns list of (problem, similarity_score) tuples.
    """
    distance_expr = Problem.concept_vector.cosine_distance(concept_vector)

    query = db.query(Problem, (1 - distance_expr).label("similarity"))

    if exclude_id is not None:
        query = query.filter(Problem.id != exclude_id)

    query = query.order_by(distance_expr).limit(limit)

    results = query.all()
    return [(problem, float(similarity)) for problem, similarity in results]


def get_problems_by_difficulty_range(
    db: Session,
    min_difficulty: float,
    max_difficulty: float,
    limit: int = 50,
) -> List[Problem]:
    """Get problems within a difficulty range."""
    return (
        db.query(Problem)
        .filter(Problem.difficulty_score >= min_difficulty)
        .filter(Problem.difficulty_score <= max_difficulty)
        .order_by(Problem.difficulty_score)
        .limit(limit)
        .all()
    )


def get_problems_by_concepts(
    db: Session,
    concepts: List[str],
    limit: int = 50,
) -> List[Problem]:
    """Get problems that contain any of the specified concepts."""
    return (
        db.query(Problem)
        .filter(Problem.concepts.overlap(concepts))
        .limit(limit)
        .all()
    )


def update_problem(
    db: Session,
    problem_id: int,
    problem_update: ProblemUpdate
) -> Optional[Problem]:
    """Update an existing problem."""
    db_problem = get_problem(db, problem_id)
    if not db_problem:
        return None

    update_data = problem_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_problem, field, value)

    db.commit()
    db.refresh(db_problem)
    return db_problem


def delete_problem(db: Session, problem_id: int) -> bool:
    """Delete a problem by ID. Returns True if deleted, False if not found."""
    db_problem = get_problem(db, problem_id)
    if not db_problem:
        return False

    db.delete(db_problem)
    db.commit()
    return True


def get_problem_count(db: Session) -> int:
    """Get total number of problems in database."""
    return db.query(func.count(Problem.id)).scalar() or 0


def get_difficulty_statistics(db: Session) -> dict:
    """Get statistics about problem difficulty distribution."""
    result = db.query(
        func.min(Problem.difficulty_score).label("min"),
        func.max(Problem.difficulty_score).label("max"),
        func.avg(Problem.difficulty_score).label("avg"),
        func.count(Problem.id).label("count"),
    ).first()

    if not result or result.count == 0:
        return {"min": 0, "max": 0, "avg": 0, "count": 0}

    return {
        "min": float(result.min),
        "max": float(result.max),
        "avg": float(result.avg),
        "count": result.count,
    }
