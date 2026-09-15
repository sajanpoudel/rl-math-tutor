"""
CSV import utility for loading Stage 5 data into the database.

Usage:
    python -m src.api.import_csv dataset/gsm8k/train_with_concepts_and_subconcepts_stage5.csv
"""

import argparse
import ast
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from .database import get_db_context, init_db
from .models import Problem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_json_list(value: str) -> List[str]:
    """Parse a JSON-like list string into a Python list."""
    if pd.isna(value) or not value:
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def parse_float_list(value: str) -> List[float]:
    """Parse a JSON-like float list string into a Python list."""
    if pd.isna(value) or not value:
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def extract_final_answer(solution_text: str) -> Optional[str]:
    """Extract the final answer from solution text (after ####)."""
    if not solution_text:
        return None
    match = re.search(r"####\s*(.+?)(?:\n|$)", solution_text)
    if match:
        return match.group(1).strip()
    return None


# Note: prerequisite_concepts is populated with the same values as sub_concepts
# for consistency with the source data model.


def import_csv(
    csv_path: str,
    batch_size: int = 500,
    use_normalized_vectors: bool = True,
) -> tuple[int, int]:
    """
    Import problems from Stage 5 CSV into the database.

    Args:
        csv_path: Path to the CSV file
        batch_size: Number of records to commit per batch
        use_normalized_vectors: If True, use concept_vector_norm; else use raw

    Returns:
        Tuple of (imported_count, failed_count)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info(f"Reading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    logger.info(f"Found {total_rows} rows")

    vector_column = "concept_vector_norm" if use_normalized_vectors else "concept_vector_raw"
    if vector_column not in df.columns:
        raise ValueError(f"Column {vector_column} not found in CSV")

    init_db()

    imported = 0
    failed = 0
    errors = []

    with get_db_context() as db:
        batch = []

        for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Importing"):
            try:
                question = row.get("question", "")
                solution = row.get("answer", "")

                if pd.isna(question) or not question:
                    failed += 1
                    continue

                final_answer = extract_final_answer(solution) if solution else None
                concepts = parse_json_list(row.get("concepts", "[]"))
                sub_concepts = parse_json_list(row.get("sub_concepts", "[]"))
                concept_vector = parse_float_list(row.get(vector_column, "[]"))

                if not concept_vector or len(concept_vector) != 58:
                    failed += 1
                    errors.append(f"Row {idx}: Invalid vector dimension")
                    continue

                difficulty = row.get("difficulty_score", 0.5)
                if pd.isna(difficulty):
                    difficulty = 0.5

                llm_reasoning = row.get("llm_reasoning")
                if pd.isna(llm_reasoning):
                    llm_reasoning = None

                problem = Problem(
                    question=str(question),
                    solution=str(solution) if solution else "",
                    answer=final_answer,
                    concepts=concepts,
                    sub_concepts=sub_concepts,
                    concept_vector=concept_vector,
                    difficulty_score=float(difficulty),
                    llm_reasoning=llm_reasoning,
                    prerequisite_concepts=sub_concepts,
                )
                batch.append(problem)
                imported += 1

                if len(batch) >= batch_size:
                    db.add_all(batch)
                    db.commit()
                    batch = []

            except Exception as e:
                failed += 1
                errors.append(f"Row {idx}: {str(e)}")
                if len(errors) <= 10:
                    logger.warning(f"Error at row {idx}: {e}")

        if batch:
            db.add_all(batch)
            db.commit()

    logger.info(f"Import complete: {imported} imported, {failed} failed")

    if errors:
        logger.warning(f"First few errors: {errors[:5]}")

    return imported, failed


def main():
    parser = argparse.ArgumentParser(
        description="Import Stage 5 CSV data into the problem bank database"
    )
    parser.add_argument(
        "csv_path",
        help="Path to the Stage 5 CSV file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for database commits (default: 500)"
    )
    parser.add_argument(
        "--use-raw-vectors",
        action="store_true",
        help="Use raw vectors instead of normalized vectors"
    )

    args = parser.parse_args()

    try:
        imported, failed = import_csv(
            args.csv_path,
            batch_size=args.batch_size,
            use_normalized_vectors=not args.use_raw_vectors,
        )
        print(f"Successfully imported {imported} problems ({failed} failed)")
        sys.exit(0 if failed == 0 else 1)
    except Exception as e:
        logger.error(f"Import failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
