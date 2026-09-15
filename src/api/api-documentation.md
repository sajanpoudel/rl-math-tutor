# Problem Bank API Documentation

## Overview

REST API for the RL-Math problem bank. Provides storage and retrieval of mathematical problems with cognitive load-based concept vectors for adaptive tutoring systems.

**Base URL:** `http://localhost:8000`
**API Prefix:** `/api/v1`

---

## Quick Start

### Prerequisites

1. PostgreSQL 14+ with pgvector extension
2. Python 3.10+

### Installation

```bash
pip install -r requirements.txt
```

### Database Setup (First Time Only)

```bash
# Create database
createdb rl_math

# Enable pgvector extension (done automatically on first run)
psql -d rl_math -c "CREATE EXTENSION IF NOT EXISTS vector"
```

### Configuration

The default configuration connects to local PostgreSQL as the current user:

```
postgresql://student@localhost:5432/rl_math
```

To override, set environment variables or create a `.env` file:

```bash
RLMATH_DATABASE_URL=postgresql://user:password@localhost:5432/rl_math
```

### Start Server

```bash
cd /path/to/RL-Math
source venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

For development with auto-reload:

```bash
uvicorn src.api.main:app --reload --port 8000
```

### Import Data (First Time Only)

```bash
python -m src.api.import_csv dataset/gsm8k/train_with_concepts_and_subconcepts_stage5.csv
```

By default, `src.api.import_csv` imports `concept_vector_norm`, which is the 58-dimensional
L2-normalized vector used by the RL environment for state representation and cosine matching.

---

## After Computer Restart

PostgreSQL auto-starts on macOS. If needed:

```bash
# Check PostgreSQL status
brew services list | grep postgresql

# Start PostgreSQL (if not running)
brew services start postgresql@14

# Start API server
cd /path/to/RL-Math
source venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

| Task | Command |
|------|---------|
| Check PostgreSQL | `brew services list \| grep postgresql` |
| Start PostgreSQL | `brew services start postgresql@14` |
| Stop PostgreSQL | `brew services stop postgresql@14` |
| Start API | `uvicorn src.api.main:app --port 8000` |

Once running:
- API: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

---

## Data Model

### Problem Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | integer | auto | Primary key |
| question | string | yes | Problem text |
| solution | string | yes | Step-by-step solution (includes #### answer) |
| answer | string | no | Final numerical answer |
| concepts | string[] | yes | High-level concepts |
| sub_concepts | string[] | yes | Fine-grained sub-concepts |
| concept_vector | float[58] | yes | L2-normalized concept vector |
| difficulty_score | float | yes | 0.0 (easy) to 1.0 (hard) |
| llm_reasoning | string | no | LLM explanation for concept extraction |
| prerequisite_concepts | string[] | no | Same as sub_concepts (for RL state representation) |
| version | integer | auto | Optimistic locking version (starts at 1) |
| created_at | datetime | auto | Creation timestamp |
| updated_at | datetime | auto | Last update timestamp |

### Vector Dimensions (58 total)

| Range | Domain |
|-------|--------|
| 0-9 | Arithmetic Operations |
| 10-17 | Fractions and Decimals |
| 18-23 | Percentages and Ratios |
| 24-29 | Algebra |
| 30-39 | Geometry and Measurement |
| 40-43 | Number Theory |
| 44-48 | Statistics and Probability |
| 49-50 | Time and Money |
| 51-57 | Problem-Solving Skills |

---

## Endpoints

### Health Check

Check API and database status.

```
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "database": true,
  "version": "0.1.0"
}
```

---

### Create Problem

```
POST /api/v1/problems
```

**Request Body:**

```json
{
  "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
  "solution": "In the beginning, Betty has only 100 / 2 = $50.\nBetty's grandparents gave her 15 * 2 = $30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $5 more.\n#### 5",
  "answer": "5",
  "concepts": ["addition", "subtraction", "multiplication", "fractions", "money_calculations"],
  "sub_concepts": ["frac_part_whole_identification", "mul_fact_retrieval", "sub_multi_digit_with_borrow", "intermediate_result_management", "wp_schema_identification", "wp_variable_definition"],
  "concept_vector": [0.0, 0.0, 0.515, 0.0, 0.129, 0.0, 0.0, 0.0, 0.0, 0.0, 0.193, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.515, 0.0, 0.644, 0.0, 0.0, 0.0, 0.0],
  "difficulty_score": 0.7284,
  "llm_reasoning": "The solution interprets 'half of the money' as 100/2, using fraction part-whole understanding..."
}
```

**Response (201 Created):**

```json
{
  "id": 1,
  "question": "Betty is saving money...",
  "solution": "In the beginning...",
  "answer": "5",
  "concepts": ["addition", "subtraction", "multiplication", "fractions", "money_calculations"],
  "sub_concepts": ["frac_part_whole_identification", "mul_fact_retrieval", "sub_multi_digit_with_borrow", "intermediate_result_management", "wp_schema_identification", "wp_variable_definition"],
  "concept_vector": [0.0, 0.0, 0.515, ...],
  "difficulty_score": 0.7284,
  "llm_reasoning": "The solution interprets...",
  "prerequisite_concepts": [],
  "version": 1,
  "created_at": "2026-01-07T10:30:00Z",
  "updated_at": "2026-01-07T10:30:00Z"
}
```

---

### Bulk Create Problems

```
POST /api/v1/problems/bulk
```

Maximum 1000 problems per request.

**Request Body:**

```json
[
  {
    "question": "Problem 1...",
    "solution": "Solution 1...",
    "concept_vector": [0.0, 0.0, ...],
    "difficulty_score": 0.65,
    "concepts": ["addition"],
    "sub_concepts": ["add_single_digit"]
  },
  {
    "question": "Problem 2...",
    "solution": "Solution 2...",
    "concept_vector": [0.0, 0.0, ...],
    "difficulty_score": 0.72,
    "concepts": ["multiplication"],
    "sub_concepts": ["mul_fact_retrieval"]
  }
]
```

**Response (201 Created):**

```json
{
  "created": 2,
  "failed": 0,
  "errors": []
}
```

---

### List Problems

```
GET /api/v1/problems
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | integer | 1 | Page number (1-indexed) |
| page_size | integer | 50 | Items per page (max 500) |
| min_difficulty | float | - | Minimum difficulty (0.0-1.0) |
| max_difficulty | float | - | Maximum difficulty (0.0-1.0) |
| concepts | string | - | Comma-separated concepts (OR logic) |
| sub_concepts | string | - | Comma-separated sub-concepts (OR logic) |

**Note:** All filter parameters are optional. To get all problems, only specify `page` and `page_size`. In Swagger UI, leave filter fields empty (do not enter placeholder text like "concepts").

**Example Requests:**

```bash
# Get all problems (page 1, 10 items)
GET /api/v1/problems?page=1&page_size=10

# Filter by concepts
GET /api/v1/problems?concepts=addition,fractions&page_size=20

# Filter by difficulty range
GET /api/v1/problems?min_difficulty=0.5&max_difficulty=0.8
```

**Example Request (with all filters):**

```
GET /api/v1/problems?page=1&page_size=20&min_difficulty=0.5&max_difficulty=0.8&concepts=fractions,percentages
```

**Response:**

```json
{
  "items": [
    {
      "id": 1,
      "question": "...",
      "solution": "...",
      "concepts": ["fractions"],
      "sub_concepts": ["frac_part_whole_identification"],
      "concept_vector": [0.0, ...],
      "difficulty_score": 0.65,
      "version": 1,
      "created_at": "2026-01-07T10:30:00Z",
      "updated_at": "2026-01-07T10:30:00Z"
    }
  ],
  "total": 1547,
  "page": 1,
  "page_size": 20,
  "pages": 78
}
```

---

### Get Problem by ID

```
GET /api/v1/problems/{problem_id}
```

**Example:**

```
GET /api/v1/problems/42
```

**Response (200 OK):**

```json
{
  "id": 42,
  "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
  "solution": "He writes each friend 3*2=6 pages a week\nSo he writes 6*2=12 pages every week\nThat means he writes 12*52=624 pages a year\n#### 624",
  "answer": "624",
  "concepts": ["multiplication", "time_calculations", "word_problem_comprehension", "multi_step_problem_solving"],
  "sub_concepts": ["intermediate_result_management", "mul_fact_retrieval", "mul_multi_digit_algorithm", "step_decomposition", "time_unit_conversion", "wp_schema_identification"],
  "concept_vector": [0.0, 0.0, 0.0, 0.0, 0.503, ...],
  "difficulty_score": 0.7162,
  "llm_reasoning": "The solution multiplies multi-digit numbers...",
  "prerequisite_concepts": ["addition"],
  "version": 1,
  "created_at": "2026-01-07T10:30:00Z",
  "updated_at": "2026-01-07T10:30:00Z"
}
```

**Response (404 Not Found):**

```json
{
  "detail": "Problem not found"
}
```

---

### Update Problem

```
PUT /api/v1/problems/{problem_id}
```

Only provided fields are updated.

**Request Body:**

```json
{
  "difficulty_score": 0.75,
  "llm_reasoning": "Updated reasoning text"
}
```

**Response (200 OK):**

```json
{
  "id": 42,
  "question": "James writes a 3-page letter...",
  "difficulty_score": 0.75,
  "llm_reasoning": "Updated reasoning text",
  ...
}
```

---

### Delete Problem

```
DELETE /api/v1/problems/{problem_id}
```

**Response:** 204 No Content (on success)

---

### Find Similar Problems

Find problems with similar concept vectors using cosine similarity.

```
GET /api/v1/problems/similar
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| vector | string | yes | Comma-separated 58 floats |
| limit | integer | no | Max results (default 10, max 100) |
| exclude_id | integer | no | Problem ID to exclude |

**How to Test:**

1. Get a concept_vector from an existing problem:
   ```bash
   curl -s "http://localhost:8000/api/v1/problems/1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(','.join(str(x) for x in d['concept_vector']))"
   ```

2. Use that vector in the similarity search (excluding the source problem):
   ```
   GET /api/v1/problems/similar?vector=<paste_vector_here>&limit=5&exclude_id=1
   ```

**Sample Test Vector (from Problem #1):**

```
0.5252256989479065,0.0,0.0,0.0,0.0,0.0,0.13130642473697662,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5252256989479065,0.0,0.6565321087837219,0.0,0.0,0.0,0.0
```

**Example Request:**

```
GET /api/v1/problems/similar?vector=0.5252256989479065,0.0,0.0,0.0,0.0,0.0,0.13130642473697662,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5252256989479065,0.0,0.6565321087837219,0.0,0.0,0.0,0.0&limit=5&exclude_id=1
```

**Response:**

```json
{
  "query_vector": [0.5, 0.0, 0.0, ...],
  "results": [
    {
      "problem": {
        "id": 15,
        "question": "...",
        "concept_vector": [0.48, 0.0, 0.0, ...],
        "difficulty_score": 0.68
      },
      "similarity": 0.9542
    },
    {
      "problem": {
        "id": 89,
        "question": "...",
        "concept_vector": [0.52, 0.0, 0.0, ...],
        "difficulty_score": 0.71
      },
      "similarity": 0.9231
    }
  ]
}
```

---

### Get Problems by Difficulty Range

```
GET /api/v1/problems/by-difficulty
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| min_difficulty | float | yes | Minimum difficulty (0.0-1.0) |
| max_difficulty | float | yes | Maximum difficulty (0.0-1.0) |
| limit | integer | no | Max results (default 50, max 500) |

**Example:**

```
GET /api/v1/problems/by-difficulty?min_difficulty=0.3&max_difficulty=0.5&limit=10
```

**Response:**

```json
[
  {
    "id": 23,
    "question": "...",
    "difficulty_score": 0.32,
    ...
  },
  {
    "id": 45,
    "question": "...",
    "difficulty_score": 0.41,
    ...
  }
]
```

---

### Get Problems by Concepts

```
GET /api/v1/problems/by-concepts
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| concepts | string | yes | Comma-separated concept names |
| limit | integer | no | Max results (default 50, max 500) |

**Example:**

```
GET /api/v1/problems/by-concepts?concepts=fractions,decimals&limit=20
```

**Response:**

```json
[
  {
    "id": 78,
    "question": "...",
    "concepts": ["fractions", "addition"],
    ...
  }
]
```

---

### Get Statistics

```
GET /api/v1/problems/stats
```

**Response:**

```json
{
  "total_problems": 7473,
  "difficulty": {
    "min": 0.4108,
    "max": 0.8905,
    "avg": 0.7163,
    "count": 7473
  }
}
```

---

## Error Responses

### 400 Bad Request

Invalid request parameters.

```json
{
  "detail": "Vector must have 58 dimensions"
}
```

### 404 Not Found

Resource does not exist.

```json
{
  "detail": "Problem not found"
}
```

### 422 Validation Error

Request body validation failed.

```json
{
  "detail": [
    {
      "loc": ["body", "concept_vector"],
      "msg": "concept_vector must have 58 dimensions, got 50",
      "type": "value_error"
    }
  ]
}
```

---

## Valid Concepts

High-level concepts recognized by the system:

- addition
- subtraction
- multiplication
- division
- order_of_operations
- fractions
- decimals
- fraction_to_decimal_conversion
- percentages
- ratios
- proportions
- rates
- variables
- algebraic_expressions
- solving_equations
- area
- perimeter
- volume
- angles
- shapes
- measurement
- unit_conversion
- factors
- multiples
- primes
- divisibility
- mean_average
- median
- mode
- range
- probability
- counting
- time_calculations
- elapsed_time
- money_calculations
- currency
- word_problem_comprehension
- multi_step_problem_solving
- logical_reasoning
- pattern_recognition
- estimation
- data_interpretation

---

## Valid Sub-Concepts

Fine-grained sub-concepts with their vector dimension indices:

### Arithmetic (Dims 0-9)

| Sub-Concept | Dimension | Intensity |
|-------------|-----------|-----------|
| add_single_digit | 0 | 0.2 |
| add_multi_digit_no_carry | 0 | 0.4 |
| add_multi_digit_with_carry | 0 | 0.8 |
| add_decimal_alignment | 0 | 0.9 |
| sub_single_digit | 2 | 0.2 |
| sub_multi_digit_no_borrow | 2 | 0.4 |
| sub_multi_digit_with_borrow | 2 | 0.8 |
| sub_across_zeros | 2 | 1.0 |
| mul_fact_retrieval | 4 | 0.2 |
| mul_multi_digit_algorithm | 4 | 0.8 |
| mul_powers_of_10 | 4 | 0.3 |
| mul_distributive_property | 5 | 0.7 |
| div_fact_retrieval | 6 | 0.2 |
| div_long_division_algorithm | 6 | 1.0 |
| div_with_remainder | 6 | 0.7 |
| div_interpret_remainder | 7 | 0.9 |
| oop_parentheses_first | 8 | 0.5 |
| oop_mul_div_left_to_right | 8 | 0.8 |
| oop_add_sub_left_to_right | 8 | 0.6 |
| oop_nested_grouping | 9 | 1.0 |

### Fractions and Decimals (Dims 10-17)

| Sub-Concept | Dimension | Intensity |
|-------------|-----------|-----------|
| frac_part_whole_identification | 10 | 0.3 |
| frac_equivalent_generation | 10 | 0.6 |
| frac_simplify | 10 | 0.5 |
| frac_improper_mixed_conversion | 11 | 0.6 |
| frac_add_sub_like_denom | 12 | 0.5 |
| frac_add_sub_unlike_denom | 12 | 1.0 |
| dec_place_value_identification | 14 | 0.4 |
| dec_rounding | 14 | 0.6 |
| dec_comparing_ordering | 14 | 0.7 |
| conv_power_of_10_denom | 16 | 0.4 |
| conv_long_division_method | 16 | 0.9 |
| conv_memorized_equivalents | 16 | 0.2 |

### Problem-Solving Skills (Dims 51-57)

| Sub-Concept | Dimension | Intensity |
|-------------|-----------|-----------|
| wp_schema_identification | 51 | 0.8 |
| wp_variable_definition | 51 | 0.6 |
| step_decomposition | 53 | 0.9 |
| intermediate_result_management | 53 | 1.0 |
| logic_deductive_chaining | 54 | 0.8 |
| logic_constraint_satisfaction | 54 | 1.0 |
| pat_arithmetic_sequence | 55 | 0.5 |
| pat_geometric_sequence | 55 | 0.8 |
| pat_visual_growth | 55 | 0.6 |
| est_rounding_strategy | 56 | 0.5 |
| est_magnitude_check | 56 | 0.6 |
| data_graph_reading | 57 | 0.6 |
| data_key_extraction | 57 | 0.7 |

For the complete list of 90+ sub-concepts, see [create-dataset-gpt5.py](../../dataset-pipeline/create-dataset-gpt5.py) lines 46-276.

---

## CLI Import Tool

Import problems from Stage 5 CSV:

```bash
python -m src.api.import_csv <csv_path> [options]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| --batch-size | 500 | Records per commit batch |
| --use-raw-vectors | false | Use raw vectors instead of normalized |

**Example:**

```bash
python -m src.api.import_csv dataset/gsm8k/train_with_concepts_and_subconcepts_stage5.csv --batch-size 1000
```
