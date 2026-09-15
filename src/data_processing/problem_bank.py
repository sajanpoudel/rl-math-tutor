"""
Problem bank creation and concept vectorization for mathematical problems.
Transforms raw problem data into structured problems with difficulty scores and concept vectors.
"""

import numpy as np
import re
import spacy
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConceptVectorizer:
    """Creates vector representations of mathematical concepts"""
    
    def __init__(self, vector_dim: int = 50):
        self.vector_dim = vector_dim
        self.concept_embeddings = {}
        self._initialize_concept_mapping()
        
    def _initialize_concept_mapping(self):
        """Initialize concept to vector dimension mapping"""
        self.concept_map = {
            # Basic Arithmetic (0-7)
            'addition': (0, 1),
            'subtraction': (2, 3), 
            'multiplication': (4, 5),
            'division': (6, 7),
            
            # Fractions and Decimals (8-15)
            'fractions': (8, 10),
            'decimals': (11, 12),
            'percentages': (13, 15),
            
            # Algebraic Thinking (16-25)
            'equations': (16, 18),
            'variables': (19, 20),
            'expressions': (21, 22),
            'linear_algebra': (23, 25),
            
            # Geometry and Measurement (26-35)
            'area': (26, 27),
            'perimeter': (28, 28),
            'volume': (29, 30),
            'angles': (31, 32),
            'coordinate_geometry': (33, 35),
            
            # Word Problems and Applications (36-42)
            'word_problems': (36, 38),
            'money_problems': (39, 40),
            'time_problems': (41, 42),
            
            # Advanced Concepts (43-49)
            'ratios_proportions': (43, 45),
            'statistics_probability': (46, 47),
            'sequences_patterns': (48, 49)
        }
        
        # Concept detection patterns
        self.concept_patterns = {
            'addition': [
                r'\b(add|plus|sum|total|altogether|combined|more than)\b',
                r'\+',
                r'increase'
            ],
            'subtraction': [
                r'\b(subtract|minus|difference|less than|fewer|remove|take away)\b',
                r'-(?!\d)',  # minus sign not followed by digit
                r'decrease'
            ],
            'multiplication': [
                r'\b(multiply|times|product|each|per|rate)\b',
                r'\*|×',
                r'\b(\d+)\s+times\b'
            ],
            'division': [
                r'\b(divide|quotient|split|share|distribute|per|ratio)\b',
                r'÷|/',
                r'how many.*in each'
            ],
            'fractions': [
                r'\b(fraction|numerator|denominator|half|third|quarter)\b',
                r'\d+/\d+',
                r'parts',
                r'piece'
            ],
            'decimals': [
                r'\d+\.\d+',
                r'\b(decimal|tenths|hundredths)\b'
            ],
            'percentages': [
                r'%|percent',
                r'\b(percentage|rate)\b'
            ],
            'equations': [
                r'=',
                r'\b(equation|solve|equals|equal to)\b',
                r'x\s*=|=\s*x'
            ],
            'variables': [
                r'\b[a-z]\s*=',
                r'\b(unknown|variable|let|represents)\b'
            ],
            'area': [
                r'\b(area|square|rectangle|triangle|circle)\b',
                r'length.*width',
                r'square (feet|meters|inches|units)'
            ],
            'perimeter': [
                r'\b(perimeter|around|border|edge)\b'
            ],
            'volume': [
                r'\b(volume|cubic|capacity|container)\b',
                r'cubic (feet|meters|inches|units)'
            ],
            'word_problems': [
                r'\b(if|when|how many|how much|what is|find)\b',
                r'story|problem|situation'
            ],
            'money_problems': [
                r'\$|\b(dollar|cent|money|cost|price|buy|sell|pay)\b',
                r'expensive|cheap'
            ],
            'time_problems': [
                r'\b(hour|minute|second|day|week|month|year|time)\b',
                r'o\'clock|am|pm'
            ],
            'ratios_proportions': [
                r'\b(ratio|proportion|rate|per|for every)\b',
                r':\s*\d+|\d+\s*:'
            ],
            'statistics_probability': [
                r'\b(average|mean|median|mode|probability|chance|likely)\b',
                r'data|survey'
            ]
        }
        
    def extract_concepts(self, question: str, solution: str = "") -> List[str]:
        """Extract mathematical concepts from problem text"""
        text = f"{question} {solution}".lower()
        detected_concepts = []
        
        for concept, patterns in self.concept_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    detected_concepts.append(concept)
                    break  # Only add concept once
                    
        # Remove duplicates while preserving order
        unique_concepts = []
        for concept in detected_concepts:
            if concept not in unique_concepts:
                unique_concepts.append(concept)
                
        return unique_concepts
    
    def create_concept_vector(self, concepts: List[str]) -> np.ndarray:
        """Create vector representation of mathematical concepts"""
        vector = np.zeros(self.vector_dim)
        
        for concept in concepts:
            if concept in self.concept_map:
                start, end = self.concept_map[concept]
                # Use gaussian distribution for smoother representation
                for i in range(start, end + 1):
                    if i < self.vector_dim:
                        vector[i] = 1.0
                        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
        
    def calculate_concept_complexity(self, concepts: List[str]) -> float:
        """Calculate complexity score based on concepts involved"""
        complexity_weights = {
            'addition': 0.1,
            'subtraction': 0.1,
            'multiplication': 0.2,
            'division': 0.3,
            'fractions': 0.4,
            'decimals': 0.3,
            'percentages': 0.4,
            'equations': 0.6,
            'variables': 0.7,
            'expressions': 0.6,
            'linear_algebra': 0.8,
            'area': 0.5,
            'perimeter': 0.4,
            'volume': 0.6,
            'angles': 0.5,
            'coordinate_geometry': 0.7,
            'word_problems': 0.3,
            'money_problems': 0.2,
            'time_problems': 0.3,
            'ratios_proportions': 0.6,
            'statistics_probability': 0.7,
            'sequences_patterns': 0.8
        }
        
        if not concepts:
            return 0.1
            
        total_complexity = sum(complexity_weights.get(concept, 0.5) for concept in concepts)
        return min(total_complexity / len(concepts), 1.0)


class DifficultyAnalyzer:
    """Analyzes problem difficulty based on multiple factors"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def calculate_difficulty_score(self, problem: Dict, concepts: List[str]) -> float:
        """Calculate comprehensive difficulty score"""
        
        # Factor 1: Solution step count (25% weight)
        step_factor = self._normalize_step_count(problem.get('step_count', 1))
        
        # Factor 2: Concept complexity (30% weight)
        concept_vectorizer = ConceptVectorizer()
        concept_factor = concept_vectorizer.calculate_concept_complexity(concepts)
        
        # Factor 3: Numerical complexity (20% weight)
        numerical_factor = self._analyze_numerical_complexity(problem['question'])
        
        # Factor 4: Language complexity (15% weight)
        language_factor = self._assess_reading_level(problem['question'])
        
        # Factor 5: Prerequisite concepts (10% weight)
        prerequisite_factor = self._count_prerequisite_concepts(concepts)
        
        # Weighted combination
        difficulty = (
            0.25 * step_factor +
            0.30 * concept_factor +
            0.20 * numerical_factor +
            0.15 * language_factor +
            0.10 * prerequisite_factor
        )
        
        return max(0.1, min(1.0, difficulty))
    
    def _normalize_step_count(self, step_count: int) -> float:
        """Normalize step count to [0, 1] range"""
        # Assume 1-10 steps is the typical range
        return min(step_count / 10.0, 1.0)
    
    def _analyze_numerical_complexity(self, question: str) -> float:
        """Analyze numerical complexity of the problem"""
        complexity = 0.0
        
        # Extract all numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', question)
        
        if not numbers:
            return 0.1
            
        num_values = [float(n) for n in numbers]
        
        # Large numbers increase complexity
        max_num = max(num_values)
        if max_num > 1000:
            complexity += 0.3
        elif max_num > 100:
            complexity += 0.2
        elif max_num > 10:
            complexity += 0.1
            
        # Decimals increase complexity
        if any('.' in n for n in numbers):
            complexity += 0.2
            
        # Many numbers increase complexity
        if len(numbers) > 5:
            complexity += 0.3
        elif len(numbers) > 3:
            complexity += 0.2
            
        return min(complexity, 1.0)
    
    def _assess_reading_level(self, text: str) -> float:
        """Assess reading difficulty of problem text"""
        try:
            # Use textstat for reading level assessment
            flesch_score = textstat.flesch_reading_ease(text)
            
            # Convert Flesch score to 0-1 difficulty scale
            # Higher Flesch score = easier reading = lower difficulty
            if flesch_score >= 90:
                return 0.1  # Very easy
            elif flesch_score >= 80:
                return 0.2  # Easy
            elif flesch_score >= 70:
                return 0.3  # Fairly easy
            elif flesch_score >= 60:
                return 0.4  # Standard
            elif flesch_score >= 50:
                return 0.6  # Fairly difficult
            elif flesch_score >= 30:
                return 0.8  # Difficult
            else:
                return 1.0  # Very difficult
                
        except:
            # Fallback: simple word count based assessment
            words = len(text.split())
            if words < 20:
                return 0.2
            elif words < 40:
                return 0.4
            elif words < 60:
                return 0.6
            else:
                return 0.8
    
    def _count_prerequisite_concepts(self, concepts: List[str]) -> float:
        """Estimate difficulty based on number of prerequisite concepts"""
        concept_count = len(concepts)
        
        if concept_count <= 1:
            return 0.1
        elif concept_count <= 2:
            return 0.3
        elif concept_count <= 3:
            return 0.6
        else:
            return 1.0


class ProblemBankCreator:
    """Creates structured problem bank from raw dataset"""
    
    def __init__(self, vector_dim: int = 50):
        self.concept_vectorizer = ConceptVectorizer(vector_dim)
        self.difficulty_analyzer = DifficultyAnalyzer()
        
    def process_problems(self, raw_problems: List[Dict]) -> List[Dict]:
        """Process raw problems into structured problem bank"""
        logger.info(f"Processing {len(raw_problems)} problems...")
        
        processed_problems = []
        
        for i, problem in enumerate(raw_problems):
            try:
                processed = self._process_single_problem(problem)
                if processed:
                    processed_problems.append(processed)
                    
                if (i + 1) % 1000 == 0:
                    logger.info(f"Processed {i + 1}/{len(raw_problems)} problems")
                    
            except Exception as e:
                logger.warning(f"Error processing problem {i}: {e}")
                continue
                
        logger.info(f"Successfully processed {len(processed_problems)} problems")
        return processed_problems
    
    def _process_single_problem(self, problem: Dict) -> Optional[Dict]:
        """Process a single problem"""
        question = problem.get('question', '').strip()
        solution = problem.get('solution', '').strip()
        answer = problem.get('answer', '').strip()
        
        if not question or not answer:
            return None
            
        # Extract concepts
        concepts = self.concept_vectorizer.extract_concepts(question, solution)
        
        if not concepts:
            # If no concepts detected, assign basic arithmetic
            concepts = ['addition']
            
        # Create concept vector
        concept_vector = self.concept_vectorizer.create_concept_vector(concepts)
        
        # Calculate difficulty score
        difficulty_score = self.difficulty_analyzer.calculate_difficulty_score(problem, concepts)
        
        # Estimate time to solve
        estimated_time = self._estimate_solution_time(problem, concepts, difficulty_score)
        
        # Identify prerequisite concepts
        prerequisite_concepts = self._identify_prerequisites(concepts)
        
        return {
            'id': problem.get('id', f"problem_{hash(question) % 100000}"),
            'source_dataset': problem.get('source_dataset', 'unknown'),
            'question': question,
            'solution': solution,
            'answer': answer,
            'concepts': concepts,
            'concept_vector': concept_vector.tolist(),
            'difficulty_score': difficulty_score,
            'estimated_time_minutes': estimated_time,
            'prerequisite_concepts': prerequisite_concepts,
            'step_count': problem.get('step_count', len(problem.get('solution_steps', []))),
            'metadata': {
                'solution_steps': problem.get('solution_steps', []),
                'problem_source': problem.get('problem_source', ''),
                'raw_data': problem.get('raw_data', {})
            }
        }
    
    def _estimate_solution_time(self, problem: Dict, concepts: List[str], 
                               difficulty: float) -> int:
        """Estimate time to solve problem in minutes"""
        base_time = 2  # Base 2 minutes
        
        # Add time based on concept complexity
        concept_time = len(concepts) * 0.5
        
        # Add time based on difficulty
        difficulty_time = difficulty * 5
        
        # Add time based on step count
        step_time = problem.get('step_count', 1) * 0.5
        
        total_time = base_time + concept_time + difficulty_time + step_time
        
        return max(1, min(30, int(total_time)))  # Cap between 1-30 minutes
    
    def _identify_prerequisites(self, concepts: List[str]) -> List[str]:
        """Identify prerequisite concepts for given concepts"""
        prerequisites_map = {
            'subtraction': ['addition'],
            'multiplication': ['addition'],
            'division': ['multiplication', 'subtraction'],
            'fractions': ['division'],
            'decimals': ['fractions', 'division'],
            'percentages': ['fractions', 'decimals'],
            'equations': ['variables', 'addition', 'subtraction'],
            'variables': ['algebra_basics'],
            'area': ['multiplication'],
            'perimeter': ['addition'],
            'volume': ['area', 'multiplication'],
            'ratios_proportions': ['fractions', 'division'],
            'statistics_probability': ['division', 'fractions']
        }
        
        prerequisites = set()
        for concept in concepts:
            if concept in prerequisites_map:
                prerequisites.update(prerequisites_map[concept])
                
        return list(prerequisites)
    
    def get_problems_by_difficulty(self, problems: List[Dict], 
                                 min_difficulty: float = 0.0,
                                 max_difficulty: float = 1.0) -> List[Dict]:
        """Filter problems by difficulty range"""
        return [
            p for p in problems 
            if min_difficulty <= p['difficulty_score'] <= max_difficulty
        ]
    
    def get_problems_by_concepts(self, problems: List[Dict], 
                               required_concepts: List[str]) -> List[Dict]:
        """Filter problems that contain any of the required concepts"""
        return [
            p for p in problems 
            if any(concept in p['concepts'] for concept in required_concepts)
        ]


if __name__ == "__main__":
    # Test the problem bank creator
    from .dataset_loaders import load_combined_datasets
    
    # Load sample data
    raw_data = load_combined_datasets(openmath_max=50)
    
    # Create problem bank
    creator = ProblemBankCreator()
    processed_problems = creator.process_problems(raw_data)
    
    if processed_problems:
        sample = processed_problems[0]
        print(f"Sample processed problem:")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Concepts: {sample['concepts']}")
        print(f"Difficulty: {sample['difficulty_score']:.2f}")
        print(f"Estimated time: {sample['estimated_time_minutes']} minutes") 