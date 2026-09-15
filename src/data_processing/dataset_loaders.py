"""
Dataset loaders for GSM8K and OpenMathInstruct-2 datasets.
Handles downloading, parsing, and preprocessing of mathematical problem datasets.
"""

import json
import re
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDatasetLoader:
    """Base class for dataset loaders"""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load(self) -> List[Dict]:
        """Load and return dataset as list of dictionaries"""
        raise NotImplementedError
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize mathematical notation
        text = text.replace('<<', '(').replace('>>', ')')
        
        return text


class GSM8KLoader(BaseDatasetLoader):
    """Loader for GSM8K dataset - Grade School Math 8K problems"""
    
    def __init__(self, cache_dir: str = "./data/cache", split: str = "train"):
        super().__init__(cache_dir)
        self.split = split
        
    def load(self) -> List[Dict]:
        """Load GSM8K dataset from Hugging Face"""
        logger.info(f"Loading GSM8K dataset ({self.split} split)...")
        
        try:
            # Load from Hugging Face datasets
            dataset = load_dataset("gsm8k", "main", cache_dir=str(self.cache_dir))
            data = dataset[self.split]
            
            processed_problems = []
            
            for idx, item in enumerate(data):
                try:
                    problem = self._process_gsm8k_item(item, idx)
                    if problem:
                        processed_problems.append(problem)
                except Exception as e:
                    logger.warning(f"Error processing GSM8K item {idx}: {e}")
                    continue
                    
            logger.info(f"Successfully loaded {len(processed_problems)} GSM8K problems")
            return processed_problems
            
        except Exception as e:
            logger.error(f"Error loading GSM8K dataset: {e}")
            return []
    
    def _process_gsm8k_item(self, item: Dict, idx: int) -> Optional[Dict]:
        """Process individual GSM8K item"""
        question = self._clean_text(item.get('question', ''))
        answer = self._clean_text(item.get('answer', ''))
        
        if not question or not answer:
            return None
            
        # Extract numerical answer from solution
        numerical_answer = self._extract_numerical_answer(answer)
        
        # Extract solution steps
        solution_steps = self._extract_solution_steps(answer)
        
        return {
            'id': f"gsm8k_{idx}",
            'source_dataset': 'gsm8k',
            'question': question,
            'solution': answer,
            'answer': numerical_answer,
            'solution_steps': solution_steps,
            'step_count': len(solution_steps),
            'raw_data': item
        }
    
    def _extract_numerical_answer(self, answer_text: str) -> str:
        """Extract final numerical answer from solution text"""
        # Look for patterns like "#### 42" or "The answer is 42"
        patterns = [
            r'####\s*([0-9,.]+)',
            r'answer is\s*([0-9,.]+)',
            r'total.*?is\s*([0-9,.]+)',
            r'equals?\s*([0-9,.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, answer_text, re.IGNORECASE)
            if match:
                return match.group(1).replace(',', '')
                
        # Fallback: extract last number in the text
        numbers = re.findall(r'([0-9,.]+)', answer_text)
        if numbers:
            return numbers[-1].replace(',', '')
            
        return ""
    
    def _extract_solution_steps(self, answer_text: str) -> List[str]:
        """Extract individual solution steps"""
        # Split by sentence endings and clean
        sentences = re.split(r'[.!?]+', answer_text)
        
        steps = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter out very short fragments
                # Skip the final answer line
                if not re.search(r'####|answer is', sentence, re.IGNORECASE):
                    steps.append(sentence)
                    
        return steps


class OpenMathInstructLoader(BaseDatasetLoader):
    """Loader for OpenMathInstruct-2 dataset"""
    
    def __init__(self, cache_dir: str = "./data/cache", max_samples: Optional[int] = None):
        super().__init__(cache_dir)
        self.max_samples = max_samples
        
    def load(self) -> List[Dict]:
        """Load OpenMathInstruct-2 dataset"""
        logger.info("Loading OpenMathInstruct-2 dataset...")
        
        try:
            # Load from Hugging Face datasets
            dataset = load_dataset("nvidia/OpenMathInstruct-2", cache_dir=str(self.cache_dir))
            data = dataset["train"]
            
            if self.max_samples:
                data = data.select(range(min(self.max_samples, len(data))))
                
            processed_problems = []
            
            for idx, item in enumerate(data):
                try:
                    problem = self._process_openmath_item(item, idx)
                    if problem:
                        processed_problems.append(problem)
                except Exception as e:
                    logger.warning(f"Error processing OpenMathInstruct item {idx}: {e}")
                    continue
                    
            logger.info(f"Successfully loaded {len(processed_problems)} OpenMathInstruct-2 problems")
            return processed_problems
            
        except Exception as e:
            logger.error(f"Error loading OpenMathInstruct-2 dataset: {e}")
            return []
    
    def _process_openmath_item(self, item: Dict, idx: int) -> Optional[Dict]:
        """Process individual OpenMathInstruct-2 item"""
        problem = self._clean_text(item.get('problem', ''))
        solution = self._clean_text(item.get('generated_solution', ''))
        expected_answer = self._clean_text(item.get('expected_answer', ''))
        source = item.get('problem_source', 'unknown')
        
        if not problem or not solution:
            return None
            
        # Extract solution steps from code-interpreter format
        solution_steps = self._extract_code_solution_steps(solution)
        
        # Extract final answer if not provided
        if not expected_answer:
            expected_answer = self._extract_answer_from_solution(solution)
            
        return {
            'id': f"openmath_{idx}",
            'source_dataset': 'openmath_instruct_2',
            'question': problem,
            'solution': solution,
            'answer': expected_answer,
            'solution_steps': solution_steps,
            'step_count': len(solution_steps),
            'problem_source': source,
            'raw_data': item
        }
    
    def _extract_code_solution_steps(self, solution: str) -> List[str]:
        """Extract solution steps from code-interpreter format"""
        steps = []
        
        # Split by code blocks and text blocks
        parts = re.split(r'```(?:python)?\n(.*?)\n```', solution, flags=re.DOTALL)
        
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
                
            if i % 2 == 0:  # Text parts
                # Split text into sentences
                sentences = re.split(r'[.!?]+', part)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 15:
                        steps.append(f"Reasoning: {sentence}")
            else:  # Code parts
                if part and len(part) > 10:
                    steps.append(f"Calculation: {part}")
                    
        return steps
    
    def _extract_answer_from_solution(self, solution: str) -> str:
        """Extract final answer from solution"""
        # Look for various answer patterns
        patterns = [
            r'answer.*?is\s*([0-9,.\\$%-]+)',
            r'result.*?is\s*([0-9,.\\$%-]+)',
            r'equals?\s*([0-9,.\\$%-]+)',
            r'final.*?([0-9,.\\$%-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if match:
                return match.group(1)
                
        # Fallback: extract last number
        numbers = re.findall(r'([0-9,.\\$%-]+)', solution)
        if numbers:
            return numbers[-1]
            
        return ""


def load_combined_datasets(gsm8k_split: str = "train", 
                          openmath_max: Optional[int] = 10000,
                          cache_dir: str = "./data/cache") -> List[Dict]:
    """Load and combine both datasets"""
    logger.info("Loading combined datasets...")
    
    # Load GSM8K
    gsm8k_loader = GSM8KLoader(cache_dir=cache_dir, split=gsm8k_split)
    gsm8k_data = gsm8k_loader.load()
    
    # Load OpenMathInstruct-2 (limited sample for initial development)
    openmath_loader = OpenMathInstructLoader(cache_dir=cache_dir, max_samples=openmath_max)
    openmath_data = openmath_loader.load()
    
    # Combine datasets
    combined_data = gsm8k_data + openmath_data
    
    logger.info(f"Combined dataset: {len(gsm8k_data)} GSM8K + {len(openmath_data)} OpenMath = {len(combined_data)} total problems")
    
    return combined_data


if __name__ == "__main__":
    # Test the loaders
    data = load_combined_datasets(openmath_max=100)
    print(f"Loaded {len(data)} problems total")
    
    if data:
        print(f"Sample problem: {data[0]['question'][:100]}...")
        print(f"Sample answer: {data[0]['answer']}") 