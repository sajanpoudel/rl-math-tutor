'''
Any LLM provider that exposes a standard API can be used to generate the dataset for training and evaluating. The implementation uses the OpenAI library as the client, but the underlying model provider can be swapped out as needed.
'''
import os
import json
import ast
import numpy as np
import pandas as pd
import textstat
import re
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from tqdm import tqdm

# ==========================================
# 1. KNOWLEDGE BASE (The Validated Ontology)
# ==========================================

# This map links Sub-Concepts -> (Vector Dimension Index, Intrinsic Load Intensity)
# Derived from our Cognitive Load & IRT Research
INTENSITY_MAP = {
    # ==========================================
    # 1. ARITHMETIC OPERATIONS (Dims 0-9)
    # ==========================================

    # --- ADDITION (Dim 0-1) ---
    "add_single_digit": (0, 0.2),              # Fact retrieval only
    "add_multi_digit_no_carry": (0, 0.4),      # Algorithm without carry
    "add_multi_digit_with_carry": (0, 0.8),    # Working memory load (carrying)
    "add_decimal_alignment": (0, 0.9),         # Alignment + place value

    # --- SUBTRACTION (Dim 2-3) ---
    "sub_single_digit": (2, 0.2),              # Fact retrieval
    "sub_multi_digit_no_borrow": (2, 0.4),     # Algorithm without borrow
    "sub_multi_digit_with_borrow": (2, 0.8),   # Working memory load
    "sub_across_zeros": (2, 1.0),              # High error-prone (buggy rules)

    # --- MULTIPLICATION (Dim 4-5) ---
    "mul_fact_retrieval": (4, 0.2),            # Times tables (0-12)
    "mul_multi_digit_algorithm": (4, 0.8),     # Standard algorithm
    "mul_powers_of_10": (4, 0.3),              # Pattern recognition
    "mul_distributive_property": (5, 0.7),     # Conceptual understanding

    # --- DIVISION (Dim 6-7) ---
    "div_fact_retrieval": (6, 0.2),            # Division facts
    "div_long_division_algorithm": (6, 1.0),   # Multi-step procedural
    "div_with_remainder": (6, 0.7),            # Remainder calculation
    "div_interpret_remainder": (7, 0.9),       # Semantic interpretation (round up/down)

    # --- ORDER OF OPERATIONS (Dim 8-9) ---
    "oop_parentheses_first": (8, 0.5),         # Grouping symbols
    "oop_mul_div_left_to_right": (8, 0.8),     # Left-to-right precedence
    "oop_add_sub_left_to_right": (8, 0.6),     # Secondary precedence
    "oop_nested_grouping": (9, 1.0),           # Multiple levels of nesting

    # ==========================================
    # 2. FRACTIONS & DECIMALS (Dims 10-17)
    # ==========================================

    # --- FRACTIONS (Dim 10-13) ---
    "frac_part_whole_identification": (10, 0.3),    # Conceptual understanding
    "frac_equivalent_generation": (10, 0.6),        # Scaling numerator/denominator
    "frac_simplify": (10, 0.5),                     # GCF application
    "frac_improper_mixed_conversion": (11, 0.6),    # Conversion procedures
    "frac_add_sub_like_denom": (12, 0.5),           # Same denominator
    "frac_add_sub_unlike_denom": (12, 1.0),         # LCD bottleneck (high load)

    # --- DECIMALS (Dim 14-15) ---
    "dec_place_value_identification": (14, 0.4),    # Place value concept
    "dec_rounding": (14, 0.6),                      # Rounding rules
    "dec_comparing_ordering": (14, 0.7),            # Overcoming whole number bias

    # --- FRACTION/DECIMAL CONVERSION (Dim 16-17) ---
    "conv_power_of_10_denom": (16, 0.4),            # Direct conversion (tenths, hundredths)
    "conv_long_division_method": (16, 0.9),         # Procedural division
    "conv_memorized_equivalents": (16, 0.2),        # Retrieval (1/4 = 0.25)

    # ==========================================
    # 3. PERCENTAGES & RATIOS (Dims 18-23)
    # ==========================================

    # --- PERCENTAGES (Dim 18-19) ---
    "perc_part_of_whole": (18, 0.5),                # Find percentage
    "perc_find_part": (18, 0.6),                    # Given percentage, find amount
    "perc_find_whole": (18, 0.8),                   # Reverse calculation
    "perc_increase_decrease": (19, 0.9),            # (New-Old)/Old schema

    # --- RATIOS (Dim 20) ---
    "ratio_identification": (20, 0.5),              # Part-to-Part vs Part-to-Whole
    "ratio_simplification": (20, 0.4),              # Reducing ratios
    "ratio_equivalence_scaling": (20, 0.7),         # Multiplicative reasoning

    # --- PROPORTIONS (Dim 21-22) ---
    "prop_setup": (21, 0.7),                        # Setting up equation (conceptual)
    "prop_cross_product": (21, 0.8),                # Cross multiplication
    "prop_scale_factor_method": (22, 0.9),          # Scaling approach

    # --- RATES (Dim 23) ---
    "rate_unit_rate_calculation": (23, 0.6),        # Cost per unit
    "rate_comparison": (23, 0.7),                   # Comparing rates
    "rate_distance_speed_time": (23, 0.8),          # d = rt applications

    # ==========================================
    # 4. ALGEBRA (Dims 24-29)
    # ==========================================

    # --- VARIABLES (Dim 24-25) ---
    "var_as_unknown": (24, 0.5),                    # Solving for x
    "var_as_varying_quantity": (24, 0.7),           # Function understanding
    "var_substitution": (25, 0.4),                  # Plug in values

    # --- ALGEBRAIC EXPRESSIONS (Dim 26-27) ---
    "expr_modeling": (26, 0.8),                     # Text to symbolic translation
    "expr_simplification": (26, 0.6),               # Combine like terms
    "expr_evaluation": (27, 0.4),                   # Substitute and calculate

    # --- SOLVING EQUATIONS (Dim 28-29) ---
    "eqn_one_step_inverse": (28, 0.5),              # Single operation
    "eqn_two_step_isolation": (28, 0.8),            # Multiple operations
    "eqn_variable_both_sides": (29, 1.0),           # Variable on both sides (high load)

    # ==========================================
    # 5. GEOMETRY & MEASUREMENT (Dims 30-39)
    # ==========================================

    # --- AREA (Dim 30-31) ---
    "area_counting_units": (30, 0.3),               # Conceptual (counting squares)
    "area_formula_rect": (30, 0.5),                 # L × W
    "area_formula_triangle": (31, 0.7),             # 1/2 × b × h
    "area_composite_shapes": (31, 1.0),             # Decomposition strategy (high load)

    # --- PERIMETER (Dim 32) ---
    "perim_sum_of_sides": (32, 0.4),                # Adding side lengths
    "perim_missing_side_deduction": (32, 0.8),      # Logic + geometry hybrid

    # --- VOLUME (Dim 33) ---
    "vol_counting_cubes": (33, 0.4),                # Concrete counting
    "vol_formula_prism": (33, 0.7),                 # L × W × H
    "vol_composite_solids": (33, 1.0),              # 3D decomposition

    # --- ANGLES (Dim 34-35) ---
    "ang_recognition": (34, 0.3),                   # Acute/obtuse/right identification
    "ang_additive_property": (34, 0.7),             # Angle A + Angle B = Total
    "ang_complementary_supplementary": (35, 0.8),   # 90° and 180° relationships

    # --- SHAPES (Dim 36) ---
    "shape_classification_2d": (36, 0.4),           # Polygon identification
    "shape_properties_3d": (36, 0.6),               # Faces/edges/vertices (hierarchical)

    # --- MEASUREMENT (Dim 37) ---
    "meas_reading_tools": (37, 0.5),                # Ruler/scale reading (fractions)
    "meas_unit_estimation": (37, 0.6),              # Estimation skills

    # --- UNIT CONVERSION (Dim 38-39) ---
    "conv_same_system": (38, 0.6),                  # Inches to feet
    "conv_metric_system": (38, 0.4),                # Powers of 10
    "conv_mixed_units": (39, 0.9),                  # Hour/min to minutes

    # ==========================================
    # 6. NUMBER THEORY (Dims 40-42)
    # ==========================================

    # --- FACTORS (Dim 40) ---
    "fact_listing_pairs": (40, 0.4),                # Listing factor pairs
    "fact_gcf_identification": (40, 0.7),           # Greatest Common Factor

    # --- MULTIPLES (Dim 41) ---
    "mult_listing_sequence": (41, 0.3),             # Listing multiples
    "mult_lcm_identification": (41, 0.7),           # Least Common Multiple

    # --- PRIMES (Dim 42) ---
    "prime_identification": (42, 0.5),              # Prime vs composite
    "prime_factorization": (42, 0.8),               # Factor trees (atomic breakdown)

    # --- DIVISIBILITY (Dim 43) ---
    "div_rule_application": (43, 0.4),              # Rules for 2, 3, 5, 10
    "div_remainder_logic": (43, 0.6),               # Modular arithmetic

    # ==========================================
    # 7. STATISTICS & PROBABILITY (Dims 44-48)
    # ==========================================

    # --- MEAN (Dim 44) ---
    "mean_calculation_algorithm": (44, 0.5),        # Sum / Count
    "mean_missing_data_problem": (44, 0.9),         # Algebraic reversal

    # --- MEDIAN (Dim 45) ---
    "med_ordering_data": (45, 0.4),                 # Sorting values
    "med_identification_odd_n": (45, 0.5),          # Middle value
    "med_calculation_even_n": (45, 0.7),            # Average of two middle

    # --- MODE & RANGE (Dim 46) ---
    "mode_identification_single": (46, 0.3),        # Most frequent
    "mode_identification_multi": (46, 0.5),         # Bimodal/multimodal
    "range_calculation": (46, 0.3),                 # Max - Min

    # --- PROBABILITY (Dim 47) ---
    "prob_simple_event": (47, 0.5),                 # Desired/Total
    "prob_compound_independent": (47, 0.9),         # Multiplication rule
    "prob_complement": (47, 0.7),                   # 1 - P(E)

    # --- COUNTING (Dim 48) ---
    "count_permutation_logic": (48, 0.8),           # Order matters
    "count_combination_logic": (48, 0.9),           # Order doesn't matter
    "count_principle_multiplication": (48, 0.6),    # Fundamental Counting Principle

    # ==========================================
    # 8. TIME & MONEY (Dims 49-50)
    # ==========================================

    # --- TIME (Dim 49) ---
    "time_reading_analog": (49, 0.6),               # Clock reading (spatial)
    "time_unit_conversion": (49, 0.5),              # Min ↔ Hr
    "time_add_sub_base_60": (49, 0.9),              # Base-60 carrying (buggy KC)
    "time_interval_counting": (49, 0.7),            # Elapsed time

    # --- MONEY (Dim 50) ---
    "money_decimal_addition": (50, 0.4),            # Decimal arithmetic
    "money_making_change": (50, 0.8),               # Counting up strategy
    "curr_coin_value_recognition": (50, 0.2),       # Fact retrieval (quarter = 25¢)
    "curr_denomination_grouping": (50, 0.6),        # Optimizing coin combinations

    # ==========================================
    # 9. PROBLEM-SOLVING SKILLS (Dims 51-56)
    # ==========================================

    # --- WORD PROBLEM COMPREHENSION (Dim 51-52) ---
    "wp_schema_identification": (51, 0.8),          # Change/Combine/Compare schema
    "wp_variable_definition": (51, 0.6),            # Let x = ...

    # --- MULTI-STEP PROBLEM SOLVING (Dim 53) ---
    "step_decomposition": (53, 0.9),                # Goal/sub-goal hierarchy
    "intermediate_result_management": (53, 1.0),    # Working memory management (high load)

    # --- LOGICAL REASONING (Dim 54) ---
    "logic_deductive_chaining": (54, 0.8),          # If A=B and B=C...
    "logic_constraint_satisfaction": (54, 1.0),     # Multiple simultaneous constraints

    # --- PATTERN RECOGNITION (Dim 55) ---
    "pat_arithmetic_sequence": (55, 0.5),           # +n patterns
    "pat_geometric_sequence": (55, 0.8),            # ×n patterns
    "pat_visual_growth": (55, 0.6),                 # Visual/spatial patterns

    # --- ESTIMATION (Dim 56) ---
    "est_rounding_strategy": (56, 0.5),             # Strategic rounding
    "est_magnitude_check": (56, 0.6),               # Metacognitive verification

    # --- DATA INTERPRETATION (Dim 57) ---
    "data_graph_reading": (57, 0.6),                # Bar/line graph reading
    "data_key_extraction": (57, 0.7),               # Extracting relevant info from tables
}

# ==========================================
# 1B. CONCEPT HIERARCHY (High-Level -> Sub-Concepts)
# ==========================================

# Maps high-level concepts to their constituent sub-concepts
# This allows the LLM to focus on relevant sub-concepts based on the problem's concepts
CONCEPT_TO_SUBCONCEPTS = {
    # --- ARITHMETIC OPERATIONS ---
    "addition": [
        "add_single_digit",
        "add_multi_digit_no_carry",
        "add_multi_digit_with_carry",
        "add_decimal_alignment"
    ],

    "subtraction": [
        "sub_single_digit",
        "sub_multi_digit_no_borrow",
        "sub_multi_digit_with_borrow",
        "sub_across_zeros"
    ],

    "multiplication": [
        "mul_fact_retrieval",
        "mul_multi_digit_algorithm",
        "mul_powers_of_10",
        "mul_distributive_property"
    ],

    "division": [
        "div_fact_retrieval",
        "div_long_division_algorithm",
        "div_with_remainder",
        "div_interpret_remainder"
    ],

    "order_of_operations": [
        "oop_parentheses_first",
        "oop_mul_div_left_to_right",
        "oop_add_sub_left_to_right",
        "oop_nested_grouping"
    ],

    # --- FRACTIONS & DECIMALS ---
    "fractions": [
        "frac_part_whole_identification",
        "frac_equivalent_generation",
        "frac_simplify",
        "frac_improper_mixed_conversion",
        "frac_add_sub_like_denom",
        "frac_add_sub_unlike_denom"
    ],

    "decimals": [
        "dec_place_value_identification",
        "dec_rounding",
        "dec_comparing_ordering"
    ],

    "fraction_to_decimal_conversion": [
        "conv_power_of_10_denom",
        "conv_long_division_method",
        "conv_memorized_equivalents"
    ],

    # --- PERCENTAGES & RATIOS ---
    "percentages": [
        "perc_part_of_whole",
        "perc_find_part",
        "perc_find_whole",
        "perc_increase_decrease"
    ],

    "ratios": [
        "ratio_identification",
        "ratio_simplification",
        "ratio_equivalence_scaling"
    ],

    "proportions": [
        "prop_setup",
        "prop_cross_product",
        "prop_scale_factor_method"
    ],

    "rates": [
        "rate_unit_rate_calculation",
        "rate_comparison",
        "rate_distance_speed_time"
    ],

    # --- ALGEBRA ---
    "variables": [
        "var_as_unknown",
        "var_as_varying_quantity",
        "var_substitution"
    ],

    "algebraic_expressions": [
        "expr_modeling",
        "expr_simplification",
        "expr_evaluation"
    ],

    "solving_equations": [
        "eqn_one_step_inverse",
        "eqn_two_step_isolation",
        "eqn_variable_both_sides"
    ],

    # --- GEOMETRY & MEASUREMENT ---
    "area": [
        "area_counting_units",
        "area_formula_rect",
        "area_formula_triangle",
        "area_composite_shapes"
    ],

    "perimeter": [
        "perim_sum_of_sides",
        "perim_missing_side_deduction"
    ],

    "volume": [
        "vol_counting_cubes",
        "vol_formula_prism",
        "vol_composite_solids"
    ],

    "angles": [
        "ang_recognition",
        "ang_additive_property",
        "ang_complementary_supplementary"
    ],

    "shapes": [
        "shape_classification_2d",
        "shape_properties_3d"
    ],

    "measurement": [
        "meas_reading_tools",
        "meas_unit_estimation"
    ],

    "unit_conversion": [
        "conv_same_system",
        "conv_metric_system",
        "conv_mixed_units"
    ],

    # --- NUMBER THEORY ---
    "factors": [
        "fact_listing_pairs",
        "fact_gcf_identification"
    ],

    "multiples": [
        "mult_listing_sequence",
        "mult_lcm_identification"
    ],

    "primes": [
        "prime_identification",
        "prime_factorization"
    ],

    "divisibility": [
        "div_rule_application",
        "div_remainder_logic"
    ],

    # --- STATISTICS & PROBABILITY ---
    "mean_average": [
        "mean_calculation_algorithm",
        "mean_missing_data_problem"
    ],

    "median": [
        "med_ordering_data",
        "med_identification_odd_n",
        "med_calculation_even_n"
    ],

    "mode": [
        "mode_identification_single",
        "mode_identification_multi"
    ],

    "range": [
        "range_calculation"
    ],

    "probability": [
        "prob_simple_event",
        "prob_compound_independent",
        "prob_complement"
    ],

    "counting": [
        "count_permutation_logic",
        "count_combination_logic",
        "count_principle_multiplication"
    ],

    # --- TIME & MONEY ---
    "time_calculations": [
        "time_reading_analog",
        "time_unit_conversion"
    ],

    "elapsed_time": [
        "time_add_sub_base_60",
        "time_interval_counting"
    ],

    "money_calculations": [
        "money_decimal_addition",
        "money_making_change"
    ],

    "currency": [
        "curr_coin_value_recognition",
        "curr_denomination_grouping"
    ],

    # --- PROBLEM-SOLVING SKILLS ---
    "word_problem_comprehension": [
        "wp_schema_identification",
        "wp_variable_definition"
    ],

    "multi_step_problem_solving": [
        "step_decomposition",
        "intermediate_result_management"
    ],

    "logical_reasoning": [
        "logic_deductive_chaining",
        "logic_constraint_satisfaction"
    ],

    "pattern_recognition": [
        "pat_arithmetic_sequence",
        "pat_geometric_sequence",
        "pat_visual_growth"
    ],

    "estimation": [
        "est_rounding_strategy",
        "est_magnitude_check"
    ],

    "data_interpretation": [
        "data_graph_reading",
        "data_key_extraction"
    ]
}

# ==========================================
# 2. LLM INTERFACE (Structured Output)
# ==========================================

class SubConceptAnalysis(BaseModel):
    """SOTA Pydantic model for validating LLM output."""
    identified_sub_concepts: List[str] = Field(
        ...,
        description="List of precise sub-concept tags found in the problem solution."
    )
    reasoning: str = Field(..., description="Brief explanation of why these tags were chosen.")

class SubConceptExtractor:
    def __init__(self, api_key: str, model="gpt-5", concept_map=CONCEPT_TO_SUBCONCEPTS):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.concept_map = concept_map

    def _get_relevant_subconcepts(self, high_level_concepts: List[str]) -> List[str]:
        """
        Filters sub-concepts based on high-level concepts.
        Only returns sub-concepts that are relevant to the problem's high-level concepts.
        """
        relevant_subconcepts = set()
        for concept in high_level_concepts:
            # Normalize to lowercase and strip whitespace for robust matching
            clean_concept = str(concept).lower().strip()
            if clean_concept in self.concept_map:
                relevant_subconcepts.update(self.concept_map[clean_concept])
        return sorted(list(relevant_subconcepts))

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def analyze_problem(self, question: str, solution: str, high_level_concepts: List[str]) -> SubConceptAnalysis:
        """
        Uses Chain-of-Thought prompting to classify problem into Micro-KCs.
        Now uses hierarchical filtering based on high-level concepts.
        """
        # Get only the relevant sub-concepts for this problem
        relevant_subconcepts = self._get_relevant_subconcepts(high_level_concepts)
        valid_tags_str = ", ".join(relevant_subconcepts)

        system_prompt = (
            "You are an Expert Math Pedagogue. Your goal is to tag math problems with "
            "precise 'Micro-Knowledge Components' (Sub-Concepts) based on the solution steps.\n\n"
            f"HIGH-LEVEL CONCEPTS: {', '.join(high_level_concepts)}\n\n"
            f"VALID SUB-CONCEPTS FOR THIS PROBLEM:\n[{valid_tags_str}]\n\n"
            "RULES:\n"
            "1. Only use sub-concept tags from the provided list above.\n"
            "2. Analyze the 'Solution' text carefully for procedural steps.\n"
            "3. Examples:\n"
            "   - If adding with carrying: 'add_multi_digit_with_carry'\n"
            "   - If adding without carrying: 'add_multi_digit_no_carry'\n"
            "   - If dividing with remainder: 'div_with_remainder'\n"
            "   - If finding LCD for fractions: 'frac_add_sub_unlike_denom'\n"
            "4. Be specific: choose the most granular sub-concept that accurately describes each step.\n"
            "5. Return output in JSON format with keys 'identified_sub_concepts' (list) and 'reasoning' (string)."
        )

        user_content = (
            f"Question: {question}\n"
            f"Solution: {solution}\n\n"
            "Identify the specific sub-concepts used in this solution:"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.0  # Deterministic output
            )

            # Parse and Validate
            content = response.choices[0].message.content
            data = json.loads(content)

            # Validate: Only accept sub-concepts that are both in INTENSITY_MAP
            # AND in the relevant_subconcepts for this problem
            valid_subconcepts = [
                tag for tag in data.get("identified_sub_concepts", [])
                if tag in INTENSITY_MAP and tag in relevant_subconcepts
            ]

            return SubConceptAnalysis(
                identified_sub_concepts=valid_subconcepts,
                reasoning=data.get("reasoning", "")
            )

        except Exception as e:
            print(f"LLM Error: {e}")
            return SubConceptAnalysis(identified_sub_concepts=[], reasoning="Error")

# ==========================================
# 3. VECTORIZATION ENGINE (HIE Logic)
# ==========================================

class ConceptVectorizer:
    def __init__(self, intensity_map=INTENSITY_MAP):
        self.map = intensity_map
        self.dim = 58  # Updated to accommodate all sub-concepts (Dims 0-57)

    def vectorize(self, sub_concepts: List[str]):
        """
        Implements Hierarchical Intensity Encoding with Max-Pooling.
        Returns:
            vector_raw: The intensity magnitudes (for Difficulty Calc)
            vector_norm: The L2 normalized vector (for RL State)
        """
        vector_raw = np.zeros(self.dim, dtype=np.float32)

        for tag in sub_concepts:
            if tag in self.map:
                dim_idx, intensity = self.map[tag]
                # MAX-POOLING: Keep the highest intensity for this dimension
                # If problem has 'add_simple' (0.2) and 'add_carry' (0.8),
                # the dimension represents the PEAK load (0.8).
                vector_raw[dim_idx] = max(vector_raw[dim_idx], intensity)

        # L2 Normalization for RL
        magnitude = np.linalg.norm(vector_raw)
        if magnitude > 0:
            vector_norm = vector_raw / magnitude
        else:
            vector_norm = vector_raw # Zero vector

        return vector_raw, vector_norm

# ==========================================
# 4. PSYCHOMETRIC DIFFICULTY CALCULATOR
# ==========================================

class DifficultyEstimator:
    @staticmethod
    def calculate(row, vector_raw, sub_concepts):
        """
        Calculates difficulty based on Intrinsic, Extraneous, and Germane load.
        """
        # --- A. INTRINSIC LOAD (Math Intensity) ---
        # This is Concept Complexity based and Numerical Complexity on our proposed model.
        # Based on Cowan (2001) Working Memory Capacity limit (~4 chunks).
        # We sum the Top 3 to model "Peak Element Interactivity".

        active_intensities = sorted([x for x in vector_raw if x > 0], reverse=True)
        if active_intensities:
            # If the student is juggling more than 3 hard concepts,
            # their Working Memory (Capacity ~4) is likely overloaded.
            # We measure this "Peak Load" rather than the total task length.
            top_k = active_intensities[:3]
            intrinsic_score = sum(top_k) / len(top_k) # Normalized to 0-1
        else:
            intrinsic_score = 0.1 # Baseline load

        # --- B. EXTRANEOUS LOAD (Structural/Linguistic Noise) ---
        # 1. Step Count (Heuristic: more steps = more noise/tracking)
        # Count lines in solution or sentences
        steps = len(str(row.get('answer', '')).split('\n'))
        step_factor = min(steps / 10.0, 1.0)

        # 2. Reading Difficulty (Textstat)
        question_text = str(row.get('question', ''))
        try:
            flesch = textstat.flesch_reading_ease(question_text) # Score = 206.835 - (1.015 * Avg Sentence Length) - (84.6 * Avg Syllables per Word)
            # Flesch: 100=Easy, 0=Hard. Map to 0.0-1.0
            # Clamp Flesch to 0-100 range first for safety (can go below 0 or above 100 in extreme cases)
            flesch_clamped = max(0, min(100, flesch))
            # Invert: 100 (Easy) -> 0.0 Load, 0 (Hard) -> 1.0 Load
            readability_load = (100 - flesch_clamped) / 100.0  # We are using the inverse of the Flesch score to model the reading difficulty .

        except:
            readability_load = 0.5

        extraneous_score = (0.6 * step_factor) + (0.4 * readability_load) # This are Factor 1 and Factor 4 in our proposed model.

        # --- C. GERMANE LOAD (Depth/Prerequisites) ---
        # Currently approximated by the count of distinct sub-concepts
        # (richer integration = higher germane load)

        # We divide by 5.0 as an implementation cap for the integration-depth proxy.
        # Halford et al. (2005) place four-variable relations near human processing
        # capacity and show that five-way interactions can fall to chance-level
        # performance. Sub-concept count is only a proxy here, not a claim that
        # every five-sub-concept problem forms a true five-variable relation.

        depth_factor = min(len(sub_concepts) / 5.0, 1.0)   # In our proposed model, this is Factor 5

        # --- TOTAL WEIGHTED SCORE ---
        # Weights based on Sweller's Cognitive Load Theory
        # Intrinsic is dominant.
        difficulty = (0.50 * intrinsic_score) + \
                     (0.30 * extraneous_score) + \
                     (0.20 * depth_factor)

        return round(max(0.1, min(1.0, difficulty)), 4)

# ==========================================
# 5. MAIN PIPELINE
# ==========================================

def process_dataset(input_csv: str, output_csv: str, api_key: str):
    # 1. Load Data
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)

    # Check required columns
    required = ['question', 'answer', 'concepts']
    if not all(col in df.columns for col in required):
        raise ValueError(f"Input CSV must contain: {required}")

    # 2. Initialize Modules
    extractor = SubConceptExtractor(api_key=api_key)
    vectorizer = ConceptVectorizer()
    estimator = DifficultyEstimator()

    # 3. Processing Loop
    # We create lists to store new data columns
    new_sub_concepts = []
    new_vectors_raw = []
    new_vectors_norm = []
    new_difficulties = []
    new_reasonings = []

    print("Starting processing pipeline (this may take time with LLM calls)...")

    # Use tqdm for progress bar
    # NOTE: For large datasets, consider processing in batches and saving incrementally
    for idx, row in tqdm(df.iterrows(), total=len(df)):

        # A. Extract Sub-Concepts (LLM)
        # Robust parsing of concepts list (handles apostrophes, mixed quotes, etc.)
        concepts_val = row['concepts']
        concepts_list = []

        if isinstance(concepts_val, list):
            concepts_list = concepts_val
        elif isinstance(concepts_val, str):
            try:
                # Use ast.literal_eval for safe parsing (handles apostrophes correctly)
                concepts_list = ast.literal_eval(concepts_val)
            except:
                # Fallback for simple comma-separated strings
                try:
                    concepts_list = [c.strip() for c in concepts_val.strip("[]").replace("'", "").replace('"', '').split(',') if c.strip()]
                except:
                    concepts_list = []  # Final fallback

        analysis = extractor.analyze_problem(
            question=row['question'],
            solution=row['answer'],
            high_level_concepts=concepts_list
        )

        # B. Vectorize (Math)
        # Safety check for empty analysis
        if not analysis.identified_sub_concepts:
            v_raw = np.zeros(vectorizer.dim, dtype=np.float32)
            v_norm = np.zeros(vectorizer.dim, dtype=np.float32)
            diff_score = 0.1  # Default baseline
        else:
            v_raw, v_norm = vectorizer.vectorize(analysis.identified_sub_concepts)

            # C. Calculate Difficulty (Psychometrics)
            diff_score = estimator.calculate(row, v_raw, analysis.identified_sub_concepts)

        # Store results
        new_sub_concepts.append(json.dumps(analysis.identified_sub_concepts))
        new_vectors_raw.append(json.dumps(v_raw.tolist())) # Save as JSON string
        new_vectors_norm.append(json.dumps(v_norm.tolist()))
        new_difficulties.append(diff_score)
        new_reasonings.append(analysis.reasoning)

    # 4. Update DataFrame
    df['sub_concepts'] = new_sub_concepts
    df['concept_vector_raw'] = new_vectors_raw # Useful for debugging/re-calcs
    df['concept_vector_norm'] = new_vectors_norm # THE RL STATE
    df['difficulty_score'] = new_difficulties
    df['llm_reasoning'] = new_reasonings

    # 5. Save
    print(f"Saving enriched dataset to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print("Pipeline Complete.")

# ==========================================
# EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # Example Usage
    # Ensure OPENAI_API_KEY is set in environment or passed directly
    API_KEY = os.getenv("OPENAI_API_KEY")

    if API_KEY:
        # Create a dummy CSV for testing if it doesn't exist
        if not os.path.exists("gsm8k_processed_stage3.csv"):
            print("No input file found. Please place 'gsm8k_processed_stage3.csv' in directory.")
        else:
            process_dataset(
                input_csv="gsm8k_processed_stage3.csv",
                output_csv="gsm8k_ready_for_rl_stage5.csv",
                api_key=API_KEY
            )
    else:
        print("Please set OPENAI_API_KEY to run the LLM extraction.")
