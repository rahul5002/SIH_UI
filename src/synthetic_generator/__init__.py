"""
Synthetic Dataset & Benchmark Module
"""
from src.synthetic_generator.generator import SyntheticDocumentGenerator
from src.synthetic_generator.tampering_injector import TamperingInjector
from src.synthetic_generator.benchmark import ScreeningBenchmark

__all__ = [
    "SyntheticDocumentGenerator",
    "TamperingInjector",
    "ScreeningBenchmark"
]
