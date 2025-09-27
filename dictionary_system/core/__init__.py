"""
Core module for dictionary system
"""

from .models import Term, BaseExtractor, ExtractorFactory
from .extractors.statistical_extractor_v2 import EnhancedTermExtractorV3

__all__ = [
    'Term',
    'BaseExtractor',
    'ExtractorFactory',
    'EnhancedTermExtractorV3'
]