"""
Core module for dictionary system
"""

from .models import Term, BaseExtractor, ExtractorFactory
from .extractors.unified_extractor import UnifiedTermExtractor

__all__ = [
    'Term',
    'BaseExtractor',
    'ExtractorFactory',
    'UnifiedTermExtractor'
]