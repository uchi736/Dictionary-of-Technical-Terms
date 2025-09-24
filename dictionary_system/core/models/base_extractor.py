"""
Base classes and data models for term extraction system
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Term:
    """用語データモデル"""
    term: str
    score: float = 0.0
    category: Optional[str] = None
    definition: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'term': self.term,
            'score': self.score,
            'category': self.category,
            'definition': self.definition,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    def __str__(self) -> str:
        return f"{self.term} (score: {self.score:.3f})"

    def __hash__(self) -> int:
        return hash(self.term)

    def __eq__(self, other) -> bool:
        if isinstance(other, Term):
            return self.term == other.term
        return False


class BaseExtractor(ABC):
    """抽出器の基底クラス"""

    def __init__(self, **kwargs):
        """
        Initialize the extractor with configuration

        Args:
            **kwargs: Configuration parameters for the extractor
        """
        self.config = kwargs

    @abstractmethod
    def extract(self, text: str, **kwargs) -> List[Term]:
        """
        Extract terms from text

        Args:
            text: Input text to extract terms from
            **kwargs: Additional parameters for extraction

        Returns:
            List of extracted Term objects
        """
        pass

    def preprocess(self, text: str) -> str:
        """
        Preprocess text before extraction

        Args:
            text: Raw input text

        Returns:
            Preprocessed text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def postprocess(self, terms: List[Term]) -> List[Term]:
        """
        Postprocess extracted terms

        Args:
            terms: List of extracted terms

        Returns:
            Processed list of terms
        """
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            if term.term not in seen:
                seen.add(term.term)
                unique_terms.append(term)

        # Sort by score
        return sorted(unique_terms, key=lambda x: x.score, reverse=True)

    def filter_terms(self, terms: List[Term], min_score: float = 0.0) -> List[Term]:
        """
        Filter terms by minimum score

        Args:
            terms: List of terms to filter
            min_score: Minimum score threshold

        Returns:
            Filtered list of terms
        """
        return [t for t in terms if t.score >= min_score]


class ExtractorFactory:
    """抽出器のファクトリクラス"""

    _extractors = {}

    @classmethod
    def register(cls, name: str, extractor_class: type):
        """
        Register an extractor class

        Args:
            name: Name identifier for the extractor
            extractor_class: Class of the extractor
        """
        cls._extractors[name] = extractor_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseExtractor:
        """
        Create an extractor instance

        Args:
            name: Name of the extractor to create
            **kwargs: Configuration for the extractor

        Returns:
            Extractor instance
        """
        if name not in cls._extractors:
            raise ValueError(f"Unknown extractor: {name}")
        return cls._extractors[name](**kwargs)

    @classmethod
    def list_extractors(cls) -> List[str]:
        """
        List available extractors

        Returns:
            List of registered extractor names
        """
        return list(cls._extractors.keys())