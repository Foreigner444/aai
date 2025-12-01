"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                      LESSON 12: CONFIDENCE SCORING                           ║
║                                                                              ║
║           Evaluating How Confident We Should Be in RAG Answers               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 CONCEPT OVERVIEW
==================

What & Why:
-----------
Confidence scoring helps you understand how trustworthy a RAG answer is.
Not all answers are created equal - some are well-supported by multiple
high-quality sources, while others are based on weak or tangential evidence.

Why confidence scoring matters:
- Users can calibrate trust in the answer
- Low confidence answers can trigger human review
- Helps identify gaps in your knowledge base
- Enables automated quality control

🎯 Real-World Analogy:
----------------------
Think of confidence scoring like a weather forecast:

Without Confidence:
"It will rain tomorrow." (Is this 50%? 90%? Who knows!)

With Confidence:
"90% chance of rain tomorrow" (High confidence - bring an umbrella!)
"30% chance of rain tomorrow" (Low confidence - maybe bring one just in case)

Confidence scoring tells users how much to trust the answer!

🔒 Type Safety Benefit:
-----------------------
With Pydantic, confidence scores are:
- Always in valid range (0.0 to 1.0)
- Never null or undefined
- Properly typed for calculations
- Validated at every step


💻 CODE IMPLEMENTATION
=====================
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, computed_field
from enum import Enum
import math


class ConfidenceLevel(str, Enum):
    """Human-readable confidence levels."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"
    UNCERTAIN = "uncertain"
    
    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Convert numeric score to confidence level."""
        if score >= 0.9:
            return cls.VERY_HIGH
        elif score >= 0.75:
            return cls.HIGH
        elif score >= 0.5:
            return cls.MEDIUM
        elif score >= 0.3:
            return cls.LOW
        elif score >= 0.1:
            return cls.VERY_LOW
        else:
            return cls.UNCERTAIN


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    source: str
    title: Optional[str] = None


class RetrievedChunk(BaseModel):
    """A retrieved chunk with similarity."""
    chunk_id: str
    content: str
    metadata: DocumentMetadata
    similarity_score: float = Field(ge=0, le=1)


class ConfidenceFactors(BaseModel):
    """
    Individual factors that contribute to overall confidence.
    Each factor can be weighted differently.
    """
    
    retrieval_similarity: float = Field(
        ge=0.0,
        le=1.0,
        description="Average similarity of retrieved chunks"
    )
    
    source_coverage: float = Field(
        ge=0.0,
        le=1.0,
        description="How well sources cover the query"
    )
    
    source_agreement: float = Field(
        ge=0.0,
        le=1.0,
        description="How much sources agree with each other"
    )
    
    source_quality: float = Field(
        ge=0.0,
        le=1.0,
        description="Quality/reliability of sources"
    )
    
    recency: float = Field(
        ge=0.0,
        le=1.0,
        description="How recent the sources are"
    )
    
    query_specificity: float = Field(
        ge=0.0,
        le=1.0,
        description="How specific/clear the query is"
    )
    
    context_relevance: float = Field(
        ge=0.0,
        le=1.0,
        description="How relevant the context is"
    )
    
    def to_dict(self) -> dict[str, float]:
        """Get all factors as a dictionary."""
        return {
            "retrieval_similarity": self.retrieval_similarity,
            "source_coverage": self.source_coverage,
            "source_agreement": self.source_agreement,
            "source_quality": self.source_quality,
            "recency": self.recency,
            "query_specificity": self.query_specificity,
            "context_relevance": self.context_relevance,
        }


class ConfidenceWeights(BaseModel):
    """
    Weights for different confidence factors.
    Adjust based on your domain and requirements.
    """
    
    retrieval_similarity: float = Field(default=0.25, ge=0, le=1)
    source_coverage: float = Field(default=0.20, ge=0, le=1)
    source_agreement: float = Field(default=0.15, ge=0, le=1)
    source_quality: float = Field(default=0.15, ge=0, le=1)
    recency: float = Field(default=0.10, ge=0, le=1)
    query_specificity: float = Field(default=0.10, ge=0, le=1)
    context_relevance: float = Field(default=0.05, ge=0, le=1)
    
    @computed_field
    @property
    def total_weight(self) -> float:
        """Sum of all weights (should equal 1.0)."""
        return (
            self.retrieval_similarity +
            self.source_coverage +
            self.source_agreement +
            self.source_quality +
            self.recency +
            self.query_specificity +
            self.context_relevance
        )
    
    def normalize(self) -> "ConfidenceWeights":
        """Normalize weights to sum to 1.0."""
        total = self.total_weight
        if total == 0:
            return self
        
        return ConfidenceWeights(
            retrieval_similarity=self.retrieval_similarity / total,
            source_coverage=self.source_coverage / total,
            source_agreement=self.source_agreement / total,
            source_quality=self.source_quality / total,
            recency=self.recency / total,
            query_specificity=self.query_specificity / total,
            context_relevance=self.context_relevance / total,
        )


class ConfidenceScore(BaseModel):
    """
    Complete confidence score with breakdown.
    """
    
    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence score"
    )
    
    level: ConfidenceLevel = Field(
        description="Human-readable confidence level"
    )
    
    factors: ConfidenceFactors = Field(
        description="Individual factor scores"
    )
    
    weights_used: ConfidenceWeights = Field(
        description="Weights applied to factors"
    )
    
    explanation: str = Field(
        description="Human-readable explanation"
    )
    
    recommendations: list[str] = Field(
        default_factory=list,
        description="Recommendations to improve confidence"
    )
    
    @computed_field
    @property
    def is_trustworthy(self) -> bool:
        """Check if confidence is high enough to trust."""
        return self.overall_score >= 0.6
    
    @computed_field
    @property
    def needs_review(self) -> bool:
        """Check if this answer needs human review."""
        return self.overall_score < 0.5


class ConfidenceScorer:
    """
    Calculates confidence scores for RAG responses.
    """
    
    def __init__(self, weights: Optional[ConfidenceWeights] = None):
        """Initialize with optional custom weights."""
        self.weights = (weights or ConfidenceWeights()).normalize()
    
    def calculate_retrieval_similarity(
        self,
        chunks: list[RetrievedChunk]
    ) -> float:
        """Calculate average retrieval similarity."""
        if not chunks:
            return 0.0
        
        scores = [c.similarity_score for c in chunks]
        return sum(scores) / len(scores)
    
    def calculate_source_coverage(
        self,
        chunks: list[RetrievedChunk],
        min_chunks: int = 3
    ) -> float:
        """
        Calculate how well sources cover the query.
        Based on number of relevant chunks found.
        """
        if not chunks:
            return 0.0
        
        relevant_chunks = [c for c in chunks if c.similarity_score >= 0.5]
        coverage = min(len(relevant_chunks) / min_chunks, 1.0)
        
        return coverage
    
    def calculate_source_agreement(
        self,
        chunks: list[RetrievedChunk]
    ) -> float:
        """
        Calculate how much sources agree with each other.
        Uses content similarity between chunks.
        """
        if len(chunks) < 2:
            return 1.0
        
        top_scores = sorted(
            [c.similarity_score for c in chunks],
            reverse=True
        )[:3]
        
        if len(top_scores) < 2:
            return 1.0
        
        variance = sum((s - sum(top_scores)/len(top_scores))**2 
                       for s in top_scores) / len(top_scores)
        
        agreement = 1.0 - min(variance * 4, 1.0)
        return max(0.0, agreement)
    
    def calculate_source_quality(
        self,
        chunks: list[RetrievedChunk],
        quality_sources: Optional[list[str]] = None
    ) -> float:
        """
        Calculate quality based on source reliability.
        """
        if not chunks:
            return 0.0
        
        quality_sources = quality_sources or [
            "official", "documentation", "policy",
            "handbook", "guide", "manual"
        ]
        
        quality_count = 0
        for chunk in chunks:
            source_lower = chunk.metadata.source.lower()
            title_lower = (chunk.metadata.title or "").lower()
            combined = source_lower + " " + title_lower
            
            if any(q in combined for q in quality_sources):
                quality_count += 1
        
        return min(quality_count / max(len(chunks), 1), 1.0)
    
    def calculate_recency(
        self,
        chunks: list[RetrievedChunk],
        max_age_days: int = 365
    ) -> float:
        """
        Calculate recency score.
        Without actual dates, return neutral score.
        """
        return 0.7
    
    def calculate_query_specificity(
        self,
        query: str
    ) -> float:
        """
        Calculate how specific/clear the query is.
        Longer, more detailed queries are more specific.
        """
        words = query.split()
        word_count = len(words)
        
        if word_count < 3:
            return 0.4
        elif word_count < 6:
            return 0.6
        elif word_count < 10:
            return 0.8
        else:
            return 0.9
    
    def calculate_context_relevance(
        self,
        chunks: list[RetrievedChunk],
        query: str
    ) -> float:
        """
        Calculate how relevant the context is to the query.
        """
        if not chunks:
            return 0.0
        
        query_words = set(query.lower().split())
        
        if not query_words:
            return 0.5
        
        relevance_scores = []
        for chunk in chunks:
            content_words = set(chunk.content.lower().split())
            overlap = len(query_words & content_words)
            relevance = min(overlap / len(query_words), 1.0)
            relevance_scores.append(relevance)
        
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
    
    def score(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        quality_sources: Optional[list[str]] = None
    ) -> ConfidenceScore:
        """
        Calculate complete confidence score.
        """
        factors = ConfidenceFactors(
            retrieval_similarity=self.calculate_retrieval_similarity(chunks),
            source_coverage=self.calculate_source_coverage(chunks),
            source_agreement=self.calculate_source_agreement(chunks),
            source_quality=self.calculate_source_quality(chunks, quality_sources),
            recency=self.calculate_recency(chunks),
            query_specificity=self.calculate_query_specificity(query),
            context_relevance=self.calculate_context_relevance(chunks, query),
        )
        
        overall = (
            factors.retrieval_similarity * self.weights.retrieval_similarity +
            factors.source_coverage * self.weights.source_coverage +
            factors.source_agreement * self.weights.source_agreement +
            factors.source_quality * self.weights.source_quality +
            factors.recency * self.weights.recency +
            factors.query_specificity * self.weights.query_specificity +
            factors.context_relevance * self.weights.context_relevance
        )
        
        overall = max(0.0, min(1.0, overall))
        
        level = ConfidenceLevel.from_score(overall)
        
        explanation = self._generate_explanation(factors, overall)
        recommendations = self._generate_recommendations(factors)
        
        return ConfidenceScore(
            overall_score=overall,
            level=level,
            factors=factors,
            weights_used=self.weights,
            explanation=explanation,
            recommendations=recommendations
        )
    
    def _generate_explanation(
        self,
        factors: ConfidenceFactors,
        overall: float
    ) -> str:
        """Generate human-readable explanation."""
        level = ConfidenceLevel.from_score(overall)
        
        if level in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH]:
            base = "High confidence in this answer."
        elif level == ConfidenceLevel.MEDIUM:
            base = "Moderate confidence - answer is likely correct but verify important details."
        else:
            base = "Low confidence - answer may be incomplete or inaccurate."
        
        details = []
        if factors.retrieval_similarity >= 0.8:
            details.append("Sources are highly relevant.")
        elif factors.retrieval_similarity < 0.5:
            details.append("Sources have limited relevance.")
        
        if factors.source_coverage >= 0.8:
            details.append("Good coverage from multiple sources.")
        elif factors.source_coverage < 0.5:
            details.append("Limited source coverage.")
        
        if factors.source_agreement >= 0.8:
            details.append("Sources are consistent.")
        elif factors.source_agreement < 0.5:
            details.append("Sources show some disagreement.")
        
        return base + " " + " ".join(details)
    
    def _generate_recommendations(
        self,
        factors: ConfidenceFactors
    ) -> list[str]:
        """Generate recommendations to improve confidence."""
        recommendations = []
        
        if factors.retrieval_similarity < 0.6:
            recommendations.append(
                "Add more specific documents related to this topic."
            )
        
        if factors.source_coverage < 0.6:
            recommendations.append(
                "Expand knowledge base with additional sources."
            )
        
        if factors.source_agreement < 0.6:
            recommendations.append(
                "Review conflicting sources and update outdated information."
            )
        
        if factors.source_quality < 0.6:
            recommendations.append(
                "Add authoritative sources (official docs, policies)."
            )
        
        if factors.query_specificity < 0.5:
            recommendations.append(
                "Encourage users to ask more specific questions."
            )
        
        return recommendations


class ScoredResponse(BaseModel):
    """
    A RAG response with confidence scoring.
    """
    
    query: str = Field(description="Original query")
    answer: str = Field(description="Generated answer")
    confidence: ConfidenceScore = Field(description="Confidence assessment")
    chunks_used: int = Field(ge=0, description="Number of chunks used")
    generated_at: datetime = Field(default_factory=datetime.now)
    
    def should_show_warning(self) -> bool:
        """Check if a confidence warning should be shown."""
        return self.confidence.overall_score < 0.6
    
    def get_warning_message(self) -> Optional[str]:
        """Get warning message if needed."""
        if not self.should_show_warning():
            return None
        
        if self.confidence.overall_score < 0.3:
            return "⚠️ Low confidence: This answer may not be reliable. Please verify with other sources."
        else:
            return "ℹ️ Moderate confidence: Some details may need verification."


def demonstrate_confidence_scoring():
    """
    Demonstrate confidence scoring for RAG.
    """
    
    print("=" * 70)
    print("CONFIDENCE SCORING DEMONSTRATION")
    print("=" * 70)
    
    scorer = ConfidenceScorer()
    
    print("\n📊 CONFIDENCE WEIGHTS")
    print("-" * 50)
    print(f"Retrieval Similarity: {scorer.weights.retrieval_similarity:.0%}")
    print(f"Source Coverage: {scorer.weights.source_coverage:.0%}")
    print(f"Source Agreement: {scorer.weights.source_agreement:.0%}")
    print(f"Source Quality: {scorer.weights.source_quality:.0%}")
    print(f"Recency: {scorer.weights.recency:.0%}")
    print(f"Query Specificity: {scorer.weights.query_specificity:.0%}")
    print(f"Context Relevance: {scorer.weights.context_relevance:.0%}")
    print(f"Total: {scorer.weights.total_weight:.0%}")
    
    print("\n" + "=" * 70)
    print("SCENARIO 1: HIGH CONFIDENCE")
    print("=" * 70)
    
    high_conf_chunks = [
        RetrievedChunk(
            chunk_id="chunk_001",
            content="Full-time employees receive 20 days of paid vacation per year. "
                   "This policy applies to all permanent staff members.",
            metadata=DocumentMetadata(
                source="hr/official_policy.md",
                title="Official HR Policy"
            ),
            similarity_score=0.95
        ),
        RetrievedChunk(
            chunk_id="chunk_002",
            content="The vacation allowance is 20 days annually for full-time positions. "
                   "Part-time employees receive prorated amounts.",
            metadata=DocumentMetadata(
                source="hr/employee_handbook.md",
                title="Employee Handbook"
            ),
            similarity_score=0.91
        ),
        RetrievedChunk(
            chunk_id="chunk_003",
            content="According to company policy, vacation days total 20 per year. "
                   "Unused days can carry over up to 5 days.",
            metadata=DocumentMetadata(
                source="hr/documentation.md",
                title="HR Documentation"
            ),
            similarity_score=0.88
        ),
    ]
    
    query1 = "How many vacation days do full-time employees get per year?"
    score1 = scorer.score(query1, high_conf_chunks)
    
    print(f"\n🔍 Query: '{query1}'")
    print(f"\n📈 Confidence Score: {score1.overall_score:.2f} ({score1.level.value})")
    print(f"   Is Trustworthy: {score1.is_trustworthy}")
    print(f"   Needs Review: {score1.needs_review}")
    
    print(f"\n📊 Factor Breakdown:")
    for name, value in score1.factors.to_dict().items():
        bar = "█" * int(value * 20)
        print(f"   {name:22} [{value:.2f}] {bar}")
    
    print(f"\n💬 Explanation: {score1.explanation}")
    
    if score1.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in score1.recommendations:
            print(f"   • {rec}")
    
    print("\n" + "=" * 70)
    print("SCENARIO 2: MEDIUM CONFIDENCE")
    print("=" * 70)
    
    medium_conf_chunks = [
        RetrievedChunk(
            chunk_id="chunk_004",
            content="Remote work policies vary by department. Some teams allow "
                   "full remote work while others require hybrid arrangements.",
            metadata=DocumentMetadata(
                source="blog/remote_work.md",
                title="Remote Work Blog Post"
            ),
            similarity_score=0.72
        ),
        RetrievedChunk(
            chunk_id="chunk_005",
            content="Working from home options are available. Contact your manager "
                   "for specific arrangements.",
            metadata=DocumentMetadata(
                source="faq/general.md",
                title="General FAQ"
            ),
            similarity_score=0.65
        ),
    ]
    
    query2 = "What is the remote work policy?"
    score2 = scorer.score(query2, medium_conf_chunks)
    
    print(f"\n🔍 Query: '{query2}'")
    print(f"\n📈 Confidence Score: {score2.overall_score:.2f} ({score2.level.value})")
    print(f"   Is Trustworthy: {score2.is_trustworthy}")
    print(f"   Needs Review: {score2.needs_review}")
    
    print(f"\n📊 Factor Breakdown:")
    for name, value in score2.factors.to_dict().items():
        bar = "█" * int(value * 20)
        print(f"   {name:22} [{value:.2f}] {bar}")
    
    print(f"\n💬 Explanation: {score2.explanation}")
    
    if score2.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in score2.recommendations:
            print(f"   • {rec}")
    
    print("\n" + "=" * 70)
    print("SCENARIO 3: LOW CONFIDENCE")
    print("=" * 70)
    
    low_conf_chunks = [
        RetrievedChunk(
            chunk_id="chunk_006",
            content="Our company values include innovation and teamwork.",
            metadata=DocumentMetadata(
                source="about/values.md",
                title="Company Values"
            ),
            similarity_score=0.35
        ),
    ]
    
    query3 = "stock options?"
    score3 = scorer.score(query3, low_conf_chunks)
    
    print(f"\n🔍 Query: '{query3}'")
    print(f"\n📈 Confidence Score: {score3.overall_score:.2f} ({score3.level.value})")
    print(f"   Is Trustworthy: {score3.is_trustworthy}")
    print(f"   Needs Review: {score3.needs_review}")
    
    print(f"\n📊 Factor Breakdown:")
    for name, value in score3.factors.to_dict().items():
        bar = "█" * int(value * 20)
        print(f"   {name:22} [{value:.2f}] {bar}")
    
    print(f"\n💬 Explanation: {score3.explanation}")
    
    if score3.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in score3.recommendations:
            print(f"   • {rec}")
    
    print("\n" + "=" * 70)
    print("SCORED RESPONSE EXAMPLE")
    print("=" * 70)
    
    response = ScoredResponse(
        query=query1,
        answer="Full-time employees receive 20 days of paid vacation per year. "
               "Unused days can be carried over up to 5 days maximum.",
        confidence=score1,
        chunks_used=len(high_conf_chunks)
    )
    
    print(f"\n📋 Scored Response:")
    print(f"   Query: {response.query}")
    print(f"   Answer: {response.answer[:80]}...")
    print(f"   Confidence: {response.confidence.overall_score:.2f} ({response.confidence.level.value})")
    print(f"   Chunks Used: {response.chunks_used}")
    print(f"   Show Warning: {response.should_show_warning()}")
    
    if response.get_warning_message():
        print(f"   Warning: {response.get_warning_message()}")
    
    print("\n" + "=" * 70)
    print("CONFIDENCE LEVELS REFERENCE")
    print("=" * 70)
    
    print("\n📊 Score to Level Mapping:")
    test_scores = [0.95, 0.8, 0.6, 0.4, 0.2, 0.05]
    for score in test_scores:
        level = ConfidenceLevel.from_score(score)
        print(f"   {score:.2f} → {level.value}")


if __name__ == "__main__":
    demonstrate_confidence_scoring()


"""
🧪 TEST & APPLY
===============

How to Test It:
---------------
    python lesson_12_confidence_scoring.py

Expected Output:
----------------
1. Confidence weights breakdown
2. High confidence scenario with detailed scores
3. Medium confidence scenario
4. Low confidence scenario
5. Scored response example
6. Confidence levels reference


⚠️ COMMON STUMBLING BLOCKS
==========================

1. "Weights don't sum to 1.0"
   
   Fix: Use normalize() method:
   weights = ConfidenceWeights(...).normalize()

2. "Confidence always low"
   
   Check:
   - Are retrieval scores low? (improve embeddings)
   - Is source quality low? (add quality source keywords)
   - Is query too vague? (prompt for more details)

3. "Confidence always high"
   
   This might mean thresholds are too lenient.
   
   Fix: Adjust weights or factor calculations:
   weights = ConfidenceWeights(
       retrieval_similarity=0.4,  # Increase importance
       ...
   )

4. "Recommendations not useful"
   
   Customize thresholds in _generate_recommendations()
   for your specific domain.


🎯 KEY TAKEAWAYS
================

1. Multi-Factor Assessment
   - No single factor determines confidence
   - Weighted combination is more robust
   - Different factors for different domains

2. Human-Readable Output
   - Numeric scores for processing
   - Levels for user display
   - Explanations for understanding

3. Actionable Recommendations
   - Don't just report low confidence
   - Suggest improvements
   - Guide knowledge base expansion

4. Calibrated Trust
   - Users know when to verify
   - System knows when to escalate
   - Quality improves over time


📚 NEXT LESSON
==============
In Lesson 13, we'll learn Answer Validation - ensuring our RAG
answers are accurate and well-grounded in the sources!
"""
