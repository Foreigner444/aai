"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                      LESSON 13: ANSWER VALIDATION                            ║
║                                                                              ║
║        Ensuring RAG Answers are Accurate and Grounded in Sources             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 CONCEPT OVERVIEW
==================

What & Why:
-----------
Answer validation ensures that generated answers are actually supported by
the retrieved sources. Even great retrieval + great generation can produce
answers that:

- Include information not in the sources (hallucination)
- Contradict the sources
- Are incomplete or misleading
- Don't actually answer the question

Validation catches these issues before they reach users!

🎯 Real-World Analogy:
----------------------
Think of answer validation like fact-checking in journalism:

Without Validation:
Reporter writes article → Published immediately
(Errors discovered by readers, reputation damaged)

With Validation:
Reporter writes article → Fact-checker verifies claims →
Corrections made → Published with confidence
(Errors caught internally, quality maintained)

Validation is your RAG system's fact-checker!

🔒 Type Safety Benefit:
-----------------------
With Pydantic validation models:
- Validation results are structured and queryable
- All issues are categorized and typed
- Severity levels are consistent
- Fix suggestions are standardized


💻 CODE IMPLEMENTATION
=====================
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, computed_field
from enum import Enum


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class ValidationIssueType(str, Enum):
    """Types of validation issues."""
    HALLUCINATION = "hallucination"
    CONTRADICTION = "contradiction"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    INCOMPLETE_ANSWER = "incomplete_answer"
    OFF_TOPIC = "off_topic"
    OUTDATED_INFO = "outdated_info"
    MISSING_CITATION = "missing_citation"
    AMBIGUOUS = "ambiguous"


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    source: str
    title: Optional[str] = None


class RetrievedChunk(BaseModel):
    """A retrieved chunk."""
    chunk_id: str
    content: str
    metadata: DocumentMetadata
    similarity_score: float = Field(ge=0, le=1)


class ValidationIssue(BaseModel):
    """
    A single validation issue found in an answer.
    """
    
    issue_type: ValidationIssueType = Field(
        description="Type of validation issue"
    )
    
    severity: ValidationSeverity = Field(
        description="Severity of the issue"
    )
    
    description: str = Field(
        description="Human-readable description of the issue"
    )
    
    problematic_text: Optional[str] = Field(
        default=None,
        description="The specific text that has the issue"
    )
    
    suggested_fix: Optional[str] = Field(
        default=None,
        description="Suggested way to fix the issue"
    )
    
    supporting_evidence: Optional[str] = Field(
        default=None,
        description="Evidence from sources (or lack thereof)"
    )


class ValidationResult(BaseModel):
    """
    Complete validation result for an answer.
    """
    
    is_valid: bool = Field(
        description="Whether the answer passed validation"
    )
    
    validation_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall validation score"
    )
    
    issues: list[ValidationIssue] = Field(
        default_factory=list,
        description="List of validation issues found"
    )
    
    grounding_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How well grounded in sources"
    )
    
    completeness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How complete the answer is"
    )
    
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How relevant to the question"
    )
    
    checked_at: datetime = Field(
        default_factory=datetime.now,
        description="When validation was performed"
    )
    
    @computed_field
    @property
    def critical_issues(self) -> list[ValidationIssue]:
        """Get only critical issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]
    
    @computed_field
    @property
    def has_critical_issues(self) -> bool:
        """Check if there are critical issues."""
        return len(self.critical_issues) > 0
    
    @computed_field
    @property
    def issue_count(self) -> int:
        """Total number of issues."""
        return len(self.issues)
    
    def get_issues_by_type(
        self,
        issue_type: ValidationIssueType
    ) -> list[ValidationIssue]:
        """Get issues of a specific type."""
        return [i for i in self.issues if i.issue_type == issue_type]
    
    def get_issues_by_severity(
        self,
        severity: ValidationSeverity
    ) -> list[ValidationIssue]:
        """Get issues of a specific severity."""
        return [i for i in self.issues if i.severity == severity]


class AnswerValidator:
    """
    Validates RAG answers against retrieved sources.
    """
    
    def __init__(
        self,
        strict_mode: bool = False,
        require_citations: bool = True
    ):
        """
        Initialize the validator.
        
        Args:
            strict_mode: If True, any issue fails validation
            require_citations: If True, answers must cite sources
        """
        self.strict_mode = strict_mode
        self.require_citations = require_citations
    
    def _check_grounding(
        self,
        answer: str,
        chunks: list[RetrievedChunk]
    ) -> tuple[float, list[ValidationIssue]]:
        """
        Check if the answer is grounded in the sources.
        Returns grounding score and any issues found.
        """
        issues = []
        
        if not chunks:
            issues.append(ValidationIssue(
                issue_type=ValidationIssueType.UNSUPPORTED_CLAIM,
                severity=ValidationSeverity.CRITICAL,
                description="No source documents provided to validate against",
                suggested_fix="Ensure documents are retrieved before generating answer"
            ))
            return 0.0, issues
        
        answer_lower = answer.lower()
        answer_words = set(answer_lower.split())
        
        source_content = " ".join(c.content.lower() for c in chunks)
        source_words = set(source_content.split())
        
        common_words = answer_words & source_words
        grounding_ratio = len(common_words) / len(answer_words) if answer_words else 0
        
        claim_indicators = [
            "according to", "states that", "mentions",
            "specifically", "exactly", "precisely"
        ]
        
        for indicator in claim_indicators:
            if indicator in answer_lower:
                idx = answer_lower.find(indicator)
                claim_area = answer[idx:idx+100]
                
                claim_words = set(claim_area.lower().split())
                claim_in_source = len(claim_words & source_words) / len(claim_words) if claim_words else 0
                
                if claim_in_source < 0.3:
                    issues.append(ValidationIssue(
                        issue_type=ValidationIssueType.UNSUPPORTED_CLAIM,
                        severity=ValidationSeverity.WARNING,
                        description=f"Claim may not be fully supported by sources",
                        problematic_text=claim_area[:50] + "...",
                        suggested_fix="Verify this claim against the source documents"
                    ))
        
        numbers_in_answer = set()
        for word in answer.split():
            clean = word.strip('.,!?')
            if clean.isdigit():
                numbers_in_answer.add(clean)
        
        numbers_in_sources = set()
        for chunk in chunks:
            for word in chunk.content.split():
                clean = word.strip('.,!?')
                if clean.isdigit():
                    numbers_in_sources.add(clean)
        
        for num in numbers_in_answer:
            if num not in numbers_in_sources and int(num) > 1:
                issues.append(ValidationIssue(
                    issue_type=ValidationIssueType.HALLUCINATION,
                    severity=ValidationSeverity.WARNING,
                    description=f"Number '{num}' not found in source documents",
                    problematic_text=num,
                    suggested_fix="Verify this number exists in the sources"
                ))
                grounding_ratio *= 0.9
        
        return min(grounding_ratio * 1.2, 1.0), issues
    
    def _check_completeness(
        self,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk]
    ) -> tuple[float, list[ValidationIssue]]:
        """
        Check if the answer completely addresses the question.
        """
        issues = []
        
        if len(answer) < 20:
            issues.append(ValidationIssue(
                issue_type=ValidationIssueType.INCOMPLETE_ANSWER,
                severity=ValidationSeverity.WARNING,
                description="Answer is very short and may be incomplete",
                suggested_fix="Provide more detail from available sources"
            ))
            return 0.3, issues
        
        question_words = {"who", "what", "where", "when", "why", "how", "which"}
        query_lower = query.lower()
        
        asked_questions = []
        for word in question_words:
            if word in query_lower:
                asked_questions.append(word)
        
        if "why" in asked_questions and "because" not in answer.lower():
            if "due to" not in answer.lower() and "reason" not in answer.lower():
                issues.append(ValidationIssue(
                    issue_type=ValidationIssueType.INCOMPLETE_ANSWER,
                    severity=ValidationSeverity.INFO,
                    description="'Why' question may not have explanation",
                    suggested_fix="Add reasoning or explanation"
                ))
        
        if "how" in asked_questions:
            step_indicators = ["first", "then", "next", "step", "1.", "2."]
            has_steps = any(ind in answer.lower() for ind in step_indicators)
            if not has_steps and len(answer) < 200:
                issues.append(ValidationIssue(
                    issue_type=ValidationIssueType.INCOMPLETE_ANSWER,
                    severity=ValidationSeverity.INFO,
                    description="'How' question may need more detailed steps",
                    suggested_fix="Consider adding step-by-step instructions"
                ))
        
        completeness = 1.0 - (len(issues) * 0.15)
        return max(completeness, 0.0), issues
    
    def _check_relevance(
        self,
        query: str,
        answer: str
    ) -> tuple[float, list[ValidationIssue]]:
        """
        Check if the answer is relevant to the question.
        """
        issues = []
        
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because",
            "until", "while", "what", "which", "who", "whom", "this",
            "that", "these", "those", "am", "i", "my", "me", "your", "you"
        }
        
        query_keywords = query_words - stop_words
        answer_keywords = answer_words - stop_words
        
        if not query_keywords:
            return 0.8, issues
        
        overlap = len(query_keywords & answer_keywords)
        relevance = min(overlap / len(query_keywords), 1.0)
        
        if relevance < 0.3:
            issues.append(ValidationIssue(
                issue_type=ValidationIssueType.OFF_TOPIC,
                severity=ValidationSeverity.WARNING,
                description="Answer may not directly address the question",
                problematic_text=answer[:100] + "...",
                suggested_fix="Focus response on the specific question asked"
            ))
        
        return max(relevance, 0.2), issues
    
    def _check_citations(
        self,
        answer: str,
        chunks: list[RetrievedChunk]
    ) -> list[ValidationIssue]:
        """
        Check if the answer has proper citations.
        """
        issues = []
        
        if not self.require_citations:
            return issues
        
        citation_patterns = [
            "according to", "source", "document", "[", 
            "reference", "cited", "states"
        ]
        
        has_citation = any(
            pattern in answer.lower() 
            for pattern in citation_patterns
        )
        
        if not has_citation and len(answer) > 100:
            issues.append(ValidationIssue(
                issue_type=ValidationIssueType.MISSING_CITATION,
                severity=ValidationSeverity.INFO,
                description="Answer does not explicitly cite sources",
                suggested_fix="Add source references to support claims"
            ))
        
        return issues
    
    def validate(
        self,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk]
    ) -> ValidationResult:
        """
        Validate an answer against the query and sources.
        """
        all_issues = []
        
        grounding_score, grounding_issues = self._check_grounding(answer, chunks)
        all_issues.extend(grounding_issues)
        
        completeness_score, completeness_issues = self._check_completeness(
            query, answer, chunks
        )
        all_issues.extend(completeness_issues)
        
        relevance_score, relevance_issues = self._check_relevance(query, answer)
        all_issues.extend(relevance_issues)
        
        citation_issues = self._check_citations(answer, chunks)
        all_issues.extend(citation_issues)
        
        validation_score = (
            grounding_score * 0.4 +
            completeness_score * 0.3 +
            relevance_score * 0.3
        )
        
        critical_issues = [
            i for i in all_issues 
            if i.severity == ValidationSeverity.CRITICAL
        ]
        
        if self.strict_mode:
            is_valid = len(all_issues) == 0
        else:
            is_valid = len(critical_issues) == 0 and validation_score >= 0.5
        
        return ValidationResult(
            is_valid=is_valid,
            validation_score=validation_score,
            issues=all_issues,
            grounding_score=grounding_score,
            completeness_score=completeness_score,
            relevance_score=relevance_score
        )


class ValidatedAnswer(BaseModel):
    """
    An answer that has been validated.
    """
    
    query: str = Field(description="Original question")
    answer: str = Field(description="Generated answer")
    validation: ValidationResult = Field(description="Validation results")
    chunks_used: list[RetrievedChunk] = Field(description="Source chunks")
    
    def get_display_answer(self) -> str:
        """Get answer with any necessary warnings."""
        if self.validation.is_valid:
            return self.answer
        
        warning = "\n\n⚠️ Note: This answer has validation concerns:\n"
        for issue in self.validation.issues[:3]:
            warning += f"- {issue.description}\n"
        
        return self.answer + warning


def demonstrate_answer_validation():
    """
    Demonstrate answer validation.
    """
    
    print("=" * 70)
    print("ANSWER VALIDATION DEMONSTRATION")
    print("=" * 70)
    
    validator = AnswerValidator(strict_mode=False, require_citations=True)
    
    print("\n" + "=" * 70)
    print("SCENARIO 1: WELL-GROUNDED ANSWER")
    print("=" * 70)
    
    chunks1 = [
        RetrievedChunk(
            chunk_id="chunk_001",
            content="Full-time employees receive 20 days of paid vacation per year. "
                   "Vacation requests must be submitted at least two weeks in advance "
                   "through the HR portal.",
            metadata=DocumentMetadata(
                source="hr/vacation_policy.md",
                title="Vacation Policy"
            ),
            similarity_score=0.95
        ),
        RetrievedChunk(
            chunk_id="chunk_002",
            content="Unused vacation days can be carried over to the next year, "
                   "up to a maximum of 5 days. Days beyond this limit are forfeited.",
            metadata=DocumentMetadata(
                source="hr/vacation_policy.md",
                title="Vacation Policy"
            ),
            similarity_score=0.88
        ),
    ]
    
    query1 = "How many vacation days do employees get?"
    answer1 = ("According to our vacation policy, full-time employees receive "
               "20 days of paid vacation per year. Vacation requests should be "
               "submitted at least two weeks in advance through the HR portal. "
               "You can carry over up to 5 unused days to the next year.")
    
    result1 = validator.validate(query1, answer1, chunks1)
    
    print(f"\n🔍 Query: '{query1}'")
    print(f"\n📝 Answer: {answer1[:100]}...")
    
    print(f"\n✅ Validation Result:")
    print(f"   Is Valid: {result1.is_valid}")
    print(f"   Validation Score: {result1.validation_score:.2f}")
    print(f"   Grounding Score: {result1.grounding_score:.2f}")
    print(f"   Completeness Score: {result1.completeness_score:.2f}")
    print(f"   Relevance Score: {result1.relevance_score:.2f}")
    print(f"   Issues Found: {result1.issue_count}")
    
    if result1.issues:
        print(f"\n⚠️  Issues:")
        for issue in result1.issues:
            print(f"   [{issue.severity.value}] {issue.issue_type.value}")
            print(f"       {issue.description}")
    
    print("\n" + "=" * 70)
    print("SCENARIO 2: ANSWER WITH HALLUCINATION")
    print("=" * 70)
    
    chunks2 = chunks1.copy()
    
    query2 = "What is the vacation policy?"
    answer2 = ("Employees receive 30 days of vacation per year. "
               "Additionally, you get 15 floating holidays and "
               "unlimited sick days. The policy was updated in 2025.")
    
    result2 = validator.validate(query2, answer2, chunks2)
    
    print(f"\n🔍 Query: '{query2}'")
    print(f"\n📝 Answer: {answer2}")
    
    print(f"\n❌ Validation Result:")
    print(f"   Is Valid: {result2.is_valid}")
    print(f"   Validation Score: {result2.validation_score:.2f}")
    print(f"   Grounding Score: {result2.grounding_score:.2f}")
    print(f"   Issues Found: {result2.issue_count}")
    
    print(f"\n⚠️  Issues:")
    for issue in result2.issues:
        print(f"\n   [{issue.severity.value}] {issue.issue_type.value}")
        print(f"   Description: {issue.description}")
        if issue.problematic_text:
            print(f"   Problematic: '{issue.problematic_text}'")
        if issue.suggested_fix:
            print(f"   Fix: {issue.suggested_fix}")
    
    print("\n" + "=" * 70)
    print("SCENARIO 3: OFF-TOPIC ANSWER")
    print("=" * 70)
    
    query3 = "What is the password reset process?"
    answer3 = ("Our company values include innovation, teamwork, and customer focus. "
               "We strive to create a positive work environment for all employees.")
    
    result3 = validator.validate(query3, answer3, chunks1)
    
    print(f"\n🔍 Query: '{query3}'")
    print(f"\n📝 Answer: {answer3}")
    
    print(f"\n❌ Validation Result:")
    print(f"   Is Valid: {result3.is_valid}")
    print(f"   Validation Score: {result3.validation_score:.2f}")
    print(f"   Relevance Score: {result3.relevance_score:.2f}")
    
    print(f"\n⚠️  Issues:")
    for issue in result3.issues:
        print(f"   [{issue.severity.value}] {issue.description}")
    
    print("\n" + "=" * 70)
    print("SCENARIO 4: INCOMPLETE ANSWER")
    print("=" * 70)
    
    query4 = "How do I request vacation time and what is the approval process?"
    answer4 = "Use the HR portal."
    
    result4 = validator.validate(query4, answer4, chunks1)
    
    print(f"\n🔍 Query: '{query4}'")
    print(f"\n📝 Answer: {answer4}")
    
    print(f"\n⚠️  Validation Result:")
    print(f"   Is Valid: {result4.is_valid}")
    print(f"   Completeness Score: {result4.completeness_score:.2f}")
    
    print(f"\n⚠️  Issues:")
    for issue in result4.issues:
        print(f"   [{issue.severity.value}] {issue.description}")
        if issue.suggested_fix:
            print(f"   Fix: {issue.suggested_fix}")
    
    print("\n" + "=" * 70)
    print("VALIDATED ANSWER DISPLAY")
    print("=" * 70)
    
    validated = ValidatedAnswer(
        query=query2,
        answer=answer2,
        validation=result2,
        chunks_used=chunks2
    )
    
    print(f"\n📋 Display Answer (with warnings if needed):")
    print("-" * 50)
    print(validated.get_display_answer())
    
    print("\n" + "=" * 70)
    print("STRICT MODE COMPARISON")
    print("=" * 70)
    
    strict_validator = AnswerValidator(strict_mode=True)
    normal_validator = AnswerValidator(strict_mode=False)
    
    result_strict = strict_validator.validate(query1, answer1, chunks1)
    result_normal = normal_validator.validate(query1, answer1, chunks1)
    
    print(f"\n📊 Same answer, different modes:")
    print(f"   Strict mode: is_valid = {result_strict.is_valid}")
    print(f"   Normal mode: is_valid = {result_normal.is_valid}")


if __name__ == "__main__":
    demonstrate_answer_validation()


"""
🧪 TEST & APPLY
===============

How to Test It:
---------------
    python lesson_13_answer_validation.py

Expected Output:
----------------
1. Well-grounded answer validation
2. Hallucination detection
3. Off-topic answer detection
4. Incomplete answer detection
5. Validated answer display with warnings
6. Strict vs normal mode comparison


⚠️ COMMON STUMBLING BLOCKS
==========================

1. "Everything fails validation"
   
   You might be in strict_mode:
   validator = AnswerValidator(strict_mode=False)

2. "Hallucinations not detected"
   
   Basic validation uses word overlap. For better detection:
   - Use Gemini to cross-check claims
   - Implement semantic similarity checking
   - Add domain-specific validators

3. "False positives for unsupported claims"
   
   Some answers paraphrase sources.
   
   Fix: Increase grounding threshold or use semantic matching:
   # Consider embeddings similarity instead of word overlap

4. "Citations always flagged as missing"
   
   Disable if not needed:
   validator = AnswerValidator(require_citations=False)


🎯 KEY TAKEAWAYS
================

1. Multi-Dimensional Validation
   - Grounding: Is it supported by sources?
   - Completeness: Does it answer fully?
   - Relevance: Does it address the question?

2. Severity Levels Matter
   - Critical: Must fix before showing
   - Warning: May need attention
   - Info: Nice to know

3. Actionable Issues
   - Clear descriptions
   - Specific problematic text
   - Suggested fixes

4. Configurable Strictness
   - Strict mode for high-stakes
   - Normal mode for general use


📚 NEXT LESSON
==============
In Lesson 14, we'll learn Fact-Checking Patterns - advanced techniques
for verifying factual accuracy in RAG answers!
"""
