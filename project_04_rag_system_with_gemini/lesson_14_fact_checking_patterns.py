"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    LESSON 14: FACT-CHECKING PATTERNS                         ║
║                                                                              ║
║         Advanced Techniques for Verifying Factual Accuracy in RAG            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 CONCEPT OVERVIEW
==================

What & Why:
-----------
Fact-checking goes beyond basic validation to actively verify specific claims
in RAG answers. This is essential for high-stakes applications where accuracy
matters: legal, medical, financial, or any domain where wrong information
can cause harm.

Key fact-checking patterns:
- Claim extraction and verification
- Cross-reference checking
- Temporal consistency validation
- Numerical accuracy verification

🎯 Real-World Analogy:
----------------------
Think of fact-checking like quality control in manufacturing:

Basic Validation: "Does this look like a car?"
Fact-Checking: 
- "Are all 4 wheels attached correctly?"
- "Does the engine match the specifications?"
- "Are all safety features functional?"
- "Does the serial number match the records?"

Fact-checking verifies SPECIFIC claims, not just overall quality!

🔒 Type Safety Benefit:
-----------------------
With Pydantic fact-checking models:
- Claims are structured and traceable
- Verification results are typed
- Evidence chains are documented
- All verdicts are validated


💻 CODE IMPLEMENTATION
=====================
"""

import re
from typing import Optional
from datetime import datetime, date
from pydantic import BaseModel, Field, computed_field
from enum import Enum


class ClaimType(str, Enum):
    """Types of claims that can be fact-checked."""
    NUMERICAL = "numerical"
    TEMPORAL = "temporal"
    CATEGORICAL = "categorical"
    PROCEDURAL = "procedural"
    DEFINITIONAL = "definitional"
    COMPARATIVE = "comparative"
    EXISTENTIAL = "existential"


class VerificationStatus(str, Enum):
    """Status of claim verification."""
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"
    PARTIALLY_VERIFIED = "partially_verified"
    NEEDS_CONTEXT = "needs_context"


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


class ExtractedClaim(BaseModel):
    """
    A claim extracted from an answer for fact-checking.
    """
    
    claim_id: str = Field(description="Unique identifier")
    claim_text: str = Field(description="The claim statement")
    claim_type: ClaimType = Field(description="Type of claim")
    
    extracted_value: Optional[str] = Field(
        default=None,
        description="Specific value in the claim (number, date, etc.)"
    )
    
    context: str = Field(
        default="",
        description="Surrounding context from the answer"
    )
    
    importance: str = Field(
        default="medium",
        description="How important this claim is: high, medium, low"
    )


class Evidence(BaseModel):
    """
    Evidence supporting or contradicting a claim.
    """
    
    source_chunk_id: str = Field(description="ID of source chunk")
    source_text: str = Field(description="Relevant text from source")
    source_path: str = Field(description="Path to source document")
    
    relevance_score: float = Field(
        ge=0,
        le=1,
        description="How relevant this evidence is"
    )
    
    supports_claim: bool = Field(
        description="Whether this evidence supports the claim"
    )
    
    extracted_value: Optional[str] = Field(
        default=None,
        description="Value found in the source (for comparison)"
    )


class ClaimVerification(BaseModel):
    """
    Result of verifying a single claim.
    """
    
    claim: ExtractedClaim = Field(description="The claim being verified")
    
    status: VerificationStatus = Field(
        description="Verification status"
    )
    
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence in the verification"
    )
    
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Evidence found"
    )
    
    explanation: str = Field(
        description="Human-readable explanation"
    )
    
    discrepancy: Optional[str] = Field(
        default=None,
        description="If contradicted, what the discrepancy is"
    )
    
    @computed_field
    @property
    def is_verified(self) -> bool:
        """Check if claim is verified."""
        return self.status == VerificationStatus.VERIFIED
    
    @computed_field
    @property
    def has_evidence(self) -> bool:
        """Check if evidence was found."""
        return len(self.evidence) > 0


class FactCheckResult(BaseModel):
    """
    Complete fact-check result for an answer.
    """
    
    answer_text: str = Field(description="The answer that was checked")
    
    claims_extracted: list[ExtractedClaim] = Field(
        description="All claims extracted"
    )
    
    verifications: list[ClaimVerification] = Field(
        description="Verification results"
    )
    
    overall_accuracy: float = Field(
        ge=0,
        le=1,
        description="Overall factual accuracy score"
    )
    
    checked_at: datetime = Field(
        default_factory=datetime.now,
        description="When fact-check was performed"
    )
    
    @computed_field
    @property
    def total_claims(self) -> int:
        return len(self.claims_extracted)
    
    @computed_field
    @property
    def verified_count(self) -> int:
        return sum(1 for v in self.verifications if v.is_verified)
    
    @computed_field
    @property
    def contradicted_count(self) -> int:
        return sum(
            1 for v in self.verifications 
            if v.status == VerificationStatus.CONTRADICTED
        )
    
    def get_contradictions(self) -> list[ClaimVerification]:
        """Get all contradicted claims."""
        return [
            v for v in self.verifications
            if v.status == VerificationStatus.CONTRADICTED
        ]
    
    def get_unverifiable(self) -> list[ClaimVerification]:
        """Get all unverifiable claims."""
        return [
            v for v in self.verifications
            if v.status == VerificationStatus.UNVERIFIABLE
        ]


class ClaimExtractor:
    """
    Extracts verifiable claims from text.
    """
    
    def __init__(self):
        self._claim_counter = 0
    
    def _generate_claim_id(self) -> str:
        """Generate unique claim ID."""
        self._claim_counter += 1
        return f"claim_{self._claim_counter:04d}"
    
    def extract_numerical_claims(self, text: str) -> list[ExtractedClaim]:
        """Extract claims containing numbers."""
        claims = []
        
        patterns = [
            r'(\d+)\s*days?\s+(?:of\s+)?(?:paid\s+)?(?:vacation|leave|pto)',
            r'(?:receive|get|entitled to)\s+(\d+)\s*(?:days?|hours?)',
            r'(\d+)\s*(?:percent|%)',
            r'maximum\s+(?:of\s+)?(\d+)',
            r'at least\s+(\d+)',
            r'up to\s+(\d+)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                full_match = match.group(0)
                number = match.group(1)
                
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                claims.append(ExtractedClaim(
                    claim_id=self._generate_claim_id(),
                    claim_text=full_match,
                    claim_type=ClaimType.NUMERICAL,
                    extracted_value=number,
                    context=context,
                    importance="high"
                ))
        
        return claims
    
    def extract_temporal_claims(self, text: str) -> list[ExtractedClaim]:
        """Extract claims about time/dates."""
        claims = []
        
        patterns = [
            r'(?:at least|within)\s+(\d+)\s*(?:days?|weeks?|hours?)\s+(?:in advance|before|notice)',
            r'(?:updated|effective|valid)\s+(?:on|from|since|until)\s+([A-Za-z]+\s+\d{1,2},?\s*\d{4})',
            r'(?:every|each)\s+(year|month|week|day)',
            r'(\d+)\s*(?:weeks?|days?|hours?)\s+(?:advance\s+)?notice',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                claims.append(ExtractedClaim(
                    claim_id=self._generate_claim_id(),
                    claim_text=match.group(0),
                    claim_type=ClaimType.TEMPORAL,
                    extracted_value=match.group(1) if match.groups() else None,
                    context=text[max(0, match.start()-30):min(len(text), match.end()+30)],
                    importance="medium"
                ))
        
        return claims
    
    def extract_procedural_claims(self, text: str) -> list[ExtractedClaim]:
        """Extract claims about procedures/processes."""
        claims = []
        
        procedure_indicators = [
            r'(?:must|should|need to)\s+([^.]+)',
            r'(?:through|via|using)\s+(?:the\s+)?([A-Za-z\s]+(?:portal|system|form))',
            r'(?:submit|request|apply)\s+(?:through|via|to)\s+([^.]+)',
        ]
        
        for pattern in procedure_indicators:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                claims.append(ExtractedClaim(
                    claim_id=self._generate_claim_id(),
                    claim_text=match.group(0)[:100],
                    claim_type=ClaimType.PROCEDURAL,
                    context=text[max(0, match.start()-20):min(len(text), match.end()+20)],
                    importance="medium"
                ))
        
        return claims
    
    def extract_all_claims(self, text: str) -> list[ExtractedClaim]:
        """Extract all types of claims from text."""
        self._claim_counter = 0
        
        all_claims = []
        all_claims.extend(self.extract_numerical_claims(text))
        all_claims.extend(self.extract_temporal_claims(text))
        all_claims.extend(self.extract_procedural_claims(text))
        
        return all_claims


class FactChecker:
    """
    Verifies extracted claims against source documents.
    """
    
    def __init__(self):
        self.extractor = ClaimExtractor()
    
    def _find_evidence(
        self,
        claim: ExtractedClaim,
        chunks: list[RetrievedChunk]
    ) -> list[Evidence]:
        """Find evidence for a claim in source chunks."""
        evidence_list = []
        
        for chunk in chunks:
            chunk_lower = chunk.content.lower()
            claim_lower = claim.claim_text.lower()
            
            relevance = 0.0
            if claim.extracted_value and claim.extracted_value in chunk.content:
                relevance = 0.9
            elif any(word in chunk_lower for word in claim_lower.split() if len(word) > 3):
                relevance = 0.5
            
            if relevance > 0:
                source_value = None
                if claim.claim_type == ClaimType.NUMERICAL:
                    numbers = re.findall(r'\d+', chunk.content)
                    if numbers:
                        source_value = numbers[0]
                
                supports = False
                if claim.extracted_value and source_value:
                    supports = claim.extracted_value == source_value
                elif relevance >= 0.7:
                    supports = True
                
                evidence_list.append(Evidence(
                    source_chunk_id=chunk.chunk_id,
                    source_text=chunk.content[:200],
                    source_path=chunk.metadata.source,
                    relevance_score=relevance,
                    supports_claim=supports,
                    extracted_value=source_value
                ))
        
        return sorted(evidence_list, key=lambda e: e.relevance_score, reverse=True)
    
    def _verify_numerical_claim(
        self,
        claim: ExtractedClaim,
        evidence: list[Evidence]
    ) -> ClaimVerification:
        """Verify a numerical claim."""
        if not evidence:
            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.3,
                evidence=[],
                explanation="No evidence found in source documents"
            )
        
        supporting = [e for e in evidence if e.supports_claim]
        contradicting = [e for e in evidence if not e.supports_claim and e.extracted_value]
        
        if supporting:
            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.VERIFIED,
                confidence=max(e.relevance_score for e in supporting),
                evidence=evidence[:3],
                explanation=f"Claim verified: '{claim.extracted_value}' found in sources"
            )
        
        if contradicting:
            source_values = set(e.extracted_value for e in contradicting if e.extracted_value)
            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.CONTRADICTED,
                confidence=0.8,
                evidence=evidence[:3],
                explanation=f"Claim contradicted by sources",
                discrepancy=f"Answer says '{claim.extracted_value}', sources say '{source_values}'"
            )
        
        return ClaimVerification(
            claim=claim,
            status=VerificationStatus.PARTIALLY_VERIFIED,
            confidence=0.5,
            evidence=evidence[:3],
            explanation="Related information found but exact value not confirmed"
        )
    
    def _verify_claim(
        self,
        claim: ExtractedClaim,
        chunks: list[RetrievedChunk]
    ) -> ClaimVerification:
        """Verify a single claim."""
        evidence = self._find_evidence(claim, chunks)
        
        if claim.claim_type == ClaimType.NUMERICAL:
            return self._verify_numerical_claim(claim, evidence)
        
        if evidence:
            best_evidence = evidence[0]
            if best_evidence.relevance_score >= 0.7:
                return ClaimVerification(
                    claim=claim,
                    status=VerificationStatus.VERIFIED,
                    confidence=best_evidence.relevance_score,
                    evidence=evidence[:3],
                    explanation="Claim supported by source documents"
                )
            else:
                return ClaimVerification(
                    claim=claim,
                    status=VerificationStatus.PARTIALLY_VERIFIED,
                    confidence=best_evidence.relevance_score,
                    evidence=evidence[:3],
                    explanation="Some related information found in sources"
                )
        
        return ClaimVerification(
            claim=claim,
            status=VerificationStatus.UNVERIFIABLE,
            confidence=0.2,
            evidence=[],
            explanation="No evidence found for this claim"
        )
    
    def fact_check(
        self,
        answer: str,
        chunks: list[RetrievedChunk]
    ) -> FactCheckResult:
        """
        Perform complete fact-check on an answer.
        """
        claims = self.extractor.extract_all_claims(answer)
        
        verifications = []
        for claim in claims:
            verification = self._verify_claim(claim, chunks)
            verifications.append(verification)
        
        if verifications:
            verified = sum(1 for v in verifications if v.is_verified)
            accuracy = verified / len(verifications)
        else:
            accuracy = 1.0
        
        return FactCheckResult(
            answer_text=answer,
            claims_extracted=claims,
            verifications=verifications,
            overall_accuracy=accuracy
        )


def demonstrate_fact_checking():
    """
    Demonstrate fact-checking patterns.
    """
    
    print("=" * 70)
    print("FACT-CHECKING PATTERNS DEMONSTRATION")
    print("=" * 70)
    
    fact_checker = FactChecker()
    
    source_chunks = [
        RetrievedChunk(
            chunk_id="chunk_001",
            content="Full-time employees receive 20 days of paid vacation per year. "
                   "Vacation requests must be submitted at least 2 weeks in advance "
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
                   "with a maximum of 5 days. Days exceeding this limit will be "
                   "forfeited on January 1st.",
            metadata=DocumentMetadata(
                source="hr/vacation_policy.md",
                title="Vacation Policy"
            ),
            similarity_score=0.88
        ),
        RetrievedChunk(
            chunk_id="chunk_003",
            content="Part-time employees receive prorated vacation based on their "
                   "scheduled hours. A 20-hour/week employee would receive 10 days.",
            metadata=DocumentMetadata(
                source="hr/vacation_policy.md",
                title="Vacation Policy"
            ),
            similarity_score=0.75
        ),
    ]
    
    print("\n" + "=" * 70)
    print("SCENARIO 1: ACCURATE ANSWER")
    print("=" * 70)
    
    accurate_answer = (
        "Full-time employees receive 20 days of paid vacation per year. "
        "You must submit vacation requests at least 2 weeks in advance "
        "through the HR portal. Up to 5 days can be carried over to the next year."
    )
    
    result1 = fact_checker.fact_check(accurate_answer, source_chunks)
    
    print(f"\n📝 Answer: {accurate_answer[:100]}...")
    
    print(f"\n📊 Fact-Check Results:")
    print(f"   Total Claims: {result1.total_claims}")
    print(f"   Verified: {result1.verified_count}")
    print(f"   Contradicted: {result1.contradicted_count}")
    print(f"   Overall Accuracy: {result1.overall_accuracy:.0%}")
    
    print(f"\n🔍 Claims Extracted:")
    for claim in result1.claims_extracted:
        print(f"\n   [{claim.claim_type.value}] {claim.claim_text}")
        if claim.extracted_value:
            print(f"   Value: {claim.extracted_value}")
    
    print(f"\n✅ Verifications:")
    for v in result1.verifications:
        status_icon = "✅" if v.is_verified else "⚠️"
        print(f"\n   {status_icon} {v.claim.claim_text}")
        print(f"   Status: {v.status.value} (confidence: {v.confidence:.0%})")
        print(f"   {v.explanation}")
    
    print("\n" + "=" * 70)
    print("SCENARIO 2: ANSWER WITH ERRORS")
    print("=" * 70)
    
    inaccurate_answer = (
        "Full-time employees receive 30 days of paid vacation per year. "
        "Requests should be submitted 1 week in advance through the HR portal. "
        "Up to 10 days can be carried over annually."
    )
    
    result2 = fact_checker.fact_check(inaccurate_answer, source_chunks)
    
    print(f"\n📝 Answer: {inaccurate_answer}")
    
    print(f"\n📊 Fact-Check Results:")
    print(f"   Total Claims: {result2.total_claims}")
    print(f"   Verified: {result2.verified_count}")
    print(f"   Contradicted: {result2.contradicted_count}")
    print(f"   Overall Accuracy: {result2.overall_accuracy:.0%}")
    
    contradictions = result2.get_contradictions()
    if contradictions:
        print(f"\n❌ Contradictions Found:")
        for v in contradictions:
            print(f"\n   Claim: {v.claim.claim_text}")
            if v.discrepancy:
                print(f"   ⚠️  {v.discrepancy}")
            print(f"   Confidence: {v.confidence:.0%}")
    
    print("\n" + "=" * 70)
    print("SCENARIO 3: UNVERIFIABLE CLAIMS")
    print("=" * 70)
    
    unverifiable_answer = (
        "According to the 2024 update, employees now get unlimited PTO. "
        "The policy was changed effective January 1st, 2025."
    )
    
    result3 = fact_checker.fact_check(unverifiable_answer, source_chunks)
    
    print(f"\n📝 Answer: {unverifiable_answer}")
    
    print(f"\n📊 Fact-Check Results:")
    print(f"   Total Claims: {result3.total_claims}")
    print(f"   Unverifiable: {len(result3.get_unverifiable())}")
    print(f"   Overall Accuracy: {result3.overall_accuracy:.0%}")
    
    unverifiable = result3.get_unverifiable()
    if unverifiable:
        print(f"\n❓ Unverifiable Claims:")
        for v in unverifiable:
            print(f"\n   Claim: {v.claim.claim_text}")
            print(f"   {v.explanation}")
    
    print("\n" + "=" * 70)
    print("CLAIM EXTRACTION DETAILS")
    print("=" * 70)
    
    extractor = ClaimExtractor()
    
    test_text = (
        "Employees receive 20 days of vacation. "
        "Requests must be submitted 2 weeks in advance. "
        "Maximum carryover is 5 days per year."
    )
    
    print(f"\n📝 Test Text: {test_text}")
    
    numerical = extractor.extract_numerical_claims(test_text)
    temporal = extractor.extract_temporal_claims(test_text)
    procedural = extractor.extract_procedural_claims(test_text)
    
    print(f"\n🔢 Numerical Claims ({len(numerical)}):")
    for claim in numerical:
        print(f"   '{claim.claim_text}' → value: {claim.extracted_value}")
    
    print(f"\n📅 Temporal Claims ({len(temporal)}):")
    for claim in temporal:
        print(f"   '{claim.claim_text}'")
    
    print(f"\n📋 Procedural Claims ({len(procedural)}):")
    for claim in procedural:
        print(f"   '{claim.claim_text[:50]}...'")


if __name__ == "__main__":
    demonstrate_fact_checking()


"""
🧪 TEST & APPLY
===============

How to Test It:
---------------
    python lesson_14_fact_checking_patterns.py

Expected Output:
----------------
1. Accurate answer fact-check with all claims verified
2. Inaccurate answer with contradictions identified
3. Unverifiable claims detection
4. Claim extraction details by type


⚠️ COMMON STUMBLING BLOCKS
==========================

1. "No claims extracted"
   
   The patterns might not match your text format.
   
   Fix: Add custom patterns for your domain:
   patterns = [r'your_domain_pattern']

2. "All claims unverifiable"
   
   Sources might not contain the exact information.
   
   Fix: 
   - Improve retrieval to get more relevant sources
   - Lower relevance threshold in evidence finding

3. "False contradictions"
   
   Number extraction might grab wrong numbers.
   
   Fix: Use more specific patterns with context:
   r'(\d+)\s*days\s+of\s+vacation'  # More specific

4. "Missing important claims"
   
   Current extraction is rule-based.
   
   Fix: For production, use Gemini to extract claims:
   - "Extract all factual claims from this text"


🎯 KEY TAKEAWAYS
================

1. Claim Extraction First
   - Break answer into verifiable claims
   - Categorize by type
   - Identify key values

2. Evidence-Based Verification
   - Find supporting/contradicting evidence
   - Calculate confidence from evidence quality
   - Document the evidence chain

3. Type-Specific Verification
   - Numbers: Exact match comparison
   - Temporal: Date/time validation
   - Procedural: Process consistency

4. Clear Discrepancy Reporting
   - Show what answer says
   - Show what sources say
   - Explain the difference


📚 NEXT LESSON
==============
In Lesson 15, we'll learn Streaming RAG Responses - delivering
real-time answers with progressive context!
"""
