# Lesson 12: Relationship Mapping

## A. Concept Overview

### What & Why
**Relationship mapping extracts not just entities, but the connections between them—who works where, who owns what, who said what to whom.** Entities alone are data points; relationships are the graph that connects them into meaningful knowledge. This transforms flat lists into rich knowledge graphs.

### Analogy
Think of relationship mapping like social network analysis:
- **Entities only**: A phone book (names and numbers, no connections)
- **With relationships**: A social network (who knows whom, who works where, who's connected)

When Gemini extracts relationships, it turns "Tim Cook" and "Apple" from isolated names into connected knowledge: "Tim Cook is CEO of Apple."

### Type Safety Benefit
Relationship mapping with Pydantic provides **validated graph structures**:
- Type-safe relationship definitions
- Validated entity references
- Constrained relationship types
- Directional clarity (subject → predicate → object)
- Temporal awareness (past, present, future relationships)
- Provenance tracking (where relationship was found)

---

## B. Key Patterns

**Pattern 1: Triple Pattern (Subject-Predicate-Object)**
```python
class Relationship(BaseModel):
    subject: str  # Entity 1
    predicate: str  # Relationship type
    object: str  # Entity 2
    confidence: float = Field(..., ge=0.0, le=1.0)
```

**Pattern 2: Typed Relationships**
```python
class RelationType(str, Enum):
    WORKS_FOR = "works_for"
    CEO_OF = "ceo_of"
    LOCATED_IN = "located_in"
    ACQUIRED = "acquired"
    PARTNERS_WITH = "partners_with"

class TypedRelationship(BaseModel):
    subject: str
    relation_type: RelationType
    object: str
```

**Pattern 3: Temporal Relationships**
```python
class TemporalRelationship(BaseModel):
    subject: str
    predicate: str
    object: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_current: bool = True
```

**Pattern 4: Rich Relationships**
```python
class DetailedRelationship(BaseModel):
    subject_id: str
    subject_type: EntityType
    predicate: str
    object_id: str
    object_type: EntityType
    context: str  # Sentence where relationship found
    confidence: float
    source_position: Tuple[int, int]
```

---

## C. Common Relationship Types

**Business Relationships:**
- employment: "X works for Y", "X is CEO of Y"
- ownership: "X owns Y", "X acquired Y"
- partnership: "X partners with Y", "X collaborates with Y"
- competition: "X competes with Y"

**Personal Relationships:**
- family: "X is parent of Y", "X is married to Y"
- professional: "X knows Y", "X worked with Y"
- hierarchy: "X reports to Y", "X manages Y"

**Location Relationships:**
- physical: "X is located in Y", "X is headquartered in Y"
- origin: "X is from Y", "X was founded in Y"
- travel: "X visited Y", "X moved to Y"

**Event Relationships:**
- participation: "X attended Y", "X spoke at Y"
- organization: "X hosted Y", "X organized Y"
- timing: "X happened during Y", "X followed Y"

---

## D. Implementation Example

```python
class PersonRelationType(str, Enum):
    WORKS_FOR = "works_for"
    CEO_OF = "ceo_of"
    FOUNDER_OF = "founder_of"
    REPORTS_TO = "reports_to"
    COLLEAGUE_OF = "colleague_of"

class OrgRelationType(str, Enum):
    ACQUIRED = "acquired"
    PARTNERS_WITH = "partners_with"
    COMPETES_WITH = "competes_with"
    SUBSIDIARY_OF = "subsidiary_of"
    INVESTED_IN = "invested_in"

class LocationRelationType(str, Enum):
    HEADQUARTERED_IN = "headquartered_in"
    LOCATED_IN = "located_in"
    OPERATES_IN = "operates_in"

class PersonToOrg(BaseModel):
    person: str
    relation: PersonRelationType
    organization: str

class OrgToOrg(BaseModel):
    subject_org: str
    relation: OrgRelationType
    object_org: str

class OrgToLocation(BaseModel):
    organization: str
    relation: LocationRelationType
    location: str

class RelationshipExtraction(BaseModel):
    """Complete relationship extraction."""
    person_to_org: List[PersonToOrg] = Field(default_factory=list)
    org_to_org: List[OrgToOrg] = Field(default_factory=list)
    org_to_location: List[OrgToLocation] = Field(default_factory=list)
```

---

## E. Validation Patterns

**1. Entity Existence Validation**
```python
@model_validator(mode='after')
def validate_entity_references(self):
    """Ensure relationship entities exist in entity lists."""
    person_names = {p.name for p in self.people}
    org_names = {o.name for o in self.organizations}
    
    for rel in self.relationships:
        if rel.subject not in (person_names | org_names):
            raise ValueError(f"Unknown subject: {rel.subject}")
        if rel.object not in (person_names | org_names):
            raise ValueError(f"Unknown object: {rel.object}")
    
    return self
```

**2. Relationship Type Validation**
```python
@field_validator('relation_type')
@classmethod
def validate_relation_type(cls, v: str, info) -> str:
    """Ensure relationship type is appropriate for entity types."""
    # Person can't "acquire" another person
    # Organization can't "work for" a location
    # Implement domain-specific rules
    return v
```

**3. Cycle Detection**
```python
def detect_cycles(relationships: List[Relationship]) -> List[List[str]]:
    """Detect circular relationships (A → B → C → A)."""
    # Build graph and detect cycles
    # Useful for hierarchies (org charts, etc.)
    pass
```

---

## F. System Prompt for Relationships

```python
system_prompt = """
You are an expert at extracting entities and their relationships from text.

For each relationship, identify:
1. Subject (first entity)
2. Predicate (relationship type)
3. Object (second entity)

Extract the following relationship types:

People-Organization:
- works_for: "Alice works for TechCorp"
- ceo_of: "Tim Cook is CEO of Apple"
- founder_of: "Mark founded Facebook"

Organization-Organization:
- acquired: "Microsoft acquired LinkedIn"
- partners_with: "Google partners with Samsung"
- competes_with: "Apple competes with Samsung"

Organization-Location:
- headquartered_in: "Apple is headquartered in Cupertino"
- operates_in: "Amazon operates in Seattle"

Be precise with relationship direction:
- "Apple acquired Intel division" → subject=Apple, object=Intel
- "Tim Cook leads Apple" → subject=Tim Cook, object=Apple

Extract ALL relationships explicitly stated or strongly implied in the text.
"""
```

---

## G. Common Stumbling Blocks

### Proactive Debugging

**Mistake 1: Bidirectional ambiguity**
```python
# ❌ Unclear direction
Relationship(subject="Apple", predicate="partnership", object="IBM")
# Who initiated? Who benefits more?

# ✅ Clear direction
Relationship(subject="Apple", predicate="announced_partnership_with", object="IBM")
```

**Mistake 2: Overloaded predicates**
```python
# ❌ Too vague
predicate="related_to"  # What kind of relationship?

# ✅ Specific
predicate="acquired"
predicate="invested_in"
predicate="partners_with"
```

**Mistake 3: Missing temporal context**
```python
# "Steve Jobs founded Apple" - past
# "Tim Cook is CEO of Apple" - present
# Different temporal contexts!

# ✅ Include temporal markers
class Relationship(BaseModel):
    subject: str
    predicate: str
    object: str
    is_current: bool
    start_date: Optional[datetime] = None
```

### Type Safety Gotchas

1. **Symmetric vs Asymmetric**: "partners_with" is symmetric, "works_for" is not
2. **Transitive relationships**: If A owns B and B owns C, does A own C?
3. **Relationship strength**: Some connections are stronger than others
4. **Implicit relationships**: "Apple's Tim Cook" implies employment
5. **Contradictions**: Text might contain conflicting relationships

---

## 🎯 Next Steps

Excellent work! You now understand:
- ✅ How to extract typed relationships between entities
- ✅ How to model different relationship patterns
- ✅ How to validate relationship integrity
- ✅ How to handle temporal relationships
- ✅ How to create knowledge graphs from text

In the next lesson, **Handling Missing Data**, we'll learn strategies for dealing with incomplete extractions and partial information.

**Ready for Lesson 13?** 🚀
