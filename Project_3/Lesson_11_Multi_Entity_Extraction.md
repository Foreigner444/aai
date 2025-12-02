# Lesson 11: Multi-Entity Extraction

## A. Concept Overview

### What & Why
**Multi-entity extraction involves extracting multiple entities of different types from a single text in one pass.** Instead of running separate extractions for people, companies, and locations, you extract all of them simultaneously with relationships and context preserved. This is more efficient and captures how entities relate to each other in the source text.

### Analogy
Think of multi-entity extraction like a crime scene investigation:
- **Single-entity extraction**: Interview each witness separately about one aspect (who was there? what happened? when?)
- **Multi-entity extraction**: Document the entire scene at once, capturing all people, objects, actions, and their relationships in context

When Gemini extracts multiple entity types together, it understands the relationships—"Tim Cook" (person) is CEO of "Apple" (organization), planning to visit "New York" (location).

### Type Safety Benefit
Multi-entity extraction with Pydantic provides **relational type safety**:
- Extract multiple types in one model
- Validate all entities and their relationships
- Preserve entity connections—who works where, who said what
- Atomic validation—all or nothing, ensuring consistency
- Richer context—entities aren't isolated
- Single API call—more efficient than multiple extractions

---

## B. Key Patterns

**Pattern 1: Flat Multi-Entity**
```python
class MultiEntityExtraction(BaseModel):
    people: List[Person]
    organizations: List[Organization]
    locations: List[Location]
    # Separate lists, no explicit relationships
```

**Pattern 2: Entity with Context**
```python
class EntityMention(BaseModel):
    text: str
    entity_type: EntityType
    context: str  # Surrounding text
    position: Tuple[int, int]  # Character positions
```

**Pattern 3: Relational Extraction**
```python
class Relationship(BaseModel):
    subject: str  # "Tim Cook"
    predicate: str  # "is CEO of"
    object: str  # "Apple Inc."

class RelationalExtraction(BaseModel):
    entities: Dict[str, Entity]
    relationships: List[Relationship]
```

**Pattern 4: Event-Centric**
```python
class Event(BaseModel):
    event_type: str
    participants: List[str]  # People involved
    organizations: List[str]  # Organizations involved
    location: Optional[str]
    date: Optional[datetime]
    description: str
```

### Real-World Example

**Input text:**
```
Apple Inc. announced that Tim Cook will visit their New York office
next Tuesday to meet with Microsoft executives, including Satya Nadella.
The companies will discuss a potential $5 billion partnership.
```

**Extracted structure:**
```python
extraction = MultiEntityExtraction(
    people=[
        Person(name="Tim Cook", company="Apple Inc."),
        Person(name="Satya Nadella", company="Microsoft")
    ],
    organizations=[
        Organization(name="Apple Inc."),
        Organization(name="Microsoft")
    ],
    locations=[
        Location(name="New York")
    ],
    amounts=[
        Amount(value=5000000000, currency="USD", context="partnership")
    ],
    relationships=[
        Relationship(subject="Tim Cook", predicate="will visit", object="New York"),
        Relationship(subject="Tim Cook", predicate="will meet with", object="Satya Nadella"),
        Relationship(subject="Apple Inc.", predicate="partnership with", object="Microsoft")
    ]
)
```

---

## C. Implementation Strategy

### Start Simple, Add Complexity

**Level 1: Basic Multi-Entity**
```python
class BasicExtraction(BaseModel):
    people: List[str] = Field(default_factory=list)
    companies: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
```

**Level 2: Structured Entities**
```python
class DetailedPerson(BaseModel):
    name: str
    role: Optional[str] = None
    company: Optional[str] = None

class IntermediateExtraction(BaseModel):
    people: List[DetailedPerson]
    companies: List[str]
    locations: List[str]
```

**Level 3: Full Relationships**
```python
class AdvancedExtraction(BaseModel):
    people: List[DetailedPerson]
    companies: List[DetailedCompany]
    locations: List[DetailedLocation]
    relationships: List[Relationship]
    events: List[Event]
```

### System Prompts for Multi-Entity

```python
system_prompt = """
You are an expert at extracting multiple types of entities from text.

Extract ALL of the following:
1. People: names, roles, affiliations
2. Organizations: names, types, locations
3. Locations: cities, countries, addresses
4. Monetary amounts: values, currencies, context
5. Dates and times: when events happen

Also identify relationships between entities:
- Who works for which organization
- Who is meeting with whom
- Which organizations are partners
- Where events are taking place

Be comprehensive but accurate. If information is uncertain, omit it.
"""
```

---

## D. Best Practices

**1. Design for Partial Success**
```python
# All entity lists optional/default to empty
class RobustExtraction(BaseModel):
    people: List[Person] = Field(default_factory=list)
    companies: List[Organization] = Field(default_factory=list)
    # Even if no companies found, extraction succeeds
```

**2. Include Confidence Scores**
```python
class EntityWithConfidence(BaseModel):
    text: str
    entity_type: EntityType
    confidence: float = Field(..., ge=0.0, le=1.0)

class ConfidenceAwareExtraction(BaseModel):
    entities: List[EntityWithConfidence]
    # Can filter low-confidence entities later
```

**3. Preserve Source Context**
```python
class EntityInContext(BaseModel):
    entity: Entity
    surrounding_text: str  # Context for verification
    sentence: str  # Sentence where entity appears
```

**4. Handle Entity Deduplication**
```python
# Gemini might extract "Apple" and "Apple Inc." as separate entities
# Use validation to deduplicate:

@model_validator(mode='after')
def deduplicate_organizations(self):
    # Implement deduplication logic
    unique = {}
    for org in self.organizations:
        normalized = org.name.lower().strip()
        if normalized not in unique:
            unique[normalized] = org
    self.organizations = list(unique.values())
    return self
```

---

## E. Common Stumbling Blocks

### Proactive Debugging

**Mistake 1: Requiring all entity types**
```python
# ❌ Forces finding something even if not present
class Extraction(BaseModel):
    people: List[Person] = Field(..., min_length=1)  # What if no people?

# ✅ All entity types optional
class Extraction(BaseModel):
    people: List[Person] = Field(default_factory=list)
```

**Mistake 2: Ignoring entity coreference**
```python
# Text: "Apple announced... The company said..."
# "Apple" and "The company" refer to same entity

# Need to handle coreference in prompt or post-processing
```

**Mistake 3: Not validating relationships**
```python
# ❌ Relationship references non-existent entity
relationships = [
    Relationship(subject="Tim Cook", object="Apple")
]
# But "Apple" not in organizations list

# ✅ Validate relationship entities exist
@model_validator(mode='after')
def validate_relationships(self):
    entity_names = {p.name for p in self.people} | {o.name for o in self.organizations}
    for rel in self.relationships:
        if rel.subject not in entity_names or rel.object not in entity_names:
            raise ValueError(f"Relationship references unknown entity")
    return self
```

### Type Safety Gotchas

1. **Entity boundaries**: Where does one entity end and another begin? ("New York City" vs "New York")
2. **Entity types**: Is "Apple" a company or a fruit? Context matters.
3. **Relationship direction**: "Apple acquired Intel division" - who acquired whom?
4. **Temporal context**: Relationships may be past, present, or future
5. **Confidence**: Some entities are certain, others ambiguous

---

## 🎯 Next Steps

Excellent! You now understand:
- ✅ How to extract multiple entity types simultaneously
- ✅ How to preserve relationships between entities
- ✅ How to design robust multi-entity models
- ✅ How to handle partial extractions gracefully
- ✅ How to validate cross-entity relationships

In the next lesson, **Relationship Mapping**, we'll dive deeper into extracting and validating complex relationships between entities.

**Ready for Lesson 12?** 🚀
