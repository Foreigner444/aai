# Lesson 2: Nested Pydantic Models

## A. Concept Overview

### What & Why
**Nested Pydantic models allow you to build complex data structures by composing smaller, reusable models together.** This is critical for data extraction because real-world data has hierarchical relationships—a company has departments, departments have teams, teams have members. By nesting models, you create clear boundaries, reusable components, and automatic validation at every level.

### Analogy
Think of nested models like LEGO blocks:
- **Simple models** are individual LEGO pieces (a Person, an Address)
- **Nested models** are sub-assemblies (a PersonWithAddress combines Person data with an Address block)
- **Complex models** are complete builds (an Organization is made of multiple sub-assemblies that work together)

Just like LEGO, once you build a component (like Address), you can reuse it anywhere (Company.headquarters, Person.address, Warehouse.location). You don't rebuild the Address block each time—you reference the same validated structure.

### Type Safety Benefit
Nested models provide **modular type safety**:
- Each model validates independently—you can test an Address without creating an entire Organization
- Reusable models ensure consistency—every Address in your system has the same validation rules
- IDE autocomplete works at every nesting level—`organization.departments[0].head.address.city` is fully type-checked
- Refactoring is safe—change the Address model once, and all uses are updated
- Error messages are precise—validation errors show the exact path through the nesting: `organization.departments.1.team_members.3.email`

---

## B. Code Implementation

### File Structure
```
data_extraction_pipeline/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py           # New: Base reusable models
│   │   ├── company.py        # New: Company-related nested models
│   │   └── structures.py     # From Lesson 1
│   └── examples/
│       ├── __init__.py
│       ├── nested_models_demo.py  # New: This lesson's examples
│       └── complex_data_demo.py
```

### Complete Code Implementation

**File: `src/models/base.py`**

```python
"""Base reusable models that can be nested in other models."""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime


class Address(BaseModel):
    """Physical address - reusable across Person, Company, Warehouse, etc."""
    street: str = Field(..., min_length=1, description="Street address")
    city: str = Field(..., min_length=1, description="City name")
    state: str = Field(..., min_length=2, max_length=2, description="Two-letter state code")
    zip_code: str = Field(..., pattern=r"^\d{5}(-\d{4})?$", description="ZIP code")
    country: str = Field(default="USA", description="Country name")
    
    def __str__(self) -> str:
        """Human-readable address."""
        return f"{self.street}, {self.city}, {self.state} {self.zip_code}"


class ContactInfo(BaseModel):
    """Contact information - reusable across Person, Company, etc."""
    email: EmailStr = Field(..., description="Email address")
    phone: Optional[str] = Field(None, pattern=r"^\+?[\d\s\-\(\)]+$", description="Phone number")
    website: Optional[str] = Field(None, description="Website URL")
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")
    
    def __str__(self) -> str:
        """Human-readable contact info."""
        parts = [f"Email: {self.email}"]
        if self.phone:
            parts.append(f"Phone: {self.phone}")
        if self.website:
            parts.append(f"Website: {self.website}")
        return ", ".join(parts)


class Person(BaseModel):
    """Person model - reusable across Employee, Customer, Contractor, etc."""
    first_name: str = Field(..., min_length=1, description="First name")
    last_name: str = Field(..., min_length=1, description="Last name")
    date_of_birth: Optional[datetime] = Field(None, description="Date of birth")
    contact: ContactInfo  # Nested model
    
    @property
    def full_name(self) -> str:
        """Computed property combining first and last name."""
        return f"{self.first_name} {self.last_name}"
    
    def __str__(self) -> str:
        return self.full_name


class DateRange(BaseModel):
    """Date range - reusable for projects, employment, events, etc."""
    start_date: datetime = Field(..., description="Start date")
    end_date: Optional[datetime] = Field(None, description="End date (None if ongoing)")
    
    @property
    def is_active(self) -> bool:
        """Check if the date range is currently active."""
        return self.end_date is None or self.end_date > datetime.now()
    
    @property
    def duration_days(self) -> Optional[int]:
        """Calculate duration in days."""
        if self.end_date:
            return (self.end_date - self.start_date).days
        return None
    
    def __str__(self) -> str:
        end = "Present" if self.end_date is None else self.end_date.strftime("%Y-%m-%d")
        return f"{self.start_date.strftime('%Y-%m-%d')} to {end}"
```

**File: `src/models/company.py`**

```python
"""Company-related nested models demonstrating composition patterns."""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime

from .base import Person, Address, ContactInfo, DateRange


class EmploymentType(str, Enum):
    """Employment type options."""
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERN = "intern"


class Department(str, Enum):
    """Department options."""
    ENGINEERING = "engineering"
    SALES = "sales"
    MARKETING = "marketing"
    OPERATIONS = "operations"
    FINANCE = "finance"
    HR = "hr"


class Employee(BaseModel):
    """
    Employee model - extends Person with employment details.
    This demonstrates composition: Employee HAS-A Person's data plus more.
    """
    person: Person  # Nested Person model
    employee_id: str = Field(..., pattern=r"^EMP\d{6}$", description="Employee ID")
    department: Department  # Enum
    employment_type: EmploymentType  # Enum
    employment_period: DateRange  # Nested DateRange
    salary: float = Field(..., gt=0, description="Annual salary")
    manager: Optional["Employee"] = None  # Self-referential nesting!
    
    # Computed properties that delegate to nested models
    @property
    def full_name(self) -> str:
        return self.person.full_name
    
    @property
    def email(self) -> str:
        return self.person.contact.email
    
    @property
    def is_active(self) -> bool:
        return self.employment_period.is_active
    
    def __str__(self) -> str:
        return f"{self.full_name} ({self.employee_id}) - {self.department.value}"


# This is required for self-referential models
Employee.model_rebuild()


class Team(BaseModel):
    """
    Team model - contains multiple Employees in a hierarchy.
    Demonstrates one-to-many nesting.
    """
    name: str = Field(..., min_length=1, description="Team name")
    department: Department
    team_lead: Employee  # Single nested Employee
    members: List[Employee] = Field(default_factory=list, description="Team members")
    
    @property
    def total_members(self) -> int:
        """Total team size including lead."""
        return len(self.members) + 1  # +1 for team lead
    
    @property
    def total_salary_cost(self) -> float:
        """Total salary cost for the entire team."""
        return self.team_lead.salary + sum(member.salary for member in self.members)
    
    def __str__(self) -> str:
        return f"Team {self.name} ({self.total_members} members)"


class Office(BaseModel):
    """
    Office location model.
    Demonstrates nesting Address and ContactInfo for a physical location.
    """
    name: str = Field(..., min_length=1, description="Office name")
    address: Address  # Nested Address
    contact: ContactInfo  # Nested ContactInfo
    capacity: int = Field(..., gt=0, description="Maximum capacity")
    current_occupancy: int = Field(default=0, ge=0, description="Current occupancy")
    
    @property
    def occupancy_rate(self) -> float:
        """Calculate occupancy as a percentage."""
        return (self.current_occupancy / self.capacity) * 100 if self.capacity > 0 else 0.0
    
    def __str__(self) -> str:
        return f"{self.name} - {self.address.city}, {self.address.state}"


class Company(BaseModel):
    """
    Complete company model - deeply nested structure.
    This demonstrates the full power of nested models:
    - Single nested objects (headquarters address)
    - Lists of nested objects (offices, teams)
    - Multiple levels of nesting (teams contain employees, employees contain persons)
    """
    name: str = Field(..., min_length=1, description="Company name")
    founded_date: datetime = Field(..., description="Company founding date")
    headquarters: Address  # Single nested Address
    contact: ContactInfo  # Single nested ContactInfo
    offices: List[Office] = Field(default_factory=list, description="All office locations")
    teams: List[Team] = Field(default_factory=list, description="All teams")
    
    @property
    def total_employees(self) -> int:
        """Count all employees across all teams."""
        return sum(team.total_members for team in self.teams)
    
    @property
    def total_payroll(self) -> float:
        """Calculate total payroll across all teams."""
        return sum(team.total_salary_cost for team in self.teams)
    
    @property
    def total_office_capacity(self) -> int:
        """Sum capacity of all offices."""
        return sum(office.capacity for office in self.offices)
    
    def get_employees_by_department(self, dept: Department) -> List[Employee]:
        """Extract all employees in a specific department."""
        employees = []
        for team in self.teams:
            if team.department == dept:
                employees.append(team.team_lead)
                employees.extend(team.members)
        return employees
    
    def __str__(self) -> str:
        return f"{self.name} - {self.total_employees} employees across {len(self.offices)} offices"
```

**File: `src/examples/nested_models_demo.py`**

```python
"""Demonstration of nested Pydantic models with real-world examples."""

from datetime import datetime, timedelta
from src.models.base import Person, Address, ContactInfo, DateRange
from src.models.company import (
    Employee,
    Team,
    Office,
    Company,
    EmploymentType,
    Department,
)


def demo_basic_nesting():
    """Demonstrate basic nesting: Person with ContactInfo."""
    print("=" * 60)
    print("BASIC NESTING: Person with ContactInfo")
    print("=" * 60)
    
    # Create the nested ContactInfo first
    contact = ContactInfo(
        email="alice.johnson@techcorp.com",
        phone="+1-555-0100",
        website="alice.dev",
        linkedin="linkedin.com/in/alicejohnson"
    )
    
    # Create Person with nested ContactInfo
    person = Person(
        first_name="Alice",
        last_name="Johnson",
        date_of_birth=datetime(1992, 5, 15),
        contact=contact
    )
    
    # Access nested data with type safety
    print(f"Name: {person.full_name}")
    print(f"Email: {person.contact.email}")  # Access nested field
    print(f"Phone: {person.contact.phone}")
    print(f"LinkedIn: {person.contact.linkedin}")
    print(f"\nString representation: {person}")
    print(f"Contact info: {contact}")
    print()


def demo_multiple_levels():
    """Demonstrate multiple levels of nesting: Employee -> Person -> ContactInfo."""
    print("=" * 60)
    print("MULTIPLE NESTING LEVELS: Employee -> Person -> ContactInfo")
    print("=" * 60)
    
    # Level 3: ContactInfo
    contact = ContactInfo(
        email="bob.smith@techcorp.com",
        phone="+1-555-0200"
    )
    
    # Level 2: Person with nested ContactInfo
    person = Person(
        first_name="Bob",
        last_name="Smith",
        date_of_birth=datetime(1988, 8, 22),
        contact=contact
    )
    
    # Level 2: DateRange
    employment_period = DateRange(
        start_date=datetime(2020, 1, 15),
        end_date=None  # Still employed
    )
    
    # Level 1: Employee with nested Person and DateRange
    employee = Employee(
        person=person,
        employee_id="EMP000001",
        department=Department.ENGINEERING,
        employment_type=EmploymentType.FULL_TIME,
        employment_period=employment_period,
        salary=120000.0
    )
    
    # Access data through multiple nesting levels
    print(f"Employee: {employee.full_name}")
    print(f"ID: {employee.employee_id}")
    print(f"Department: {employee.department.value}")
    print(f"Email: {employee.person.contact.email}")  # Three levels deep!
    print(f"Employment: {employee.employment_period}")
    print(f"Is Active: {employee.is_active}")
    print(f"Salary: ${employee.salary:,.2f}")
    print()


def demo_self_referential():
    """Demonstrate self-referential nesting: Employee with manager (also an Employee)."""
    print("=" * 60)
    print("SELF-REFERENTIAL NESTING: Employee with Manager")
    print("=" * 60)
    
    # Create the manager first
    manager_person = Person(
        first_name="Sarah",
        last_name="Director",
        contact=ContactInfo(email="sarah.director@techcorp.com")
    )
    
    manager = Employee(
        person=manager_person,
        employee_id="EMP000010",
        department=Department.ENGINEERING,
        employment_type=EmploymentType.FULL_TIME,
        employment_period=DateRange(start_date=datetime(2018, 3, 1)),
        salary=180000.0,
        manager=None  # Top-level manager has no manager
    )
    
    # Create employee with manager
    employee_person = Person(
        first_name="Charlie",
        last_name="Developer",
        contact=ContactInfo(email="charlie.dev@techcorp.com")
    )
    
    employee = Employee(
        person=employee_person,
        employee_id="EMP000050",
        department=Department.ENGINEERING,
        employment_type=EmploymentType.FULL_TIME,
        employment_period=DateRange(start_date=datetime(2022, 6, 1)),
        salary=100000.0,
        manager=manager  # Nested Employee!
    )
    
    print(f"Employee: {employee.full_name}")
    print(f"Manager: {employee.manager.full_name if employee.manager else 'None'}")
    print(f"Manager's Email: {employee.manager.person.contact.email}")
    print(f"Manager's Salary: ${employee.manager.salary:,.2f}")
    print()


def demo_collection_nesting():
    """Demonstrate collections of nested models: Team with multiple Employees."""
    print("=" * 60)
    print("COLLECTION NESTING: Team with Multiple Employees")
    print("=" * 60)
    
    # Create team lead
    lead_person = Person(
        first_name="Diana",
        last_name="Lead",
        contact=ContactInfo(email="diana.lead@techcorp.com")
    )
    
    team_lead = Employee(
        person=lead_person,
        employee_id="EMP000100",
        department=Department.ENGINEERING,
        employment_type=EmploymentType.FULL_TIME,
        employment_period=DateRange(start_date=datetime(2019, 1, 1)),
        salary=150000.0
    )
    
    # Create team members
    members = []
    for i, (first, last) in enumerate([("Eve", "Dev1"), ("Frank", "Dev2"), ("Grace", "Dev3")], start=1):
        person = Person(
            first_name=first,
            last_name=last,
            contact=ContactInfo(email=f"{first.lower()}.{last.lower()}@techcorp.com")
        )
        member = Employee(
            person=person,
            employee_id=f"EMP00010{i}",
            department=Department.ENGINEERING,
            employment_type=EmploymentType.FULL_TIME,
            employment_period=DateRange(start_date=datetime(2021, 1, 1)),
            salary=90000.0 + (i * 5000),
            manager=team_lead
        )
        members.append(member)
    
    # Create team
    team = Team(
        name="Backend Engineering",
        department=Department.ENGINEERING,
        team_lead=team_lead,
        members=members
    )
    
    print(f"Team: {team.name}")
    print(f"Department: {team.department.value}")
    print(f"Team Lead: {team.team_lead.full_name}")
    print(f"Total Members: {team.total_members}")
    print(f"Total Salary Cost: ${team.total_salary_cost:,.2f}")
    print(f"\nTeam Members:")
    for member in team.members:
        print(f"  - {member.full_name} ({member.employee_id}): ${member.salary:,.2f}")
    print()


def demo_deeply_nested_company():
    """Demonstrate deeply nested structure: Complete Company model."""
    print("=" * 60)
    print("DEEPLY NESTED STRUCTURE: Complete Company")
    print("=" * 60)
    
    # Company headquarters
    hq_address = Address(
        street="100 Tech Drive",
        city="San Francisco",
        state="CA",
        zip_code="94105"
    )
    
    company_contact = ContactInfo(
        email="info@techcorp.com",
        phone="+1-800-TECH-CORP",
        website="https://techcorp.com"
    )
    
    # Office 1: SF Office
    sf_office = Office(
        name="San Francisco HQ",
        address=hq_address,
        contact=company_contact,
        capacity=200,
        current_occupancy=150
    )
    
    # Office 2: NY Office
    ny_office = Office(
        name="New York Office",
        address=Address(
            street="500 Broadway",
            city="New York",
            state="NY",
            zip_code="10012"
        ),
        contact=ContactInfo(
            email="ny@techcorp.com",
            phone="+1-212-555-0100"
        ),
        capacity=150,
        current_occupancy=100
    )
    
    # Create Engineering Team
    eng_lead = Employee(
        person=Person(
            first_name="Sarah",
            last_name="Tech",
            contact=ContactInfo(email="sarah.tech@techcorp.com")
        ),
        employee_id="EMP000001",
        department=Department.ENGINEERING,
        employment_type=EmploymentType.FULL_TIME,
        employment_period=DateRange(start_date=datetime(2018, 1, 1)),
        salary=180000.0
    )
    
    eng_members = [
        Employee(
            person=Person(
                first_name=f"Dev{i}",
                last_name="Engineer",
                contact=ContactInfo(email=f"dev{i}@techcorp.com")
            ),
            employee_id=f"EMP00000{i+1}",
            department=Department.ENGINEERING,
            employment_type=EmploymentType.FULL_TIME,
            employment_period=DateRange(start_date=datetime(2020, 1, 1)),
            salary=120000.0
        )
        for i in range(1, 6)
    ]
    
    engineering_team = Team(
        name="Platform Engineering",
        department=Department.ENGINEERING,
        team_lead=eng_lead,
        members=eng_members
    )
    
    # Create Sales Team
    sales_lead = Employee(
        person=Person(
            first_name="John",
            last_name="Sales",
            contact=ContactInfo(email="john.sales@techcorp.com")
        ),
        employee_id="EMP000010",
        department=Department.SALES,
        employment_type=EmploymentType.FULL_TIME,
        employment_period=DateRange(start_date=datetime(2019, 6, 1)),
        salary=160000.0
    )
    
    sales_members = [
        Employee(
            person=Person(
                first_name=f"Sales{i}",
                last_name="Rep",
                contact=ContactInfo(email=f"sales{i}@techcorp.com")
            ),
            employee_id=f"EMP00001{i}",
            department=Department.SALES,
            employment_type=EmploymentType.FULL_TIME,
            employment_period=DateRange(start_date=datetime(2021, 1, 1)),
            salary=100000.0
        )
        for i in range(1, 4)
    ]
    
    sales_team = Team(
        name="Enterprise Sales",
        department=Department.SALES,
        team_lead=sales_lead,
        members=sales_members
    )
    
    # Create Company
    company = Company(
        name="TechCorp Industries",
        founded_date=datetime(2015, 3, 15),
        headquarters=hq_address,
        contact=company_contact,
        offices=[sf_office, ny_office],
        teams=[engineering_team, sales_team]
    )
    
    # Access deeply nested data
    print(f"Company: {company.name}")
    print(f"Founded: {company.founded_date.strftime('%B %d, %Y')}")
    print(f"Headquarters: {company.headquarters}")
    print(f"Total Employees: {company.total_employees}")
    print(f"Total Payroll: ${company.total_payroll:,.2f}")
    print(f"Office Capacity: {company.total_office_capacity}")
    
    print(f"\nOffices:")
    for office in company.offices:
        print(f"  - {office.name}: {office.current_occupancy}/{office.capacity} "
              f"({office.occupancy_rate:.1f}% occupancy)")
    
    print(f"\nTeams:")
    for team in company.teams:
        print(f"  - {team.name} ({team.department.value})")
        print(f"    Lead: {team.team_lead.full_name}")
        print(f"    Members: {len(team.members)}")
        print(f"    Total Cost: ${team.total_salary_cost:,.2f}")
    
    # Demonstrate filtering
    print(f"\nEngineering Department Employees:")
    eng_employees = company.get_employees_by_department(Department.ENGINEERING)
    for emp in eng_employees:
        print(f"  - {emp.full_name} ({emp.employee_id}): ${emp.salary:,.2f}")
    
    print()


if __name__ == "__main__":
    print("\n🎯 NESTED PYDANTIC MODELS DEMONSTRATION\n")
    
    demo_basic_nesting()
    demo_multiple_levels()
    demo_self_referential()
    demo_collection_nesting()
    demo_deeply_nested_company()
    
    print("=" * 60)
    print("✅ All demonstrations completed!")
    print("=" * 60)
```

### Line-by-Line Explanation

**Base Models (`base.py`):**

1. **Address**: Reusable location model with field constraints (state must be 2 chars, zip code follows regex pattern)
2. **ContactInfo**: Communication details with EmailStr for automatic email validation
3. **Person**: Core person model that nests ContactInfo, with computed property `full_name`
4. **DateRange**: Time period model with computed properties for active status and duration

**Company Models (`company.py`):**

1. **Employee**: Extends Person by nesting it, adds employment details, demonstrates self-referential nesting with optional `manager` field
2. **Team**: Contains one lead Employee and a list of member Employees, shows one-to-many relationships
3. **Office**: Combines Address and ContactInfo for physical locations
4. **Company**: Top-level model containing lists of Offices and Teams, with methods to aggregate data across nested structures

**Key Patterns:**

- **Composition over Inheritance**: Employee HAS-A Person, not IS-A Person
- **Reusability**: Address is used in Office, Company.headquarters, Person (if extended)
- **Self-Reference**: Employee.manager is another Employee
- **Deep Nesting**: Company -> Team -> Employee -> Person -> ContactInfo (5 levels!)
- **Computed Properties**: Use @property to add methods that delegate to nested models

### The "Why" Behind the Pattern

**Separation of Concerns:**
Each model has a single responsibility. Address handles location data, Person handles identity, Employee handles employment. When Gemini extracts data, each validator focuses on its domain.

**Reusability and Consistency:**
Define Address once, use it everywhere. Every address in your system follows the same validation rules. When you need to add a field (like `apartment_number`), add it once and all uses inherit it.

**Maintenance and Refactoring:**
If email validation rules change, update ContactInfo once. All models using ContactInfo automatically get the new validation. Your IDE can find all uses of a model, making refactoring safe.

**Type Safety Through Composition:**
`company.teams[0].team_lead.person.contact.email` is fully typed. Your IDE knows the type at every level. Mistakes like `company.teams[0].email` are caught immediately.

**Validation Cascades Automatically:**
When you validate a Company, Pydantic automatically validates every Office, Team, Employee, Person, ContactInfo, Address, and DateRange in the entire structure. One call, complete validation.

---

## C. Test & Apply

### How to Test It

**Step 1: Create the new files**
```bash
cd data_extraction_pipeline
touch src/models/base.py
touch src/models/company.py
touch src/examples/nested_models_demo.py
```

**Step 2: Copy the code**
Copy the code from above into each file.

**Step 3: Update `src/models/__init__.py`**
```python
"""Models package for data extraction pipeline."""

from .base import Address, ContactInfo, Person, DateRange
from .company import Employee, Team, Office, Company, Department, EmploymentType

__all__ = [
    "Address",
    "ContactInfo",
    "Person",
    "DateRange",
    "Employee",
    "Team",
    "Office",
    "Company",
    "Department",
    "EmploymentType",
]
```

**Step 4: Install email validation support**
```bash
pip install "pydantic[email]"
```

**Step 5: Run the demonstration**
```bash
python -m src.examples.nested_models_demo
```

### Expected Result

You should see detailed output showing:

```
🎯 NESTED PYDANTIC MODELS DEMONSTRATION

==================================================
BASIC NESTING: Person with ContactInfo
==================================================
Name: Alice Johnson
Email: alice.johnson@techcorp.com
Phone: +1-555-0100
LinkedIn: linkedin.com/in/alicejohnson

String representation: Alice Johnson
Contact info: Email: alice.johnson@techcorp.com, Phone: +1-555-0100, Website: alice.dev

==================================================
MULTIPLE NESTING LEVELS: Employee -> Person -> ContactInfo
==================================================
Employee: Bob Smith
ID: EMP000001
Department: engineering
Email: bob.smith@techcorp.com
Employment: 2020-01-15 to Present
Is Active: True
Salary: $120,000.00

==================================================
SELF-REFERENTIAL NESTING: Employee with Manager
==================================================
Employee: Charlie Developer
Manager: Sarah Director
Manager's Email: sarah.director@techcorp.com
Manager's Salary: $180,000.00

==================================================
COLLECTION NESTING: Team with Multiple Employees
==================================================
Team: Backend Engineering
Department: engineering
Team Lead: Diana Lead
Total Members: 4
Total Salary Cost: $435,000.00

Team Members:
  - Eve Dev1 (EMP000101): $95,000.00
  - Frank Dev2 (EMP000102): $100,000.00
  - Grace Dev3 (EMP000103): $105,000.00

==================================================
DEEPLY NESTED STRUCTURE: Complete Company
==================================================
Company: TechCorp Industries
Founded: March 15, 2015
Headquarters: 100 Tech Drive, San Francisco, CA 94105
Total Employees: 11
Total Payroll: $1,560,000.00
Office Capacity: 350

Offices:
  - San Francisco HQ: 150/200 (75.0% occupancy)
  - New York Office: 100/150 (66.7% occupancy)

Teams:
  - Platform Engineering (engineering)
    Lead: Sarah Tech
    Members: 5
    Total Cost: $780,000.00
  - Enterprise Sales (sales)
    Lead: John Sales
    Members: 3
    Total Cost: $460,000.00

Engineering Department Employees:
  - Sarah Tech (EMP000001): $180,000.00
  - Dev1 Engineer (EMP000002): $120,000.00
  - Dev2 Engineer (EMP000003): $120,000.00
  - Dev3 Engineer (EMP000004): $120,000.00
  - Dev4 Engineer (EMP000005): $120,000.00
  - Dev5 Engineer (EMP000006): $120,000.00

==================================================
✅ All demonstrations completed!
==================================================
```

### Validation Examples

**Create a file `src/examples/nested_validation_demo.py`:**

```python
"""Demonstrate validation errors in nested models."""

from src.models.base import Person, ContactInfo, Address
from src.models.company import Employee, Department, EmploymentType, DateRange
from datetime import datetime
from pydantic import ValidationError


def demo_nested_validation_errors():
    """Show how validation errors work in nested structures."""
    
    print("🚫 NESTED VALIDATION ERROR EXAMPLES\n")
    
    # Error 1: Invalid email in nested ContactInfo
    print("Error 1: Invalid email in nested ContactInfo")
    try:
        person = Person(
            first_name="Test",
            last_name="User",
            contact=ContactInfo(
                email="not-an-email",  # Invalid email
                phone="+1-555-0100"
            )
        )
    except ValidationError as e:
        print(f"❌ {e}\n")
    
    # Error 2: Invalid field in deeply nested model
    print("Error 2: Invalid ZIP code in nested Address")
    try:
        address = Address(
            street="123 Main St",
            city="Springfield",
            state="IL",
            zip_code="invalid-zip"  # Must match regex pattern
        )
    except ValidationError as e:
        print(f"❌ {e}\n")
    
    # Error 3: Missing required nested object
    print("Error 3: Missing required nested ContactInfo")
    try:
        person = Person(
            first_name="Test",
            last_name="User"
            # contact is missing!
        )
    except ValidationError as e:
        print(f"❌ {e}\n")
    
    # Error 4: Invalid data in self-referential nesting
    print("Error 4: Invalid employee ID format")
    try:
        employee = Employee(
            person=Person(
                first_name="Test",
                last_name="User",
                contact=ContactInfo(email="test@example.com")
            ),
            employee_id="INVALID",  # Must match pattern EMP\d{6}
            department=Department.ENGINEERING,
            employment_type=EmploymentType.FULL_TIME,
            employment_period=DateRange(start_date=datetime.now()),
            salary=100000.0
        )
    except ValidationError as e:
        print(f"❌ {e}\n")
    
    print("✅ Validation demonstration complete!")


if __name__ == "__main__":
    demo_nested_validation_errors()
```

**Run it:**
```bash
python -m src.examples.nested_validation_demo
```

### Type Checking

**Create `src/examples/type_safety_demo.py`:**

```python
"""Demonstrate type safety in nested models."""

from src.models.company import Company, Team, Employee, Office

# Your IDE will autocomplete all of these!

def process_company(company: Company) -> None:
    """Process company data with full type safety."""
    
    # IDE knows company.name is str
    company_name: str = company.name.upper()
    
    # IDE knows company.teams is List[Team]
    first_team: Team = company.teams[0]
    
    # IDE knows first_team.team_lead is Employee
    lead: Employee = first_team.team_lead
    
    # IDE knows lead.person.full_name is str (from @property)
    lead_name: str = lead.full_name
    
    # IDE knows lead.person.contact.email is str (actually EmailStr)
    lead_email: str = lead.email
    
    # This would show an error in your IDE:
    # invalid = company.invalid_field  # ❌ Company has no field 'invalid_field'
    # invalid = company.name.some_method()  # ❌ str has no method 'some_method'
    # invalid = first_team.members.invalid  # ❌ List has no attribute 'invalid'
    
    print(f"Processed {company_name}")
    print(f"First team lead: {lead_name} ({lead_email})")


# Type checking also works for function parameters
def get_employee_email(emp: Employee) -> str:
    """Extract email from employee - fully type safe."""
    return emp.person.contact.email  # Type checker verifies this entire path


def calculate_team_cost(team: Team) -> float:
    """Calculate team cost - type safe aggregation."""
    # Type checker knows team_lead.salary is float
    lead_cost: float = team.team_lead.salary
    
    # Type checker knows members is List[Employee] and salary is float
    members_cost: float = sum(member.salary for member in team.members)
    
    return lead_cost + members_cost
```

**Run mypy:**
```bash
pip install mypy
mypy src/ --strict
```

If there are no type errors, you'll see:
```
Success: no issues found in X source files
```

---

## D. Common Stumbling Blocks

### Proactive Debugging

**Mistake 1: Forgetting to rebuild self-referential models**

When a model references itself (like Employee.manager being another Employee), you must call `Model.model_rebuild()` after the class definition:

```python
class Employee(BaseModel):
    # ... fields ...
    manager: Optional["Employee"] = None  # Forward reference as string

# Required! Without this, Pydantic can't resolve the forward reference
Employee.model_rebuild()
```

**Mistake 2: Circular imports between models**

If `company.py` imports from `employee.py` and `employee.py` imports from `company.py`, you'll get an ImportError.

**The Fix:** Keep related models in the same file, or use string forward references:

```python
# Instead of:
from .employee import Employee  # Might cause circular import

# Use forward reference:
class Team(BaseModel):
    lead: "Employee"  # String reference, resolved later
```

**Mistake 3: Confusing nesting with inheritance**

```python
# ❌ WRONG - Trying to inherit
class Employee(Person):  # This makes Employee a subclass of Person
    employee_id: str

# ✅ CORRECT - Composition (nesting)
class Employee(BaseModel):
    person: Person  # Employee HAS-A Person
    employee_id: str
```

Use composition (nesting) for Pydantic models. Inheritance can cause validation issues and breaks the separation of concerns.

### Show the Error

**Error 1: Invalid nested field**

```python
person = Person(
    first_name="Alice",
    last_name="Johnson",
    contact=ContactInfo(
        email="invalid-email",  # Not a valid email format
        phone="+1-555-0100"
    )
)
```

**Error message:**
```
ValidationError: 1 validation error for Person
contact.email
  value is not a valid email address: The part after the @ sign is not valid. It should have a period. [type=value_error, input_value='invalid-email', input_type=str]
```

Notice the error path: `contact.email` tells you exactly where in the nesting the error occurred.

**Error 2: Wrong type for nested object**

```python
person = Person(
    first_name="Bob",
    last_name="Smith",
    contact="bob@example.com"  # ❌ String instead of ContactInfo object
)
```

**Error message:**
```
ValidationError: 1 validation error for Person
contact
  Input should be a valid dictionary or instance of ContactInfo [type=model_type, input_value='bob@example.com', input_type=str]
```

**Error 3: Validation failure deep in nesting**

```python
employee = Employee(
    person=Person(
        first_name="Charlie",
        last_name="Dev",
        contact=ContactInfo(
            email="charlie@example.com",
            phone="invalid-phone-format-###"  # Doesn't match regex pattern
        )
    ),
    employee_id="EMP000001",
    department=Department.ENGINEERING,
    employment_type=EmploymentType.FULL_TIME,
    employment_period=DateRange(start_date=datetime.now()),
    salary=100000.0
)
```

**Error message:**
```
ValidationError: 1 validation error for Employee
person.contact.phone
  String should match pattern '^\+?[\d\s\-\(\)]+$' [type=string_pattern_mismatch, input_value='invalid-phone-format-###', input_type=str]
```

The error path `person.contact.phone` shows you the exact location through three levels of nesting!

### Explain the Fix

**For Invalid Email:**
- Use a properly formatted email address: `user@domain.com`
- Pydantic uses `EmailStr` which validates email format automatically
- Check for typos like missing `@` or `.com`

**For Wrong Type in Nested Object:**
- Pydantic can auto-convert dicts to models, but not strings
- Either pass a dict: `contact={"email": "bob@example.com"}` (Pydantic converts it)
- Or create the object: `contact=ContactInfo(email="bob@example.com")`

**For Deep Validation Failures:**
- Read the error path from left to right: `person.contact.phone`
- This means: in the `person` field, in its `contact` field, the `phone` field is invalid
- Fix the field at that exact location in your data

### Type Safety Gotchas

1. **Forward References**: When a model references itself or another model defined later, use string annotations: `"Employee"` not `Employee`.

2. **Model Rebuild**: Always call `Model.model_rebuild()` after defining self-referential models.

3. **Optional Nested Objects**: `manager: Optional[Employee] = None` means the manager can be None, but if provided, it must be a valid Employee.

4. **List Type Parameters**: `List[Employee]` means a list of Employee objects. `List` alone loses type safety.

5. **Circular Imports**: Keep related models in the same file or use forward references to avoid circular imports.

6. **Computed Properties**: Use `@property` for read-only computed values. These aren't validated by Pydantic but provide type-safe access.

---

## 🎯 Next Steps

Excellent work! You now understand:
- ✅ How to compose complex models from simple reusable components
- ✅ How to nest models multiple levels deep
- ✅ How to create self-referential models (Employee.manager)
- ✅ How to work with collections of nested objects
- ✅ How validation cascades through the entire nested structure
- ✅ How type safety works at every nesting level

In the next lesson, we'll explore **Lists and Collections in Models** in depth—learning advanced patterns for handling arrays of data, validation of list items, and collection-specific constraints.

**Ready for Lesson 3, or would you like to practice building nested structures?** 🚀
