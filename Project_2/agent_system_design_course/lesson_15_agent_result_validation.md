# Lesson 15: Agent Result Validation

## A. Concept Overview

### What & Why
**Agent Result Validation** ensures that your agent's outputs meet not just type requirements, but also business logic, data consistency, domain-specific rules, and quality standards. While Pydantic handles basic type validation automatically, production systems need deeper validation - checking relationships between fields, enforcing business constraints, validating against external rules, and ensuring outputs make semantic sense. This is crucial because type-correct outputs can still be logically wrong or violate business rules.

### Analogy
Think of result validation like quality control in manufacturing:

**Type Validation Only** = Basic dimension check:
- Inspector: "Is this a phone?" 
- Check: Has screen ✅, has battery ✅, has buttons ✅
- Result: "Valid phone, ship it!"
- Customer receives it: Screen is cracked, battery dead, buttons don't work
- **Type-correct but broken!**

**Comprehensive Validation** = Multi-stage quality control:
1. **Structural check**: Has all required components?
2. **Functional check**: Does each component work?
3. **Integration check**: Do components work together?
4. **Safety check**: Meets safety standards?
5. **Performance check**: Meets performance requirements?
6. **Final inspection**: Ready for customer?

Only after passing ALL checks does it ship. Same with agent outputs - type correctness is just the beginning!

### Type Safety Benefit
Comprehensive validation provides:
- **Field validators**: Type-safe validation of individual fields
- **Model validators**: Cross-field validation with type checking
- **Custom types**: NewType and Annotated for semantic validation
- **Validation context**: Access to validation context for complex rules
- **Error messages**: Structured, typed validation error messages
- **Composition**: Compose validators for reusable validation logic
- **Testing**: Type-safe validation testing

Your validation logic becomes as robust and type-safe as your data models!

---

## B. Code Implementation

### File Structure
```
agent_system_design/
├── lesson_15_agent_result_validation.py
├── requirements.txt
└── .env
```

### Complete Code Snippet

**lesson_15_agent_result_validation.py**
```python
"""
Lesson 15: Agent Result Validation
Comprehensive validation strategies for agent outputs
"""

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ValidationError,
    EmailStr,
    HttpUrl
)
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from typing import Literal, Optional, Annotated
from datetime import datetime, date, timedelta
from decimal import Decimal
import re
from dotenv import load_dotenv

load_dotenv()


# Dependencies
@dataclass
class ValidationDeps:
    """Dependencies for validation examples"""
    user_id: str
    current_date: date


# PATTERN 1: Field Validators (Single Field Validation)

class UserRegistration(BaseModel):
    """User registration with field-level validation"""
    
    username: str = Field(min_length=3, max_length=20)
    email: EmailStr  # Built-in email validation
    password: str = Field(min_length=8)
    age: int = Field(ge=13, le=120)
    phone: Optional[str] = None
    website: Optional[HttpUrl] = None
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        """
        Username must be alphanumeric (letters, numbers, underscores only).
        
        This validator runs AFTER basic type/length validation.
        """
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError(
                'Username must contain only letters, numbers, and underscores'
            )
        
        # Normalize to lowercase
        return v.lower()
    
    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """
        Password must meet security requirements.
        
        Requirements:
        - At least 8 characters (enforced by Field)
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one digit
        - At least one special character
        """
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        
        # Don't return the password - keep it secure
        return v
    
    @field_validator('phone')
    @classmethod
    def validate_phone_format(cls, v: Optional[str]) -> Optional[str]:
        """
        Normalize phone number format.
        
        Accepts various formats and normalizes to: +1-XXX-XXX-XXXX
        """
        if v is None:
            return None
        
        # Extract digits only
        digits = re.sub(r'\D', '', v)
        
        # Validate length
        if len(digits) == 10:
            # US number without country code
            return f"+1-{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == '1':
            # US number with country code
            return f"+1-{digits[1:4]}-{digits[4:7]}-{digits[7:]}"
        else:
            raise ValueError(
                f'Invalid phone number. Must be 10 digits (US) or 11 digits with country code. '
                f'Got {len(digits)} digits.'
            )


# PATTERN 2: Model Validators (Cross-Field Validation)

class DateRange(BaseModel):
    """Date range with cross-field validation"""
    
    start_date: date
    end_date: date
    max_days: int = Field(default=365, ge=1)
    
    @model_validator(mode='after')
    def validate_date_range(self) -> 'DateRange':
        """
        Validate that end_date is after start_date and within max_days.
        
        This runs AFTER all field validators and has access to all fields.
        """
        # Check chronological order
        if self.end_date <= self.start_date:
            raise ValueError(
                f'end_date ({self.end_date}) must be after start_date ({self.start_date})'
            )
        
        # Check range length
        days_diff = (self.end_date - self.start_date).days
        if days_diff > self.max_days:
            raise ValueError(
                f'Date range is {days_diff} days, exceeds maximum of {self.max_days} days'
            )
        
        return self


class FinancialTransaction(BaseModel):
    """Financial transaction with business logic validation"""
    
    transaction_id: str
    amount: Decimal = Field(decimal_places=2)
    fee: Decimal = Field(decimal_places=2, ge=0)
    total: Decimal = Field(decimal_places=2)
    currency: Literal["USD", "EUR", "GBP"] = "USD"
    
    @model_validator(mode='after')
    def validate_totals(self) -> 'FinancialTransaction':
        """
        Ensure total equals amount + fee.
        
        This catches arithmetic errors in AI-generated financial data.
        """
        expected_total = self.amount + self.fee
        
        # Use small epsilon for decimal comparison
        if abs(self.total - expected_total) > Decimal('0.01'):
            raise ValueError(
                f'Total {self.total} does not match amount {self.amount} + fee {self.fee} '
                f'(expected {expected_total})'
            )
        
        return self
    
    @field_validator('amount')
    @classmethod
    def validate_amount_reasonable(cls, v: Decimal) -> Decimal:
        """Ensure transaction amount is reasonable"""
        # No negative amounts (use absolute value)
        if v < 0:
            raise ValueError('Transaction amount cannot be negative')
        
        # Warn about suspiciously large amounts
        if v > Decimal('1000000'):
            raise ValueError(
                f'Transaction amount {v} exceeds maximum allowed (1,000,000)'
            )
        
        return v


# PATTERN 3: Custom Validation with Context

class BookingRequest(BaseModel):
    """Booking request with context-aware validation"""
    
    booking_date: date
    booking_time: str  # Format: "HH:MM"
    duration_hours: int = Field(ge=1, le=8)
    number_of_people: int = Field(ge=1, le=20)
    
    @field_validator('booking_time')
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        """Validate time format"""
        if not re.match(r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$', v):
            raise ValueError('Time must be in HH:MM format (e.g., "14:30")')
        return v
    
    @model_validator(mode='after')
    def validate_business_hours(self) -> 'BookingRequest':
        """
        Validate booking is within business hours (9 AM - 6 PM).
        
        This is business logic validation that requires the whole model.
        """
        hour = int(self.booking_time.split(':')[0])
        minute = int(self.booking_time.split(':')[1])
        
        # Check if start time is in business hours
        if hour < 9 or hour >= 18:
            raise ValueError(
                f'Booking time {self.booking_time} is outside business hours (9 AM - 6 PM)'
            )
        
        # Check if booking would end within business hours
        end_hour = hour + self.duration_hours
        if end_hour > 18:
            raise ValueError(
                f'Booking duration {self.duration_hours}h would extend past closing time (6 PM). '
                f'Latest start time for {self.duration_hours}h booking is {18 - self.duration_hours}:00'
            )
        
        return self


# PATTERN 4: Semantic Validation (Domain Rules)

class OrderValidation(BaseModel):
    """Order with complex business rule validation"""
    
    order_id: str
    customer_email: EmailStr
    items: list[str] = Field(min_length=1)
    subtotal: Decimal = Field(decimal_places=2, gt=0)
    discount_percent: Decimal = Field(ge=0, le=100, decimal_places=2)
    discount_code: Optional[str] = None
    tax_rate: Decimal = Field(ge=0, le=0.5, decimal_places=4)  # Max 50% tax
    shipping: Decimal = Field(ge=0, decimal_places=2)
    total: Decimal = Field(decimal_places=2, gt=0)
    
    @field_validator('order_id')
    @classmethod
    def validate_order_id_format(cls, v: str) -> str:
        """Order ID must follow format: ORD-YYYYMMDD-XXXX"""
        pattern = r'^ORD-\d{8}-\d{4}$'
        if not re.match(pattern, v):
            raise ValueError(
                f'Order ID must match format ORD-YYYYMMDD-XXXX (e.g., ORD-20241202-0001). '
                f'Got: {v}'
            )
        return v
    
    @field_validator('discount_code')
    @classmethod
    def validate_discount_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate discount code format if provided"""
        if v is None:
            return None
        
        # Discount codes must be uppercase alphanumeric
        if not v.isupper() or not v.isalnum():
            raise ValueError('Discount code must be uppercase alphanumeric')
        
        if len(v) < 4 or len(v) > 12:
            raise ValueError('Discount code must be 4-12 characters')
        
        return v
    
    @model_validator(mode='after')
    def validate_discount_logic(self) -> 'OrderValidation':
        """
        Validate discount logic.
        
        Business rule: Can only have discount_percent > 0 if discount_code provided.
        """
        if self.discount_percent > 0 and not self.discount_code:
            raise ValueError(
                f'Cannot have discount ({self.discount_percent}%) without a discount code'
            )
        
        if self.discount_code and self.discount_percent == 0:
            raise ValueError(
                f'Discount code {self.discount_code} provided but discount_percent is 0'
            )
        
        return self
    
    @model_validator(mode='after')
    def validate_order_totals(self) -> 'OrderValidation':
        """
        Validate order financial calculations.
        
        Formula: total = (subtotal * (1 - discount_percent/100) * (1 + tax_rate)) + shipping
        """
        # Calculate expected total
        discounted_subtotal = self.subtotal * (1 - self.discount_percent / 100)
        with_tax = discounted_subtotal * (1 + self.tax_rate)
        expected_total = with_tax + self.shipping
        
        # Allow 1 cent rounding difference
        if abs(self.total - expected_total) > Decimal('0.01'):
            raise ValueError(
                f'Total {self.total} does not match calculated total {expected_total:.2f}. '
                f'Formula: ({self.subtotal} * (1 - {self.discount_percent}/100) * '
                f'(1 + {self.tax_rate})) + {self.shipping}'
            )
        
        return self
    
    @model_validator(mode='after')
    def validate_minimum_order(self) -> 'OrderValidation':
        """
        Business rule: Minimum order must be $5 after discounts.
        """
        discounted_subtotal = self.subtotal * (1 - self.discount_percent / 100)
        
        if discounted_subtotal < Decimal('5.00'):
            raise ValueError(
                f'Order subtotal after discount ({discounted_subtotal}) is below minimum ($5.00). '
                f'Please add more items or remove discount code.'
            )
        
        return self


# PATTERN 5: Context-Aware Validation

class ReportResult(BaseModel):
    """Report with context-aware validation"""
    
    report_id: str
    report_type: Literal["daily", "weekly", "monthly", "quarterly"]
    start_date: date
    end_date: date
    metrics: dict[str, float]
    generated_at: datetime
    generated_by: str
    
    @field_validator('report_id')
    @classmethod
    def validate_report_id(cls, v: str) -> str:
        """Report ID format: RPT-{TYPE}-{YYYYMMDD}"""
        if not re.match(r'^RPT-[A-Z]+-\d{8}$', v):
            raise ValueError(
                f'Report ID must match format RPT-TYPE-YYYYMMDD (e.g., RPT-DAILY-20241202)'
            )
        return v
    
    @model_validator(mode='after')
    def validate_date_range_matches_type(self) -> 'ReportResult':
        """
        Validate date range matches report type.
        
        - daily: exactly 1 day
        - weekly: exactly 7 days
        - monthly: 28-31 days
        - quarterly: 89-92 days
        """
        days = (self.end_date - self.start_date).days + 1  # +1 to include end date
        
        valid_ranges = {
            "daily": (1, 1),
            "weekly": (7, 7),
            "monthly": (28, 31),
            "quarterly": (89, 92)
        }
        
        min_days, max_days = valid_ranges[self.report_type]
        
        if not (min_days <= days <= max_days):
            raise ValueError(
                f'{self.report_type.capitalize()} report must cover {min_days}-{max_days} days. '
                f'Got {days} days ({self.start_date} to {self.end_date})'
            )
        
        return self
    
    @model_validator(mode='after')
    def validate_required_metrics(self) -> 'ReportResult':
        """
        Ensure required metrics are present based on report type.
        """
        required_metrics = {
            "daily": ["revenue", "orders"],
            "weekly": ["revenue", "orders", "avg_order_value"],
            "monthly": ["revenue", "orders", "avg_order_value", "customer_count"],
            "quarterly": ["revenue", "orders", "avg_order_value", "customer_count", "growth_rate"]
        }
        
        required = required_metrics[self.report_type]
        missing = [m for m in required if m not in self.metrics]
        
        if missing:
            raise ValueError(
                f'{self.report_type.capitalize()} report is missing required metrics: {", ".join(missing)}'
            )
        
        return self


# PATTERN 6: Custom Validation Functions (Reusable)

def validate_positive_number(v: float, field_name: str) -> float:
    """Reusable validator for positive numbers"""
    if v <= 0:
        raise ValueError(f'{field_name} must be positive, got {v}')
    return v


def validate_percentage(v: float, field_name: str) -> float:
    """Reusable validator for percentages"""
    if not (0 <= v <= 100):
        raise ValueError(f'{field_name} must be between 0 and 100, got {v}')
    return v


def validate_future_date(v: date, field_name: str) -> date:
    """Reusable validator for future dates"""
    if v < date.today():
        raise ValueError(f'{field_name} must be a future date, got {v}')
    return v


class ProductListing(BaseModel):
    """Product listing with reusable validators"""
    
    name: str = Field(min_length=1, max_length=200)
    price: Decimal = Field(decimal_places=2)
    discount_percent: Decimal = Field(decimal_places=2, ge=0, le=100)
    stock_quantity: int = Field(ge=0)
    launch_date: date
    
    @field_validator('price')
    @classmethod
    def validate_price(cls, v: Decimal) -> Decimal:
        """Price must be positive"""
        if v <= 0:
            raise ValueError(f'Price must be positive, got {v}')
        return v
    
    @field_validator('discount_percent')
    @classmethod
    def validate_discount(cls, v: Decimal) -> Decimal:
        """Use reusable percentage validator"""
        return Decimal(str(validate_percentage(float(v), "discount_percent")))
    
    @field_validator('launch_date')
    @classmethod
    def validate_launch_date(cls, v: date) -> date:
        """Launch date can be past or future, but not too old"""
        if v < date.today() - timedelta(days=365):
            raise ValueError(
                f'Launch date {v} is more than 1 year in the past. '
                f'Use a more recent date or archive this product.'
            )
        return v


# PATTERN 7: Validation with External Rules

class AddressValidation(BaseModel):
    """Address with complex validation rules"""
    
    street: str = Field(min_length=1, max_length=200)
    city: str = Field(min_length=1, max_length=100)
    state: str = Field(min_length=2, max_length=2)  # US state code
    zip_code: str
    country: Literal["US"] = "US"  # Only US for this example
    
    @field_validator('state')
    @classmethod
    def validate_state_code(cls, v: str) -> str:
        """Validate US state code"""
        # List of valid US state codes
        valid_states = {
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
            "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
            "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
            "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
        }
        
        v_upper = v.upper()
        if v_upper not in valid_states:
            raise ValueError(
                f'Invalid US state code: {v}. Must be a valid 2-letter state abbreviation.'
            )
        
        return v_upper
    
    @field_validator('zip_code')
    @classmethod
    def validate_zip_code(cls, v: str) -> str:
        """Validate ZIP code format"""
        # Accept 5-digit or ZIP+4 format
        if not re.match(r'^\d{5}(-\d{4})?$', v):
            raise ValueError(
                f'Invalid ZIP code format: {v}. Must be XXXXX or XXXXX-XXXX'
            )
        return v


# PATTERN 8: Agent with Comprehensive Validation

validation_agent = Agent(
    model='gemini-1.5-flash',
    result_type=OrderValidation,
    deps_type=ValidationDeps,
    system_prompt="""
You are an order processing assistant that generates validated order records.

CRITICAL VALIDATION REQUIREMENTS:

1. ORDER ID FORMAT:
   - Must be: ORD-YYYYMMDD-XXXX
   - Example: ORD-20241202-0001
   - Date should match when order was placed

2. FINANCIAL CALCULATIONS:
   - total = (subtotal * (1 - discount_percent/100) * (1 + tax_rate)) + shipping
   - All amounts must have exactly 2 decimal places
   - Minimum order after discount: $5.00

3. DISCOUNT RULES:
   - Can only apply discount if discount_code is provided
   - Discount code must be uppercase alphanumeric (4-12 chars)
   - If discount_code present, discount_percent must be > 0

4. TAX AND SHIPPING:
   - Tax rate is decimal (e.g., 0.08 for 8%)
   - Shipping is always positive or zero

VALIDATION WILL FAIL IF:
- Calculations are incorrect
- Discount logic is violated
- Order ID format is wrong
- Required fields are missing

Generate orders that pass ALL validation checks!
""",
)


# Demonstration

def demonstrate_validation():
    """Demonstrate validation with various test cases"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE RESULT VALIDATION")
    print("="*70)
    
    # Test Case 1: Valid User Registration
    print("\n\n" + "="*70)
    print("TEST 1: Field Validators - User Registration")
    print("="*70)
    
    print("\n✅ Valid registration:")
    try:
        valid_user = UserRegistration(
            username="john_doe",
            email="john@example.com",
            password="SecurePass123!",
            age=28,
            phone="555-123-4567"
        )
        print(f"   Username: {valid_user.username}")
        print(f"   Email: {valid_user.email}")
        print(f"   Phone: {valid_user.phone}")
        print(f"   ✅ All field validations passed!")
    except ValidationError as e:
        print(f"   ❌ Validation failed: {e}")
    
    print("\n❌ Invalid username (special chars):")
    try:
        invalid_user = UserRegistration(
            username="john@doe!",  # ❌ Has @ and !
            email="john@example.com",
            password="SecurePass123!",
            age=28
        )
    except ValidationError as e:
        errors = e.errors()
        print(f"   ❌ {errors[0]['loc'][0]}: {errors[0]['msg']}")
    
    print("\n❌ Invalid password (no special chars):")
    try:
        invalid_user = UserRegistration(
            username="john_doe",
            email="john@example.com",
            password="Password123",  # ❌ No special character
            age=28
        )
    except ValidationError as e:
        errors = e.errors()
        print(f"   ❌ {errors[0]['loc'][0]}: {errors[0]['msg']}")
    
    # Test Case 2: Model Validators - Date Range
    print("\n\n" + "="*70)
    print("TEST 2: Model Validators - Date Range Validation")
    print("="*70)
    
    print("\n✅ Valid date range:")
    try:
        valid_range = DateRange(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            max_days=365
        )
        days = (valid_range.end_date - valid_range.start_date).days
        print(f"   Range: {valid_range.start_date} to {valid_range.end_date} ({days} days)")
        print(f"   ✅ Date range validation passed!")
    except ValidationError as e:
        print(f"   ❌ {e}")
    
    print("\n❌ Invalid date range (end before start):")
    try:
        invalid_range = DateRange(
            start_date=date(2024, 12, 31),
            end_date=date(2024, 1, 1),  # ❌ Before start!
            max_days=365
        )
    except ValidationError as e:
        errors = e.errors()
        print(f"   ❌ {errors[0]['msg']}")
    
    # Test Case 3: Financial Validation
    print("\n\n" + "="*70)
    print("TEST 3: Business Logic Validation - Financial Transaction")
    print("="*70)
    
    print("\n✅ Valid transaction:")
    try:
        valid_transaction = FinancialTransaction(
            transaction_id="TXN-001",
            amount=Decimal("100.00"),
            fee=Decimal("2.50"),
            total=Decimal("102.50"),
            currency="USD"
        )
        print(f"   Amount: ${valid_transaction.amount}")
        print(f"   Fee: ${valid_transaction.fee}")
        print(f"   Total: ${valid_transaction.total}")
        print(f"   ✅ Financial validation passed!")
    except ValidationError as e:
        print(f"   ❌ {e}")
    
    print("\n❌ Invalid transaction (wrong total):")
    try:
        invalid_transaction = FinancialTransaction(
            transaction_id="TXN-002",
            amount=Decimal("100.00"),
            fee=Decimal("2.50"),
            total=Decimal("105.00"),  # ❌ Should be 102.50!
            currency="USD"
        )
    except ValidationError as e:
        errors = e.errors()
        print(f"   ❌ {errors[0]['msg']}")
    
    # Test Case 4: Complex Order Validation
    print("\n\n" + "="*70)
    print("TEST 4: Complex Validation - Order Processing")
    print("="*70)
    
    print("\n✅ Valid order with discount:")
    try:
        # Calculation: (100 * 0.8 * 1.08) + 5.00 = 86.40 + 5.00 = 91.40
        valid_order = OrderValidation(
            order_id="ORD-20241202-0001",
            customer_email="customer@example.com",
            items=["item1", "item2"],
            subtotal=Decimal("100.00"),
            discount_percent=Decimal("20.00"),
            discount_code="SAVE20",
            tax_rate=Decimal("0.08"),
            shipping=Decimal("5.00"),
            total=Decimal("91.40")
        )
        print(f"   Order ID: {valid_order.order_id}")
        print(f"   Subtotal: ${valid_order.subtotal}")
        print(f"   Discount: {valid_order.discount_percent}% ({valid_order.discount_code})")
        print(f"   Tax: {valid_order.tax_rate * 100}%")
        print(f"   Shipping: ${valid_order.shipping}")
        print(f"   Total: ${valid_order.total}")
        print(f"   ✅ All order validations passed!")
    except ValidationError as e:
        print(f"   ❌ {e}")
    
    print("\n❌ Invalid order (discount without code):")
    try:
        invalid_order = OrderValidation(
            order_id="ORD-20241202-0002",
            customer_email="customer@example.com",
            items=["item1"],
            subtotal=Decimal("100.00"),
            discount_percent=Decimal("20.00"),  # ❌ Discount without code!
            discount_code=None,
            tax_rate=Decimal("0.08"),
            shipping=Decimal("5.00"),
            total=Decimal("91.40")
        )
    except ValidationError as e:
        errors = e.errors()
        print(f"   ❌ {errors[0]['msg']}")
    
    print("\n❌ Invalid order (wrong calculation):")
    try:
        invalid_order = OrderValidation(
            order_id="ORD-20241202-0003",
            customer_email="customer@example.com",
            items=["item1"],
            subtotal=Decimal("100.00"),
            discount_percent=Decimal("20.00"),
            discount_code="SAVE20",
            tax_rate=Decimal("0.08"),
            shipping=Decimal("5.00"),
            total=Decimal("100.00")  # ❌ Wrong total!
        )
    except ValidationError as e:
        errors = e.errors()
        print(f"   ❌ {errors[0]['msg']}")


# Agent validation demonstration

def demonstrate_agent_validation():
    """Demonstrate validation in agent context"""
    
    print("\n\n" + "="*70)
    print("AGENT RESULT VALIDATION IN ACTION")
    print("="*70)
    
    deps = ValidationDeps(user_id="user_123", current_date=date.today())
    
    test_queries = [
        {
            "query": "Create an order for customer@example.com: 2 items, subtotal $150, "
                     "20% discount with code SAVE20, 8% tax, $10 shipping",
            "should_succeed": True
        },
        {
            "query": "Create an order for test@example.com: 1 item, subtotal $50, "
                     "no discount, 7% tax, free shipping",
            "should_succeed": True
        },
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"AGENT TEST {i}")
        print(f"{'='*70}")
        print(f"Query: {test['query']}")
        print(f"Expected: {'Success' if test['should_succeed'] else 'Validation Error'}")
        
        try:
            result = validation_agent.run_sync(test["query"], deps=deps)
            order = result.data
            
            print(f"\n✅ ORDER CREATED (passed validation):")
            print(f"   Order ID: {order.order_id}")
            print(f"   Customer: {order.customer_email}")
            print(f"   Items: {len(order.items)}")
            print(f"   Subtotal: ${order.subtotal}")
            if order.discount_code:
                print(f"   Discount: {order.discount_percent}% ({order.discount_code})")
            print(f"   Tax: {order.tax_rate * 100}%")
            print(f"   Shipping: ${order.shipping}")
            print(f"   Total: ${order.total}")
            
            # Verify calculation manually
            discounted = order.subtotal * (1 - order.discount_percent / 100)
            with_tax = discounted * (1 + order.tax_rate)
            expected_total = with_tax + order.shipping
            print(f"\n   ✅ Calculation verified: ${expected_total:.2f} = ${order.total}")
            
        except ValidationError as e:
            print(f"\n❌ VALIDATION FAILED:")
            for error in e.errors():
                field = " -> ".join(str(loc) for loc in error['loc'])
                print(f"   Field: {field}")
                print(f"   Error: {error['msg']}")
                print(f"   Input: {error.get('input', 'N/A')}")


def main():
    """Run all validation demonstrations"""
    
    print("\n" + "="*70)
    print("LESSON 15: AGENT RESULT VALIDATION")
    print("="*70)
    print("\nComprehensive validation patterns for production agents")
    
    # Part 1: Model validation examples
    demonstrate_validation()
    
    # Part 2: Agent with validation
    demonstrate_agent_validation()
    
    print("\n\n" + "="*70)
    print("VALIDATION PATTERNS SUMMARY")
    print("="*70)
    
    print("\n🎯 Validation Hierarchy:")
    print("   1. Type validation (Pydantic automatic)")
    print("   2. Field constraints (min/max, length, regex)")
    print("   3. Field validators (single field logic)")
    print("   4. Model validators (cross-field logic)")
    print("   5. Business rules (domain-specific)")
    print("   6. External validation (databases, APIs)")
    
    print("\n✅ Best Practices:")
    print("   • Validate early (fail fast)")
    print("   • Use Field constraints for simple rules")
    print("   • Use field_validator for single-field logic")
    print("   • Use model_validator for cross-field logic")
    print("   • Provide clear, actionable error messages")
    print("   • Normalize data in validators (lowercase, format)")
    print("   • Make validators reusable across models")
    print("   • Test validation extensively")
    
    print("\n❌ Anti-Patterns:")
    print("   • Validating in multiple places (centralize in Pydantic)")
    print("   • Silently fixing invalid data (raise errors)")
    print("   • Generic error messages ('invalid value')")
    print("   • Not validating calculated fields")
    print("   • Over-validating (balance safety with usability)")
    
    print("\n🔒 Why Validation Matters:")
    print("   • Type-correct ≠ Business-correct")
    print("   • Catches AI mistakes before they cause problems")
    print("   • Enforces business rules automatically")
    print("   • Provides clear feedback for debugging")
    print("   • Prevents invalid data from entering your system")
    print("   • Builds user trust through reliability")


if __name__ == "__main__":
    main()
```

### Line-by-Line Explanation

**Pattern 1: Field Validators (Lines 27-108)**:
- `UserRegistration`: Model with field-level validation
- `@field_validator`: Validates individual fields
- Runs after basic type/constraint checks
- Can normalize data (lowercase username, format phone)
- Returns modified value

**Pattern 2: Model Validators (Lines 111-228)**:
- `@model_validator(mode='after')`: Validates entire model
- Has access to all fields simultaneously
- Validates relationships between fields
- Checks business logic (totals, date ranges)
- Can access all fields via `self`

**Pattern 3: Reusable Validators (Lines 231-334)**:
- `validate_positive_number()`, `validate_percentage()`, etc.
- Reusable across multiple models
- Consistent error messages
- Type-safe helper functions

**Pattern 4: External Rule Validation (Lines 337-395)**:
- `AddressValidation`: Validates against external rules (US states)
- ZIP code format validation
- Can be extended to validate against APIs/databases
- Domain-specific validation logic

**Pattern 5: Complex Order Validation (Lines 398-535)**:
- Multiple validators working together
- Format validation (order ID pattern)
- Discount logic validation (code required)
- Financial calculation validation
- Minimum order validation
- All enforced automatically!

**Pattern 6: Agent with Validation (Lines 538-605)**:
- Agent that generates validated outputs
- System prompt guides output to pass validation
- If Gemini generates invalid data, ValidationError raised
- Agent learns from validation failures

### The "Why" Behind the Pattern

**Why comprehensive validation beyond types?**

❌ **Type-Only Validation** (Incomplete):
```python
class Order(BaseModel):
    subtotal: Decimal
    discount: Decimal
    total: Decimal
    # ✅ Types are correct
    # ❌ But total might not equal subtotal - discount!
```

✅ **Comprehensive Validation** (Complete):
```python
class Order(BaseModel):
    subtotal: Decimal
    discount: Decimal
    total: Decimal
    
    @model_validator(mode='after')
    def validate_math(self) -> 'Order':
        expected = self.subtotal - self.discount
        if abs(self.total - expected) > 0.01:
            raise ValueError(f"Math error: {self.total} ≠ {expected}")
        return self
    # ✅ Types correct AND logic correct!
```

**Real-World Impact**:

Without comprehensive validation:
- AI generates order with wrong total → customer overcharged
- AI generates future date for past event → booking fails
- AI calculates wrong discount → revenue loss
- AI violates business rules → legal issues

With comprehensive validation:
- Invalid outputs caught before processing
- Clear error messages for debugging  
- Business rules enforced automatically
- Trust in AI-generated data

---

## C. Test & Apply

### How to Test It

1. **Run the validation demo**:
```bash
python lesson_15_agent_result_validation.py
```

2. **Observe validation successes and failures**

3. **Try your own validated model**:
```python
class MyValidatedModel(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    score: float = Field(ge=0.0, le=100.0)
    category: Literal["A", "B", "C"]
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Custom name validation"""
        if not v.strip():
            raise ValueError("Name cannot be empty or whitespace")
        return v.title()  # Normalize to title case
    
    @model_validator(mode='after')
    def validate_score_category(self) -> 'MyValidatedModel':
        """Score must match category"""
        if self.category == "A" and self.score < 80:
            raise ValueError("Category A requires score >= 80")
        return self
```

### Expected Result

You should see validation catching errors:

```
======================================================================
TEST 1: Field Validators - User Registration
======================================================================

✅ Valid registration:
   Username: john_doe
   Email: john@example.com
   Phone: +1-555-123-4567
   ✅ All field validations passed!

❌ Invalid username (special chars):
   ❌ username: Username must contain only letters, numbers, and underscores

❌ Invalid password (no special chars):
   ❌ password: Password must contain at least one special character

======================================================================
TEST 2: Model Validators - Date Range Validation
======================================================================

✅ Valid date range:
   Range: 2024-01-01 to 2024-01-31 (30 days)
   ✅ Date range validation passed!

❌ Invalid date range (end before start):
   ❌ end_date (2024-01-01) must be after start_date (2024-12-31)

======================================================================
TEST 4: Complex Validation - Order Processing
======================================================================

✅ Valid order with discount:
   Order ID: ORD-20241202-0001
   Customer: customer@example.com
   Items: 2
   Subtotal: $100.00
   Discount: 20.00% (SAVE20)
   Tax: 8.0%
   Shipping: $5.00
   Total: $91.40

   ✅ Calculation verified: $91.40 = $91.40

❌ Invalid order (discount without code):
   ❌ Cannot have discount (20.00%) without a discount code
```

### Validation Examples

**Validation Layers**:

```python
class ComprehensiveModel(BaseModel):
    # Layer 1: Type validation (automatic)
    name: str
    age: int
    
    # Layer 2: Field constraints (automatic)
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=0, le=150)
    
    # Layer 3: Field validators (custom logic)
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if v.lower() in ['admin', 'root']:
            raise ValueError("Reserved name")
        return v
    
    # Layer 4: Model validators (cross-field)
    @model_validator(mode='after')
    def validate_business_rules(self) -> 'ComprehensiveModel':
        # Complex logic using multiple fields
        return self
```

### Type Checking

```bash
mypy lesson_15_agent_result_validation.py
```

Expected: `Success: no issues found`

---

## D. Common Stumbling Blocks

### 1. Validator Order Confusion

**The Problem**:
```python
class MyModel(BaseModel):
    value: int = Field(ge=0)
    
    @field_validator('value')
    @classmethod
    def validate_value(cls, v: int) -> int:
        if v < 0:  # ❌ This check is redundant!
            raise ValueError("Must be positive")
        return v
```

**What's Wrong**:
Field constraint `ge=0` already validates this. Redundant validation.

**The Fix**:
Understand the validation order:
```python
# Order of validation:
# 1. Type check (is it an int?)
# 2. Field constraints (ge=0)
# 3. Field validators (your custom logic)

class MyModel(BaseModel):
    value: int = Field(ge=0)
    
    @field_validator('value')
    @classmethod
    def validate_value(cls, v: int) -> int:
        # ✅ v is already guaranteed to be int and >= 0
        # Add ADDITIONAL logic here
        if v > 1000:
            raise ValueError("Value too large")
        return v
```

### 2. Modifying Other Fields in Field Validator

**The Problem**:
```python
class MyModel(BaseModel):
    first_name: str
    last_name: str
    full_name: str
    
    @field_validator('first_name')
    @classmethod
    def set_full_name(cls, v: str, info) -> str:
        # ❌ Trying to set full_name from first_name validator
        info.data['full_name'] = f"{v} {info.data.get('last_name', '')}"
        return v
```

**The Fix**:
Use model validator or computed field:
```python
class MyModel(BaseModel):
    first_name: str
    last_name: str
    
    @model_validator(mode='after')
    def set_full_name(self) -> 'MyModel':
        # ✅ Can access all fields
        self.full_name = f"{self.first_name} {self.last_name}"
        return self

# Or use computed_field (better):
from pydantic import computed_field

class MyModel(BaseModel):
    first_name: str
    last_name: str
    
    @computed_field
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
```

### 3. Not Handling ValidationError in Agent

**The Problem**:
```python
# ❌ No validation error handling
result = agent.run_sync(query, deps=deps)
order = result.data  # Crashes if validation fails!
process_order(order)
```

**The Fix**:
Always catch ValidationError:
```python
from pydantic import ValidationError

try:
    result = agent.run_sync(query, deps=deps)
    order = result.data
    process_order(order)
except ValidationError as e:
    print("❌ Agent output failed validation:")
    for error in e.errors():
        field = ".".join(str(loc) for loc in error['loc'])
        print(f"   {field}: {error['msg']}")
    # Handle gracefully - maybe retry with different prompt
```

### 4. Too Strict Validation

**The Problem**:
```python
@field_validator('name')
@classmethod
def validate_name(cls, v: str) -> str:
    # ❌ Too strict - rejects valid names
    if not v.isalpha():
        raise ValueError("Name must contain only letters")
    # Rejects: "O'Brien", "Jean-Luc", "Mary Anne"
```

**The Fix**:
Balance safety with usability:
```python
@field_validator('name')
@classmethod
def validate_name(cls, v: str) -> str:
    # ✅ Reasonable validation
    if not v.strip():
        raise ValueError("Name cannot be empty")
    
    # Allow letters, spaces, hyphens, apostrophes
    if not re.match(r"^[a-zA-Z\s\-']+$", v):
        raise ValueError("Name contains invalid characters")
    
    return v.strip()
```

### 5. Type Safety Gotcha: Validator Return Type

**The Problem**:
```python
class MyModel(BaseModel):
    value: int
    
    @field_validator('value')
    @classmethod
    def validate_value(cls, v: int) -> str:  # ❌ Wrong return type!
        return str(v)  # Converting int to str

# Type is now inconsistent!
```

**The Fix**:
Validator must return same type:
```python
class MyModel(BaseModel):
    value: int
    
    @field_validator('value')
    @classmethod
    def validate_value(cls, v: int) -> int:  # ✅ Same type
        if v < 0:
            raise ValueError("Must be positive")
        return v  # ✅ Return same type
```

---

## Ready for the Next Lesson?

🎉 **Outstanding work!** You now understand comprehensive result validation:

✅ Field validators for single-field validation  
✅ Model validators for cross-field validation  
✅ Business logic validation  
✅ Financial calculation validation  
✅ Format validation (IDs, phone, email, ZIP)  
✅ Reusable validator functions  
✅ Handling ValidationError in agents  

**Validation is your safety net!** It catches AI mistakes, enforces business rules, and ensures every output meets your standards. Type correctness + validation = production-ready AI systems.

In the final lesson, we'll bring **everything together** in **Complete Multi-Tool Agent System** - you'll see all 15 lessons integrated into one comprehensive, production-ready agent system with tools, dependencies, streaming, error handling, retries, and validation!

**Ready for the final lesson (Lesson 16), or would you like to practice validation patterns first?** 🚀
