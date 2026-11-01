"""Pydantic schemas for AST analysis and code inspection."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class VariableInfo(BaseModel):
    name: str
    line_number: int
    is_assignment: bool = True
    assigned_value_type: Optional[str] = None
    target_scope: str = "global"

    model_config = ConfigDict(from_attributes=True)


class LoopInfo(BaseModel):
    loop_type: str  # "for" or "while"
    line_number: int
    end_line_number: Optional[int] = None
    target_variable: Optional[str] = None
    iterable_expression: Optional[str] = None
    has_break: bool = False
    has_continue: bool = False
    nesting_depth: int = 1

    model_config = ConfigDict(from_attributes=True)


class ConditionInfo(BaseModel):
    line_number: int
    test_expression: str
    has_elif: bool = False
    has_else: bool = False
    nesting_depth: int = 1

    model_config = ConfigDict(from_attributes=True)


class FunctionInfo(BaseModel):
    name: str
    line_number: int
    end_line_number: Optional[int] = None
    arguments: List[str] = Field(default_factory=list)
    has_return: bool = False
    return_line_numbers: List[int] = Field(default_factory=list)
    is_recursive: bool = False
    docstring: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class VivaCandidateConcept(BaseModel):
    concept_key: str
    concept_name: str
    line_number: int
    focal_code: str
    description: str
    suggested_difficulty: str = "medium"

    model_config = ConfigDict(from_attributes=True)


class ASTAnalysisResult(BaseModel):
    is_syntax_valid: bool
    syntax_error_details: Optional[Dict[str, Any]] = None
    
    # Structural features
    variables: List[VariableInfo] = Field(default_factory=list)
    loops: List[LoopInfo] = Field(default_factory=list)
    conditions: List[ConditionInfo] = Field(default_factory=list)
    functions: List[FunctionInfo] = Field(default_factory=list)
    
    # Aggregated metrics
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    cyclomatic_complexity: int = 1
    max_nesting_depth: int = 0
    builtin_calls: List[str] = Field(default_factory=list)
    
    # Concepts identified for Viva generation
    candidate_concepts: List[VivaCandidateConcept] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class ASTAnalysisRequest(BaseModel):
    code: str = Field(..., min_length=1, description="Python source code to parse")
