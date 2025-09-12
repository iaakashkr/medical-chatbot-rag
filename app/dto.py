from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class QueryDTO:
    # Input
    user_question: str
    
    # Retrieval
    retrieved_context: str = ""
    few_shot_examples: Dict[str, str] = field(default_factory=dict)
    matched_indices: List[int] = field(default_factory=list)
    
    # LLM
    answer: str = ""
    source_examples: List[str] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
