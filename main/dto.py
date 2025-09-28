from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class QueryDTO:
    user_question: str
    retrieved_context: str = ""
    few_shot_examples: Dict[str, str] = field(default_factory=dict)
    matched_indices: List[int] = field(default_factory=list)
    answer: str = ""
    source_examples: List[str] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    step_log: List[str] = field(default_factory=list)   # optional
    status: str = "pending"  # pending | success | retrieval_failed | llm_failed

