from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ChatSession:
    """Conversation state for the demo chatbot."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Any] = field(default_factory=list)
    uploaded_doc_ids: List[str] = field(default_factory=list)
    scratchpad: Dict[str, str] = field(default_factory=dict)

    def clear_scratchpad(self) -> None:
        """Clear all within-turn working memory."""
        self.scratchpad.clear()
