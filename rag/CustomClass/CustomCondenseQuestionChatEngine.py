import logging
from threading import Thread
from typing import Any, Listm, Optional

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)
from llama_index.core.bridge.pydantic import BaseModel

logger = logging.getLogger(__name__)

class CondenseResponse(BaseModel):
