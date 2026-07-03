"""Public database API for the learning-support platform.

Keeping exports here gives the rest of the application one stable import path,
while the implementation remains split into focused modules.
"""

from .database import DatabaseManager
from .models import EmotionRecord, User

__all__ = ["DatabaseManager", "EmotionRecord", "User"]

