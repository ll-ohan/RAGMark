from typing import Any

class LangDetectException(Exception):
    pass

def detect(text: str) -> str: ...
def detect_langs(text: str) -> list[Any]: ...
