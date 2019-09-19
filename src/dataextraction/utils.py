import re
from typing import List

DOCSTRING_REGEX_TOKENIZER = re.compile(r"[^\s,'\"`.():\[\]=*;>{\}+-/\\]+|\\+|\.+|\(\)|{\}|\[\]|\(+|\)+|:+|\[+|\]+|{+|\}+|=+|\*+|;+|>+|\++|-+|/+")


def tokenize_docstring_from_string(docstr: str) -> List[str]:
    return [t for t in DOCSTRING_REGEX_TOKENIZER.findall(docstr) if t is not None and len(t) > 0]