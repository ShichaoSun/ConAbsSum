#!/usr/bin/env python
# copying from https://github.com/huggingface/transformers/tree/v4.0.1/examples
# fix some bugs

from filelock import FileLock


try:
    import nltk

    NLTK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NLTK_AVAILABLE = False

if NLTK_AVAILABLE:
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    if "<n>" in x:
        return x.replace("<n>", "\n")  # remove pegasus newline char
    else:
        return x
