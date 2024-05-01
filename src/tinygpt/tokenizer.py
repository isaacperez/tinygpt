import re
from typing import Optional, Union


class RegexPatterns:
    GPT4 = r"""
        # This is an approximation of the pattern used by the tokenizer of GPT-4 but using re library instead of regex.
        (?i:[sdmt]|ll|ve|re)| # Contractions, case insensitive
        [^\r\n\w\d]?+\w+|     # Words, allowing an optional non-alphabetic/non-numeric character at the start
        [^\s\w\d]++[\r\n]*|   # Sequences of non-letters, non-digits, and non-spaces, possibly followed by new lines
        \s*[\r\n]|            # Spaces followed by new lines
        \s+(?!\S)|            # Spaces at the end of a line
        \s+                   # One or more spaces
        """


def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):  # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1

    return newids