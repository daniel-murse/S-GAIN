"""
Contains code to parse modalities for dynamic sparse training.
The general idea is
Section[param1;param2;]OtherSection[]LastSection[param]

We have sections which are letters, then params which can be letters or numbers separated by semicolons
"""

import re

full_pattern = re.compile(
    r'^(?:[A-Za-z0-9_+-]+\@(?:[A-Za-z0-9_+-]+(?:\+[A-Za-z0-9_+-]+)*)?\@)+$'
)

extract_pattern = re.compile(
    r'([A-Za-z0-9_+-]+)\@(?:([A-Za-z0-9_+-]+(?:\+[A-Za-z0-9_+-]+)*))?\@'
)

def tokenise_modality(s):
    """
    Parses a string like "Section[param1;param2;]OtherSection[]LastSection[param]" (^(?:[A-Za-z]+\@[A-Za-z0-9]+(?:;[A-Za-z0-9]+)*\@)+$)
    Into a list of tuples [(Section, params)]
    """
    if not full_pattern.match(s):
        raise ValueError("Invalid format", s)
    
    matches = extract_pattern.findall(s)
    sections = []
    for name, params_str in matches:
        params = params_str.split('+') if params_str else []
        sections.append((name, params))
    return sections

