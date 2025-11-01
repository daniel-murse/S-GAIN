"""
Contains code to parse modalities for dynamic sparse training.

Since underscores are already used by S-GAIN for the names of files to extract information, to prevent issues/fragility, we
use "@" and "+" for separating syntax elements for modality parameters. 

The idea is that a modality consists of sections, which wrap a parameter list in "@", where parameters inside are delimited by "+".
This allows unambiguous parsing of arguments and information from the modalities. The recommended use is to use the first section as a sort of
polymorphic discriminator for how to parse/interpret the modality, then to use as many other sections as needed.

For example, the currently implemented parsing requires modalities to be prefixed with a section "constant_sparsity",
as that is what was also implemented in the project. New ways of parsing modalities can either modify the way this parsing occurs or to use a new discriminator.

For modalities prefixed with "constant_sparsity", two sections are expected, one describing the initialisation of the mask, and the other one describing the
dynamic part of training.
"""

import re

full_pattern = re.compile(
    r'^(?:[A-Za-z0-9_.+-]+\@(?:[A-Za-z0-9_.+-]+(?:\+[A-Za-z0-9_.+-]+)*)?\@)+$'
)

extract_pattern = re.compile(
    r'([A-Za-z0-9_.+-]+)\@(?:([A-Za-z0-9_.+-]+(?:\+[A-Za-z0-9_.+-]+)*))?\@'
)


def tokenise_modality(s):
    """
    Parses a modality string containing named sections with arguments into a list of tuples of (section_name, section_arguments)
    All arguments are strings.
    """
    if not full_pattern.match(s):
        raise ValueError("Invalid format", s)
    
    matches = extract_pattern.findall(s)
    sections = []
    for name, params_str in matches:
        params = params_str.split('+') if params_str else []
        sections.append((name, params))
    return sections

