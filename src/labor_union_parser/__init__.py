"""
Labor Union Parser - Extract affiliation and designation from union names.

Example:
    >>> from labor_union_parser import Extractor
    >>> extractor = Extractor()
    >>> extractor.extract("SEIU Local 1199")
    {'is_union': True, 'union_score': 0.99, 'affiliation': 'SEIU',
     'affiliation_unrecognized': False, 'designation': '1199', 'aff_score': 0.99}

    >>> from labor_union_parser import lookup_fnum
    >>> lookup_fnum("SEIU", "1199")
    [31847, 69557, ...]
"""

from .extractor import Extractor, lookup_fnum

__version__ = "0.1.0"
__all__ = ["Extractor", "lookup_fnum"]
