from __future__ import annotations

import inspect


def get_all_argument_names(func):
    sig = inspect.signature(func)
    return [
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]


def remove_irrelevant_arguments(func, kwargs):
    return {k: v for k, v in kwargs.items() if k in get_all_argument_names(func)}
