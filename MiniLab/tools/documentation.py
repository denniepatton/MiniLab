"""
Dynamic Documentation Retrieval Tool

Inspired by CellVoyager's approach: parse generated code with AST to extract
function calls, then retrieve actual docstrings from libraries to ground
the LLM in real API signatures.

This prevents hallucinated parameters and incorrect API usage.
"""

import ast
import inspect
import importlib
from typing import Dict, List, Set, Optional


# Libraries we support with documentation retrieval
SUPPORTED_LIBRARIES = {
    "pd": "pandas",
    "pandas": "pandas",
    "np": "numpy",
    "numpy": "numpy",
    "plt": "matplotlib.pyplot",
    "matplotlib": "matplotlib",
    "sns": "seaborn",
    "seaborn": "seaborn",
    "scipy": "scipy",
    "sklearn": "sklearn",
    "lifelines": "lifelines",
    "scanpy": "scanpy",
    "sc": "scanpy",
}

# Package allowlist for code generation prompts
AVAILABLE_PACKAGES = (
    "pandas, numpy, matplotlib, seaborn, scipy, sklearn, lifelines, "
    "scanpy, anndata, statsmodels, pillow"
)


def extract_function_calls(code: str) -> Set[str]:
    """
    Parse Python code and extract all function/method calls.
    
    Returns a set of call names like:
    - 'pd.read_csv'
    - 'plt.figure'
    - 'sklearn.model_selection.train_test_split'
    - 'df.groupby'
    
    Args:
        code: Python source code string
        
    Returns:
        Set of function call names found in the code
    """
    calls = set()
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If code has syntax errors, return empty set
        return calls
    
    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            call_name = _get_call_name(node.func)
            if call_name:
                calls.add(call_name)
            self.generic_visit(node)
    
    CallVisitor().visit(tree)
    return calls


def _get_call_name(node) -> Optional[str]:
    """Extract the full dotted name from a call's func node."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        value_name = _get_call_name(node.value)
        if value_name:
            return f"{value_name}.{node.attr}"
        return node.attr
    return None


def _resolve_module_path(call_name: str) -> Optional[tuple]:
    """
    Resolve a call name to (module, object_path).
    
    Examples:
        'pd.read_csv' -> ('pandas', 'read_csv')
        'plt.figure' -> ('matplotlib.pyplot', 'figure')
        'sklearn.model_selection.train_test_split' -> ('sklearn.model_selection', 'train_test_split')
    """
    parts = call_name.split('.')
    if not parts:
        return None
    
    # Check if first part is an alias
    first = parts[0]
    if first in SUPPORTED_LIBRARIES:
        module_name = SUPPORTED_LIBRARIES[first]
        obj_path = '.'.join(parts[1:]) if len(parts) > 1 else None
        return (module_name, obj_path)
    
    return None


def get_docstring(call_name: str) -> Optional[str]:
    """
    Get the docstring for a function/method call.
    
    Args:
        call_name: Dotted call name like 'pd.read_csv' or 'plt.figure'
        
    Returns:
        Docstring if found, None otherwise
    """
    resolved = _resolve_module_path(call_name)
    if not resolved:
        return None
    
    module_name, obj_path = resolved
    
    try:
        module = importlib.import_module(module_name)
        
        if obj_path:
            obj = module
            for attr in obj_path.split('.'):
                obj = getattr(obj, attr, None)
                if obj is None:
                    return None
        else:
            obj = module
        
        doc = inspect.getdoc(obj)
        return doc if doc else None
        
    except (ImportError, AttributeError, TypeError):
        return None


def get_documentation_for_code(code: str, max_chars: int = 8000) -> str:
    """
    Extract function calls from code and retrieve their documentation.
    
    This grounds the LLM in actual API signatures, preventing hallucinated
    parameters and incorrect usage.
    
    Args:
        code: Python source code
        max_chars: Maximum characters to return (to manage context window)
        
    Returns:
        Formatted string with function signatures and docstrings
    """
    calls = extract_function_calls(code)
    
    # Filter to only supported library calls
    relevant_calls = []
    for call in calls:
        parts = call.split('.')
        if parts and parts[0] in SUPPORTED_LIBRARIES:
            relevant_calls.append(call)
    
    # Sort for consistent output
    relevant_calls = sorted(set(relevant_calls))
    
    docs = []
    total_chars = 0
    
    for call in relevant_calls:
        doc = get_docstring(call)
        if doc:
            # Truncate individual docs if too long
            if len(doc) > 1500:
                doc = doc[:1500] + "..."
            
            entry = f"### {call}\n{doc}\n"
            
            if total_chars + len(entry) > max_chars:
                docs.append(f"... ({len(relevant_calls) - len(docs)} more functions truncated)")
                break
            
            docs.append(entry)
            total_chars += len(entry)
    
    if not docs:
        return ""
    
    return "## API Documentation\n\n" + "\n".join(docs)


def get_package_constraint_prompt() -> str:
    """
    Get the package constraint text for code generation prompts.
    
    Returns:
        Formatted constraint text
    """
    return f"""PACKAGE CONSTRAINTS:
You can ONLY use these packages: {AVAILABLE_PACKAGES}
Do NOT import or use any other packages. If you need functionality not in these packages,
find an alternative using only the allowed packages."""


# Convenience function for common patterns
def get_common_api_docs() -> Dict[str, str]:
    """
    Get documentation for commonly used functions to include in prompts.
    
    Returns cached docs for frequently needed functions.
    """
    common_functions = [
        "pd.read_csv",
        "pd.DataFrame.groupby",
        "pd.DataFrame.merge",
        "np.random.seed",
        "plt.figure",
        "plt.savefig",
        "plt.subplots",
        "sns.heatmap",
        "sns.boxplot",
        "scipy.stats.ttest_ind",
        "scipy.stats.mannwhitneyu",
        "sklearn.model_selection.train_test_split",
        "sklearn.preprocessing.StandardScaler",
        "lifelines.KaplanMeierFitter",
        "lifelines.CoxPHFitter",
    ]
    
    docs = {}
    for func in common_functions:
        doc = get_docstring(func)
        if doc:
            # Just get the first part (signature + brief description)
            lines = doc.split('\n')
            brief = '\n'.join(lines[:20]) if len(lines) > 20 else doc
            docs[func] = brief
    
    return docs


if __name__ == "__main__":
    # Test the module
    test_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

df = pd.read_csv('data.csv')
df_grouped = df.groupby('category').mean()

kmf = KaplanMeierFitter()
kmf.fit(df['time'], df['event'])
kmf.plot_survival_function()
plt.savefig('survival.png')
"""
    
    print("Extracted calls:")
    calls = extract_function_calls(test_code)
    for call in sorted(calls):
        print(f"  - {call}")
    
    print("\nDocumentation:")
    docs = get_documentation_for_code(test_code, max_chars=3000)
    print(docs[:2000])
