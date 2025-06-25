# Sphinx Documentation System - Context7 Cache

## Core API Documentation Features

### Automatic API Documentation Generation
```python
# sphinx-apidoc command-line usage
sphinx-apidoc -f -o docs/source projectdir
```

### Configuration for apidoc Module
```python
apidoc_modules = [
    {
        'path': 'path/to/module', 
        'destination': 'source/',
        'exclude_patterns': ['**/test*'],
        'max_depth': 4,
        'follow_links': False,
        'separate_modules': False,
        'include_private': False,
        'no_headings': False,
        'module_first': False,
        'implicit_namespaces': False,
        'automodule_options': {
            'members', 'show-inheritance', 'undoc-members'
        },
    },
]
```

## Core Documentation Directives

### Automatic Documentation Directives
```rst
.. automodule:: module_name
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ClassName
   :members:
   :inherited-members:

.. autofunction:: function_name

.. autosummary::
   :toctree: generated
   :recursive:
   
   package_name
```

### Python Domain API Documentation
```rst
.. py:module:: module_name

.. py:function:: function_name(param1, param2)
   
   Function description
   
   :param param1: Description of param1
   :type param1: str
   :param param2: Description of param2
   :type param2: int
   :returns: Description of return value
   :rtype: bool

.. py:class:: ClassName
   
   Class description
   
   .. py:method:: method_name(self, arg)
      
      Method description
```

## Interactive Documentation Features

### Cross-references and Linking
```rst
:py:func:`module.function_name`
:py:class:`module.ClassName`
:py:meth:`ClassName.method_name`
:py:attr:`ClassName.attribute_name`
```

### Code Examples with Syntax Highlighting
```rst
.. code-block:: python
   :linenos:
   :emphasize-lines: 3,5
   
   def example_function():
       """Example function docstring."""
       result = some_computation()
       print("Processing...")
       return result
```

## Builder Configuration

### HTML Builder for Interactive Docs
```python
# conf.py configuration
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': 'UA-XXXXXXX-1',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'white',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
```

### Extensions for Enhanced Documentation
```python
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'inherited-members': True,
    'member-order': 'bysource',
}

# Napoleon configuration for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
```

## Advanced API Documentation Patterns

### Complex Class Documentation
```rst
.. autoclass:: ComplexClass
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:
   
   .. automethod:: __init__
   
   .. rubric:: Methods
   
   .. autosummary::
      :toctree:
      
      method1
      method2
      method3
```

### Module-level Documentation with Examples
```rst
Module Title
============

.. automodule:: module_name
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Usage:

.. code-block:: python

   from module_name import ClassName
   
   instance = ClassName()
   result = instance.method()

Advanced Usage:

.. code-block:: python

   # More complex example
   with ClassName() as instance:
       result = instance.advanced_method(param1, param2)
```

## Interactive Features and Best Practices

### Intersphinx for Cross-Project Linking
```python
# Link to external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}
```

### TODO and Development Notes
```rst
.. todo::
   
   Add more examples for this function.
   
.. note::
   
   This API is experimental and may change in future versions.

.. warning::
   
   This function can be slow for large datasets.
```

### API Versioning and Deprecation
```rst
.. versionadded:: 1.2.0
   
.. versionchanged:: 1.3.0
   Added new parameter `timeout`.
   
.. deprecated:: 1.4.0
   Use :func:`new_function` instead.
```

## Developer Experience Enhancements

1. **Searchable API Reference**
   - Full-text search across all documentation
   - Object-specific search filters
   - Quick access to API elements

2. **Interactive Examples**
   - Runnable code blocks
   - Live API testing interface
   - Interactive tutorials

3. **Source Code Integration**
   - Direct links to source code
   - Syntax highlighting
   - Version control integration

4. **Multi-format Output**
   - HTML for web viewing
   - PDF for offline reading
   - EPUB for mobile devices

## Build and Deployment

### Automatic Documentation Build
```bash
# Build documentation
sphinx-build -b html source build/html

# Clean build
sphinx-build -E -a -b html source build/html

# Check for broken links
sphinx-build -b linkcheck source build/linkcheck
```

### GitHub Actions Integration
```yaml
name: docs
on:
  push:
    branches: [main]
jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install sphinx sphinx-rtd-theme
    - name: Build docs
      run: |
        sphinx-build -b html docs docs/_build/html
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: docs/_build/html
```
