"""edge_fhir_hybrid package.

Expose the Flask application factory for use by `python -m app.server` or
external WSGI servers.
"""
__version__ = "0.1.0"

from .server import create_app  # re-export the app factory
