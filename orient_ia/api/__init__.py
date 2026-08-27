"""Module API REST et Serveur Web pour ORIENT'IA."""
try:
    from orient_ia.api.routes import app, create_app
except ImportError:
    app = None
    create_app = None

__all__ = ["app", "create_app"]
