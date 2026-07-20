"""
HomeGuard AI Engine Diagnostics

Run:

python -m app.tests.run_diagnostics

This package contains integration diagnostics for the AI Engine.
These are NOT unit tests.

They verify that:

- Runtime loads
- Models load
- Cameras connect
- Django API is reachable
- Qwen is reachable
- Pipeline components initialize

without requiring the frontend.
"""