"""Modal entrypoints. Run as a module path so sibling imports resolve in the container:

modal run -m modal_app.train::upload_corpus

Entrypoint names are unique across the package, not just within a module: they all register
on the one `modal.App`, and `modal_app.deploy` imports every module at once.
"""
