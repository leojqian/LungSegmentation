# Ensures the project root is on sys.path so tests/ can import geometry/,
# segmentation/, formats/, and rendering/ as packages (no __init__.py — these
# are implicit namespace packages) regardless of how pytest is invoked.
