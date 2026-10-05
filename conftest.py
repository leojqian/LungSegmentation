# Ensures the project root is on sys.path regardless of how pytest is invoked,
# so tests/ can import the lungmap package (and the root-level Step*/submission
# scripts) from a plain checkout, without `pip install -e .` first.
