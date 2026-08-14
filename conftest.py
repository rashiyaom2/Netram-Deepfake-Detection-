"""Root-level pytest configuration."""
import pytest


# pytest-asyncio: auto-mode so @pytest.mark.asyncio is applied automatically
# to all async test functions (avoids needing the decorator on every test).
def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker("asyncio") is None and hasattr(item, "function"):
            import asyncio
            if asyncio.iscoroutinefunction(item.function):
                item.add_marker(pytest.mark.asyncio)
