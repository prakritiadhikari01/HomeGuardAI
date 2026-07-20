from __future__ import annotations

from app.application.runtime_manager import RuntimeManager
from app.infrastructure.ai.known_faces_store import KnownFacesStore


def runtime_tests(runner):

    runner.run(
        "KnownFacesStore Singleton",
        test_known_faces_store,
    )

    runner.run(
        "RuntimeManager Initialization",
        test_runtime_manager,
    )


def test_known_faces_store():

    store1 = KnownFacesStore()
    store2 = KnownFacesStore()

    assert store1 is store2

    print(f"Cached Faces : {len(store1.get_all_faces())}")


def test_runtime_manager():

    manager = RuntimeManager()

    assert manager is not None

    print("RuntimeManager initialized successfully.")