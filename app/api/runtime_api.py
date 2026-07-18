# infrastructure/api/runtime/runtime_api.py

from __future__ import annotations

import requests
from typing import Any

from app.core.config import settings


class RuntimeApiClient:
    """
    Synchronization gateway between the AI Engine and Django.

    This client is ONLY responsible for downloading
    runtime state required by the AI Engine.
    """

    def __init__(self):

        self.base_url = settings.DJANGO_API_URL.rstrip("/")

        self.session = requests.Session()

        self.timeout = 10

    # -------------------------
    # Runtime Health
    # -------------------------

    def health_check(self) -> bool:

        response = self.session.get(
            f"{self.base_url}/health/",
            timeout=self.timeout,
        )

        return response.status_code == 200

    # -------------------------
    # Devices
    # -------------------------

    def get_runtime_devices(self) -> dict[str, Any]:

        response = self.session.get(
            f"{self.base_url}/runtime/devices/",
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    # -------------------------
    # Faces
    # -------------------------

    def get_face_profiles(self) -> dict[str, Any]:

        response = self.session.get(
            f"{self.base_url}/runtime/faces/",
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    # -------------------------
    # Runtime Version
    # -------------------------

    def get_runtime_version(self) -> dict[str, Any]:

        response = self.session.get(
            f"{self.base_url}/runtime/version/",
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()