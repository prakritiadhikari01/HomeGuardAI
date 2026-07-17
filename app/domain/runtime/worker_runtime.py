from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID


@dataclass(slots=True)
class WorkerRuntime:
    """
    Runtime state of one processing worker.

    A worker owns exactly one camera.
    """

    worker_id: UUID

    camera_id: UUID

    running: bool = False

    crashed: bool = False

    restart_count: int = 0

    last_error: str | None = None

    started_at: datetime = field(default_factory=datetime.utcnow)

    last_heartbeat: datetime | None = None

    def heartbeat(self) -> None:
        self.last_heartbeat = datetime.utcnow()

    def mark_running(self) -> None:
        self.running = True
        self.crashed = False

    def mark_stopped(self) -> None:
        self.running = False

    def mark_crashed(self, error: str) -> None:
        self.running = False
        self.crashed = True
        self.last_error = error
        self.restart_count += 1