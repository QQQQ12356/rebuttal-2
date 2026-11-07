"""Live benchmark progress for terminals and line-oriented logs."""
import time


def log_progress(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + message, flush=True)


class BatchProgress:
    """Report the first and last completed batches, with updates in between."""

    def __init__(self, label: str, total: int):
        self.label = label
        self.total = total
        self.completed = 0
        self.started = self.last_report = time.monotonic()
        log_progress(f"{self.label}: starting samples=0/{self.total}")

    def advance(self, count: int) -> None:
        first_batch = self.completed == 0
        self.completed += count
        now = time.monotonic()
        if first_batch or self.completed == self.total or now - self.last_report >= 10:
            percent = 100 * self.completed / self.total
            log_progress(
                f"{self.label}: samples={self.completed}/{self.total} ({percent:.1f}%), "
                f"elapsed={now - self.started:.1f}s"
            )
            self.last_report = now
