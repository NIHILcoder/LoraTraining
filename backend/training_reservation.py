"""Process-wide reservation for the GPU training slot.

LoRATrainer.is_training is only set once the blocking GPU loop starts. The HTTP
/training/start handler copies and converts the dataset first, so two overlapping
starts (double-click, or Start becoming clickable again during loading_model)
would otherwise both pass the busy check and run two train() threads on one GPU.
"""

import threading


class TrainingReservation:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held = False

    def try_acquire(self) -> bool:
        with self._lock:
            if self._held:
                return False
            self._held = True
            return True

    def release(self) -> None:
        with self._lock:
            self._held = False

    @property
    def held(self) -> bool:
        return self._held
