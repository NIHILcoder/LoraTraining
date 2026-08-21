import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training_reservation import TrainingReservation


class TrainingReservationTests(unittest.TestCase):
    def test_second_acquire_fails_until_release(self):
        slot = TrainingReservation()
        self.assertTrue(slot.try_acquire())
        self.assertTrue(slot.held)
        self.assertFalse(slot.try_acquire())
        slot.release()
        self.assertFalse(slot.held)
        self.assertTrue(slot.try_acquire())
        slot.release()

    def test_release_is_idempotent(self):
        slot = TrainingReservation()
        slot.release()
        slot.release()
        self.assertFalse(slot.held)
        self.assertTrue(slot.try_acquire())
        slot.release()

    def test_only_one_concurrent_acquirer_wins(self):
        slot = TrainingReservation()
        winners = []
        barrier = threading.Barrier(8)

        def attempt():
            barrier.wait()
            if slot.try_acquire():
                winners.append(threading.get_ident())

        threads = [threading.Thread(target=attempt) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(winners), 1)
        self.assertTrue(slot.held)
        slot.release()


if __name__ == "__main__":
    unittest.main()
