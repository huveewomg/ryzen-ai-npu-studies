from __future__ import annotations

import unittest

from benchmarks.run_matrix import build_schedule, cell_id, timeout_text
from benchmarks.summarize_results import distribution


class MatrixScheduleTests(unittest.TestCase):
    def test_schedule_is_deterministic_and_complete(self):
        first = build_schedule(processes_per_cell=5, seed=7)
        second = build_schedule(processes_per_cell=5, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 90)
        self.assertEqual(len(set(first)), 90)
        self.assertNotEqual(first, build_schedule(processes_per_cell=5, seed=8))

    def test_schedule_can_select_one_extension_cell(self):
        schedule = build_schedule(
            processes_per_cell=5,
            seed=7,
            providers=("cpu",),
            batch_sizes=(32,),
            sequence_lengths=(512,),
        )
        self.assertEqual(len(schedule), 5)
        self.assertEqual(
            {
                (precision, provider, batch_size, seq_len)
                for precision, provider, batch_size, seq_len, _ in schedule
            },
            {("fp32", "cpu", 32, 512)},
        )
        self.assertEqual({repeat for *_, repeat in schedule}, {1, 2, 3, 4, 5})

    def test_cell_id_excludes_repeat(self):
        self.assertEqual(cell_id("fp32", "npu", 8, 128), "npu-fp32-b8-s128")

    def test_timeout_text_handles_subprocess_bytes(self):
        self.assertEqual(timeout_text(b"partial output\xff"), "partial output\ufffd")
        self.assertEqual(timeout_text(None), "")

    def test_process_distribution_uses_student_t_interval(self):
        result = distribution([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(result["n"], 5)
        self.assertEqual(result["mean"], 3.0)
        self.assertAlmostEqual(result["ci95_low"], 1.0368, places=3)
        self.assertAlmostEqual(result["ci95_high"], 4.9632, places=3)


if __name__ == "__main__":
    unittest.main()
