"""Guard tests run on CPU and never launch the model or initialize CUDA."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from scripts import run_inference
from scripts.prepare_raft import patch_normalization


class GuardTests(unittest.TestCase):
    def test_failed_preflight_never_queries_or_launches_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("sys.argv", ["infer", "--output-dir", directory]),
                patch(
                    "scripts.run_inference.subprocess.run",
                    return_value=CompletedProcess([], 2, "", ""),
                ) as call,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(run_inference.main(), 2)
                self.assertEqual(call.call_count, 1)

    def test_busy_gpu_never_creates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "output"
            results = [CompletedProcess([], 0, "", ""), CompletedProcess([], 0, "100\n", "")]
            with (
                patch("sys.argv", ["infer", "--output-dir", str(out)]),
                patch("scripts.run_inference.subprocess.run", side_effect=results) as call,
                patch.object(run_inference, "RUNTIME", Path(directory) / "runtime"),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(run_inference.main(), 3)
                self.assertEqual(call.call_count, 2)
                self.assertFalse(out.exists())

    def test_existing_results_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.txt"
            result.write_text("keep")
            with (
                patch("sys.argv", ["infer", "--output-dir", directory]),
                patch("scripts.run_inference.subprocess.run") as call,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                run_inference.main()
            call.assert_not_called()
            self.assertEqual(result.read_text(), "keep")

    def test_raft_adjustment_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raft.py"
            path.write_text(
                "        image1 = 2 * (image1 / 255.0) - 1.0\n"
                "        image2 = 2 * (image2 / 255.0) - 1.0\n"
            )
            patch_normalization(path)
            once = path.read_text()
            patch_normalization(path)
            self.assertEqual(path.read_text(), once)
            self.assertNotIn("/ 255.0", once)

    def test_unexpected_raft_source_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raft.py"
            path.write_text("unexpected implementation")
            with self.assertRaisesRegex(RuntimeError, "Unexpected"):
                patch_normalization(path)
            self.assertEqual(path.read_text(), "unexpected implementation")


if __name__ == "__main__":
    unittest.main()
