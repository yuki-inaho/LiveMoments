"""CPU tests for portable path resolution and inference admission."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from live_utils.config import load_config, runtime_dir


class RuntimeConfigTests(unittest.TestCase):
    def test_runtime_override(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"LIVEMOMENTS_RUNTIME_DIR": directory}):
                self.assertEqual(runtime_dir(), Path(directory).resolve())

    def test_explicit_relative_and_runtime_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "settings.yml"
            config.write_text(
                'lr_path: "images/lr"\nref_path: "${LIVEMOMENTS_RUNTIME_DIR}/images/ref"\nlr_res: 768.0\n'
            )
            with patch.dict(os.environ, {"LIVEMOMENTS_RUNTIME_DIR": directory}):
                data = load_config(config)
            self.assertEqual(data["lr_path"], str(Path(directory) / "images/lr"))
            self.assertEqual(data["ref_path"], str(Path(directory) / "images/ref"))
            self.assertEqual(data["lr_res"], 768.0)

    def test_unresolved_variable_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "settings.yml"
            config.write_text('lr_path: "${LIVEMOMENTS_UNSET_TEST_VARIABLE}/lr"\n')
            with self.assertRaisesRegex(ValueError, "Unresolved"):
                load_config(config)


if __name__ == "__main__":
    unittest.main()
