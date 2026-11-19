"""Offline CLI tests. No model downloads or GPU are required."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mms_tts import cli


ROOT = Path(__file__).resolve().parents[1]


class InputTests(unittest.TestCase):
    def test_normalizes_whitespace_and_decomposed_vietnamese(self):
        self.assertEqual(
            cli.normalize_text(" \tXin\ncha\u0300o\u00a0bạn! "), "Xin chào bạn!"
        )

    def test_reads_utf8_bom_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.txt"
            path.write_text("\ufeff Xin\n chào! ", encoding="utf-8")
            args = cli.build_parser().parse_args(["--text-file", str(path)])
            self.assertEqual(cli.read_text(args), "Xin chào!")

    def test_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            cli.normalize_text(" \n\t ")

    def test_default_model_and_language_alias(self):
        parser = cli.build_parser()
        self.assertEqual(
            cli.resolve_model(parser.parse_args(["--text", "Hi"])),
            "facebook/mms-tts-vie",
        )
        self.assertEqual(
            cli.resolve_model(parser.parse_args(["--text", "Hi", "--lang", "ENG"])),
            "facebook/mms-tts-eng",
        )
        self.assertEqual(
            cli.resolve_model(
                parser.parse_args(["--text", "Hi", "--model", "local/model"])
            ),
            "local/model",
        )

    def test_rejects_malformed_language_and_empty_model(self):
        parser = cli.build_parser()
        for options in (["--lang", "../eng"], ["--model-id", "   "]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                cli.resolve_model(parser.parse_args(["--text", "Hi", *options]))


class DeviceTests(unittest.TestCase):
    def test_auto_falls_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(cli.select_device("auto").type, "cpu")
            self.assertEqual(cli.select_device("cpu").type, "cpu")
            with self.assertRaisesRegex(ValueError, "CUDA"):
                cli.select_device("cuda")

    def test_auto_selects_available_cuda_but_cpu_can_be_forced(self):
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(cli.select_device("auto").type, "cuda")
            self.assertEqual(cli.select_device("cpu").type, "cpu")


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "mms_tts", *args],
            cwd=ROOT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_help_works_without_importing_ml_libraries(self):
        result = subprocess.run(
            [sys.executable, "-S", "-m", "mms_tts", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--text-file", result.stdout)
        self.assertIn("--model-id", result.stdout)

    def test_input_and_model_options_are_mutually_exclusive(self):
        for args in (
            [],
            ["--text", "hi", "--text-file", "input.txt"],
            ["--text", "hi", "--model-id", "model", "--lang", "eng"],
        ):
            with self.subTest(args=args):
                self.assertEqual(self.run_cli(*args).returncode, 2)

    def test_invalid_inputs_fail_cleanly_without_downloads(self):
        for args in (
            ["--text", " \n "],
            ["--text-file", "/missing/input.txt"],
            ["--text", "Hi", "--output", "out.mp3"],
            ["--text", "Hi", "--cpu-threads", "0"],
        ):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Traceback", result.stderr)

    def test_input_file_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            path.write_text("Xin chào!", encoding="utf-8")
            result = self.run_cli("--text-file", str(path), "--output", str(path))
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(path.read_text(encoding="utf-8"), "Xin chào!")

    def test_hard_link_output_cannot_overwrite_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.txt"
            output = Path(directory) / "output.wav"
            source.write_text("Xin chào!", encoding="utf-8")
            os.link(source, output)
            # Replace only expensive model inference; exercise the real CLI validation.
            with patch.object(cli, "synthesize", return_value=None):
                with self.assertLogs("mms_tts", level="ERROR"):
                    self.assertEqual(
                        cli.main(
                            [
                                "--text-file",
                                str(source),
                                "--output",
                                str(output),
                                "--device",
                                "cpu",
                            ]
                        ),
                        1,
                    )
            self.assertEqual(source.read_text(encoding="utf-8"), "Xin chào!")


class WavTests(unittest.TestCase):
    def test_writes_mono_pcm16_with_configured_sample_rate(self):
        import numpy as np
        from scipy.io import wavfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "speech.wav"
            duration = cli.write_wav(path, np.array([0.0, 0.5, -0.5, 2.0]), 16000)
            rate, audio = wavfile.read(path)
            self.assertEqual(rate, 16000)
            self.assertEqual(audio.shape, (4,))
            self.assertEqual(audio.dtype, np.int16)
            self.assertEqual(audio.tolist(), [0, 16384, -16384, 32767])
            self.assertEqual(duration, 0.00025)

    def test_rejects_empty_nonfinite_or_wrong_shape_waveform(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.wav"
            for waveform in ([], [float("nan")], [float("inf")], [[0.0], [1.0]]):
                with self.subTest(waveform=waveform), self.assertRaises(ValueError):
                    cli.write_wav(path, np.array(waveform), 16000)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
