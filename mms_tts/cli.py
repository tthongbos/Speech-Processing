"""MMS/VITS inference CLI; ML imports are deferred so --help stays lightweight."""

import argparse
import logging
from pathlib import Path
import re
import time
import unicodedata


LOGGER = logging.getLogger("mms_tts")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m mms_tts",
        description="Generate a mono WAV with MMS TTS (Vietnamese by default).",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text to synthesize; quote it in your shell.")
    source.add_argument(
        "--text-file", type=Path, help="Read text from a UTF-8 file (BOM accepted)."
    )
    model = parser.add_mutually_exclusive_group()
    model.add_argument(
        "--model-id",
        "--model",
        help="Hugging Face VITS model ID or local model directory (default: facebook/mms-tts-vie).",
    )
    model.add_argument(
        "--lang",
        help="MMS language code, e.g. vie or eng; selects facebook/mms-tts-<lang>.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/mms_tts.wav"),
        help="Output .wav path (default: outputs/mms_tts.wav). Existing WAV files are replaced.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto selects CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for VITS duration/noise sampling (default: 42).",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="Torch CPU threads; avoids oversubscription on large hosts (default: 4).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load only cached/local model files, without downloads.",
    )
    return parser


def normalize_text(text):
    text = " ".join(unicodedata.normalize("NFC", text).split())
    if not text:
        raise ValueError("Input text is empty after whitespace normalization.")
    return text


def read_text(args):
    text = (
        args.text_file.read_text(encoding="utf-8-sig") if args.text_file else args.text
    )
    return normalize_text(text)


def resolve_model(args):
    if args.lang is not None:
        language = args.lang.strip().lower()
        if not re.fullmatch(r"[a-z]{3}(?:-[a-z0-9_]+)*", language):
            raise ValueError("Use an MMS language code such as vie or eng for --lang.")
        return f"facebook/mms-tts-{language}"
    model_id = (
        args.model_id.strip() if args.model_id is not None else "facebook/mms-tts-vie"
    )
    if not model_id:
        raise ValueError("Model ID must not be empty.")
    return model_id


def select_device(requested):
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested but is unavailable. Install a CUDA-enabled PyTorch build and check the GPU, or use --device cpu."
        )
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def write_wav(output, waveform, sampling_rate):
    import numpy as np
    from scipy.io import wavfile

    waveform = np.asarray(waveform)
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Model returned an empty, non-finite or non-mono waveform.")
    if sampling_rate <= 0:
        raise ValueError("Model sampling rate must be positive.")
    pcm = np.rint(np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
    output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(output), sampling_rate, pcm)
    return waveform.size / sampling_rate


def synthesize(text, model_id, output, device, seed, cpu_threads, local_files_only):
    import torch
    from transformers import AutoTokenizer, VitsModel, set_seed

    torch.set_num_threads(cpu_threads)
    set_seed(seed)
    LOGGER.info(
        "Model: %s | device: %s | seed: %d | characters: %d",
        model_id,
        device,
        seed,
        len(text),
    )
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, local_files_only=local_files_only
    )
    # Validate tokens before loading the larger model; some tokenizers strip punctuation entirely.
    inputs = tokenizer(text, return_tensors="pt")
    speech_tokens = inputs["input_ids"]
    meaningful = torch.ones_like(speech_tokens, dtype=torch.bool)
    for token_id in (tokenizer.pad_token_id, tokenizer.unk_token_id):
        if token_id is not None:
            meaningful &= speech_tokens != token_id
    if not meaningful.any().item():
        raise ValueError(
            "Text contains no speakable tokens for this model; use text in the selected language."
        )
    model = (
        VitsModel.from_pretrained(model_id, local_files_only=local_files_only)
        .to(device)
        .eval()
    )
    inputs = inputs.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started
    LOGGER.info("Model/tokenizer ready in %.3f s", load_seconds)

    started = time.perf_counter()
    with torch.inference_mode():
        waveform = model(**inputs).waveform[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    duration = write_wav(
        output, waveform.detach().cpu().float().numpy(), model.config.sampling_rate
    )
    LOGGER.info(
        "Saved: %s | mono PCM16 | %d Hz | audio: %.3f s",
        output.resolve(),
        model.config.sampling_rate,
        duration,
    )
    LOGGER.info(
        "Inference: %.3f s | RTF: %.3f (inference/audio duration; lower is faster)",
        inference_seconds,
        inference_seconds / duration,
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        text = read_text(args)
        model_id = resolve_model(args)
        if args.output.suffix.lower() != ".wav":
            raise ValueError("Output must have a .wav extension.")
        if args.text_file and (
            args.output.resolve() == args.text_file.resolve()
            or (args.output.exists() and args.output.samefile(args.text_file))
        ):
            raise ValueError("Output must not overwrite the input text file.")
        if args.cpu_threads < 1:
            raise ValueError("--cpu-threads must be positive.")
        device = select_device(args.device)
        synthesize(
            text,
            model_id,
            args.output,
            device,
            args.seed,
            args.cpu_threads,
            args.local_files_only,
        )
    except ImportError as error:
        LOGGER.error(
            "Missing/incompatible dependency: %s. Install: python -m pip install -r mms_tts/requirements.txt (some languages also need: python -m pip install uroman).",
            error,
        )
        return 1
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        LOGGER.error("%s", error)
        LOGGER.error(
            "Check the model ID, installed dependencies, network/cache and output permissions. Offline: --local-files-only with a previously downloaded model."
        )
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted.")
        return 130
    return 0
