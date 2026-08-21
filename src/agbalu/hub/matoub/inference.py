"""Matoub-82M — standalone inference.

No agbalu package required. Requires only:
    pip install torch torchaudio librosa soundfile huggingface_hub

Usage (command line):
    python inference.py --text "Azul fell-awen, amek i telliḍ taṣebḥit-a?" --out out.wav

Usage (Python):
    from inference import MatoubTTS
    tts = MatoubTTS.load()
    tts.synthesise("Azul fell-awen.", "out.wav")
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

try:
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from huggingface_hub import hf_hub_download
except ImportError as e:
    sys.exit(
        f"Missing dependency: {e}\n"
        "Install with: pip install torch torchaudio librosa soundfile huggingface_hub"
    )

REPO_ID = "agbalu/Matoub-82M"
CHECKPOINT_FILE = "epoch_2nd_00003.pth"
SAMPLE_RATE = 24_000

# A copy of `agbalu.tts.g2p`'s table, not an import: nothing under `hub/` may import
# `agbalu`, because the published repository ships without it. The two must agree.
_KAB_G2P: dict[str, str] = {
    "b": "b",
    "d": "d",
    "g": "ɡ",
    "k": "k",
    "p": "p",
    "t": "t",
    "q": "q",
    "f": "f",
    "v": "v",
    "s": "s",
    "z": "z",
    "x": "x",
    "ɣ": "ɣ",
    "ğ": "ɣ",
    "Ɣ": "ɣ",
    "ɛ": "ɛ",
    "h": "h",
    "ḥ": "ħ",
    "ṣ": "sˤ",
    "ẓ": "zˤ",
    "ḍ": "dˤ",
    "ṭ": "tˤ",
    "ṛ": "rˤ",
    "č": "tʃ",
    "ǧ": "dʒ",
    "m": "m",
    "n": "n",
    "l": "l",
    "r": "r",
    "w": "w",
    "y": "j",
    "a": "a",
    "e": "ə",
    "i": "i",
    "u": "u",
    "A": "a",
    "E": "ə",
    "I": "i",
    "U": "u",
}

_AFFRICATE_FOLD = {"tʃ": "ʧ", "dʒ": "ʤ"}


def _phonemize(text: str) -> str:
    text = text.strip()
    ipa_chars: list[str] = []
    for c in text:
        if c in " .,!?:;-'\"()[]/_":
            ipa_chars.append(c)
            continue
        ipa_chars.append(_KAB_G2P.get(c, c))
    ipa = "".join(ipa_chars)
    for src, tgt in _AFFRICATE_FOLD.items():
        ipa = ipa.replace(src, tgt)
    return ipa


def _add_styletts2_to_path(styletts2_dir: str | Path) -> None:
    root = str(Path(styletts2_dir).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


_VOCAB = (
    "$",
    "ɑ",
    "ɐ",
    "ɒ",
    "æ",
    "ə",
    "ɚ",
    "ʌ",
    "ɔ",
    "ɛ",
    "ɜ",
    "ɝ",
    "ɞ",
    "ɟ",
    "ɡ",
    "ɣ",
    "ʜ",
    "ɦ",
    "ħ",
    "ɨ",
    "ɪ",
    "ɫ",
    "ɬ",
    "ɭ",
    "ɮ",
    "ʎ",
    "ɱ",
    "ɯ",
    "ɰ",
    "ŋ",
    "ɳ",
    "ɲ",
    "ɴ",
    "ø",
    "ɵ",
    "ɸ",
    "θ",
    "œ",
    "ɶ",
    "ʘ",
    "ɹ",
    "ɺ",
    "ɾ",
    "ɻ",
    "ʀ",
    "ʁ",
    "ɽ",
    "ʂ",
    "ʃ",
    "ʈ",
    "ʧ",
    "ʉ",
    "ʊ",
    "ʋ",
    "ⱱ",
    "ʌ",
    "ɣ",
    "ʍ",
    "χ",
    "ʎ",
    "ʏ",
    "ʑ",
    "ʐ",
    "ʒ",
    "ʔ",
    "ʡ",
    "ʕ",
    "ʢ",
    "ǀ",
    "ǁ",
    "ǂ",
    "ǃ",
    "ˈ",
    "ˌ",
    "ː",
    "ˑ",
    "ʼ",
    "ʴ",
    "ʰ",
    "ʱ",
    "ʲ",
    "ʷ",
    "ʸ",
    "˞",
    "↓",
    "↑",
    "→",
    "↗",
    "↘",
    "'",
    '"',
    "ˆ",
    "ˋ",
    " ",
    "q",
    "ʤ",
    "ħ",
    "a",
    "b",
    "d",
    "e",
    "f",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "ɛ",
    "ɡ",
    "dˤ",
    "tˤ",
    "sˤ",
    "zˤ",
    "rˤ",
)

_SYM_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(_VOCAB)}


def _ipa_to_tokens(ipa: str) -> list[int]:
    ids = [0]
    for ch in ipa:
        idx = _SYM_TO_IDX.get(ch)
        if idx is not None:
            ids.append(idx)
    return ids


class MatoubTTS:
    def __init__(
        self,
        model: dict[str, Any],
        model_params: Any,
        ref_style: torch.Tensor,
        sampler: Any,
        device: torch.device,
        alpha: float = 0.0,
        beta: float = 0.0,
        diffusion_steps: int = 10,
        embedding_scale: float = 1.0,
    ) -> None:
        self.model = model
        self.model_params = model_params
        self.ref_style = ref_style
        self.sampler = sampler
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.diffusion_steps = diffusion_steps
        self.embedding_scale = embedding_scale

    @classmethod
    def load(
        cls,
        checkpoint: str | Path | None = None,
        reference_wav: str | Path | None = None,
        styletts2_dir: str | Path = ".",
        alpha: float = 0.0,
        beta: float = 0.0,
        diffusion_steps: int = 10,
        embedding_scale: float = 1.0,
        device: str | None = None,
    ) -> MatoubTTS:
        _add_styletts2_to_path(styletts2_dir)

        torch_device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if checkpoint is None:
            checkpoint = hf_hub_download(repo_id=REPO_ID, filename=CHECKPOINT_FILE)
        ckpt_path = Path(checkpoint)

        import yaml

        config_candidates = [
            ckpt_path.parent / "config_stage2_kab_male.yml",
            ckpt_path.parent / "config_stage2.yml",
            ckpt_path.parent / "config.yml",
            Path(styletts2_dir) / "Configs" / "config_ft.yml",
            Path(styletts2_dir) / "Configs" / "config.yml",
        ]
        config_file = next((c for c in config_candidates if c.is_file()), None)
        if config_file is None:
            message = (
                "No config YAML found. Pass the StyleTTS2 repository path as "
                "`styletts2_dir`, or place a config.yml next to the checkpoint."
            )
            raise RuntimeError(message)
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))

        from models import build_model, load_ASR_models, load_F0_models
        from utils import recursive_munch
        from Utils.PLBERT.util import load_plbert

        styletts2 = Path(styletts2_dir)
        text_aligner = load_ASR_models(
            str(styletts2 / "Utils" / "ASR" / "epoch_00080.pth"),
            str(styletts2 / "Utils" / "ASR" / "config.yml"),
        )
        pitch_extractor = load_F0_models(str(styletts2 / "Utils" / "JDC" / "bst.t7"))
        plbert = load_plbert(str(styletts2 / "Utils" / "PLBERT"))

        model_params = recursive_munch(config.get("model_params", {}))
        model = build_model(model_params, text_aligner, pitch_extractor, plbert)
        for module in model.values():
            if isinstance(module, torch.nn.Module):
                module.to(torch_device).eval()

        state = torch.load(ckpt_path, map_location=torch_device, weights_only=False)
        net = state.get("net", state)
        for key in model:
            if key in net and hasattr(model[key], "load_state_dict"):
                try:
                    model[key].load_state_dict(net[key])
                except Exception:
                    sd = OrderedDict(
                        (k[7:] if k.startswith("module.") else k, v) for k, v in net[key].items()
                    )
                    model[key].load_state_dict(sd, strict=False)

        if reference_wav is None:
            reference_wav = hf_hub_download(repo_id=REPO_ID, filename="reference_kab_male.wav")
        ref_path = Path(reference_wav)

        wave, sr = librosa.load(str(ref_path), sr=SAMPLE_RATE)
        audio_trimmed, _ = librosa.effects.trim(wave, top_db=30)
        if sr != SAMPLE_RATE:
            audio_trimmed = librosa.resample(audio_trimmed, orig_sr=sr, target_sr=SAMPLE_RATE)

        to_mel = torchaudio.transforms.MelSpectrogram(
            n_fft=2048,
            win_length=1200,
            hop_length=300,
            n_mels=80,
            f_min=0,
            f_max=8000,
        )
        mel = to_mel(torch.from_numpy(audio_trimmed).float().unsqueeze(0))
        mel = (torch.log(1e-5 + mel) - (-4)) / 4
        mel = mel.to(torch_device)

        with torch.no_grad():
            ref_s = model["style_encoder"](mel.unsqueeze(1))
            ref_p = model["predictor_encoder"](mel.unsqueeze(1))
            ref_style = torch.cat([ref_s, ref_p], dim=1)

        from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule

        sampler = DiffusionSampler(
            model["diffusion"].diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )

        return cls(
            model=model,
            model_params=model_params,
            ref_style=ref_style,
            sampler=sampler,
            device=torch_device,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
        )

    def synthesise(self, text: str, output_path: str | Path = "output.wav") -> Path:
        from utils import length_to_mask

        ipa = _phonemize(text)
        token_ids = _ipa_to_tokens(ipa)
        tokens = torch.LongTensor([token_ids]).to(self.device)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model["text_encoder"](tokens, input_lengths, text_mask)
            bert_dur = self.model["bert"](tokens, attention_mask=(~text_mask).int())
            d_en = self.model["bert_encoder"](bert_dur).transpose(-1, -2)

            s_pred = self.sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                embedding=bert_dur,
                embedding_scale=self.embedding_scale,
                features=self.ref_style,
                num_steps=self.diffusion_steps,
            ).squeeze(1)

            ref = self.alpha * s_pred[:, :128] + (1 - self.alpha) * self.ref_style[:, :128]
            s = self.beta * s_pred[:, 128:] + (1 - self.beta) * self.ref_style[:, 128:]

            d = self.model["predictor"].text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = self.model["predictor"].lstm(d)
            duration = torch.sigmoid(self.model["predictor"].duration_proj(x)).sum(dim=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            if pred_dur.dim() == 0:
                pred_dur = pred_dur.unsqueeze(0)
            pred_dur[-1] += 5

            n_tokens = int(input_lengths.item())
            n_frames = int(pred_dur.sum().item())
            pred_aln_trg = torch.zeros(n_tokens, n_frames)
            c = 0
            for i in range(n_tokens):
                di = int(pred_dur[i].item())
                pred_aln_trg[i, c : c + di] = 1
                c += di
            pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)

            en = d.transpose(-1, -2) @ pred_aln_trg
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model["predictor"].F0Ntrain(en, s)

            asr = t_en @ pred_aln_trg
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model["decoder"](asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
            audio = out.squeeze().cpu().numpy()[..., :-50]

        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = (audio / max_val) * 0.95

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, SAMPLE_RATE)
        return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Matoub-82M Kabyle TTS")
    parser.add_argument("--text", required=True, help="Kabyle text to synthesise")
    parser.add_argument("--out", default="output.wav", help="Output WAV path")
    parser.add_argument("--checkpoint", default=None, help="Path to epoch_2nd_00003.pth")
    parser.add_argument("--reference", default=None, help="Reference speaker WAV (24 kHz)")
    parser.add_argument("--styletts2", default=".", help="StyleTTS2 repo root directory")
    parser.add_argument("--alpha", type=float, default=0.0, help="Acoustic style blend")
    parser.add_argument("--beta", type=float, default=0.0, help="Prosodic style blend")
    parser.add_argument("--steps", type=int, default=10, help="Diffusion steps")
    parser.add_argument("--device", default=None, help="cuda / cpu")
    args = parser.parse_args()

    tts = MatoubTTS.load(
        checkpoint=args.checkpoint,
        reference_wav=args.reference,
        styletts2_dir=args.styletts2,
        alpha=args.alpha,
        beta=args.beta,
        diffusion_steps=args.steps,
        device=args.device,
    )
    out = tts.synthesise(args.text, args.out)
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
