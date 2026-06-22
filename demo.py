"""
demo.py — Sketch -> Qwen prompt -> ControlNet -> rembg -> TripoSR

Pipeline:
  1. Hand sketch -> Qwen -> ControlNet prompt
  2. ControlNet Scribble -> generated image
  3. rembg -> foreground RGBA kept in memory
  4. Qwen image validation -> retry up to max attempts
  5. TripoSR -> 3D mesh -> interactive viewer
  6. Natural-language edit -> Qwen rewrites prompt -> repeat from step 2

The script writes only inside one output directory. Each run gets a numbered
subdirectory, and edit iterations overwrite stage files inside that run.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import warnings
warnings.filterwarnings("ignore")

import logging
logging.disable(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent / "TripoSR"))

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh as tm_lib
from PIL import Image


QWEN_MODEL_ID = "Qwen/Qwen3.5-0.8B"
CONTROLNET_ID = "lllyasviel/sd-controlnet-scribble"
CONTROLNET_BASE_ID = "runwayml/stable-diffusion-v1-5"
TRIPOSR_MODEL_ID = "stabilityai/TripoSR"

# Qwen fills OBJECT and DESCRIPTORS only; Python assembles the rest.
PROMPT_TEMPLATE = "{object}, {descriptors}, isometric view, no background"
PROMPT_SUFFIX = "isometric view, no background"
QWEN_SKETCH_INSTRUCTION = """\
Look at the sketch and infer the real-world object it represents.
Describe the object as if it were a real physical object, not as a drawing.

Do not describe the sketch medium or drawing style.

Example for an apple: green apple, natural texture
Example for a warrior: Greek warrior, bronze armor, red plume, leather straps
Example for a dog: golden retriever, fluffy brown fur, playful expression"""

_SUFFIX_KEYWORDS = [
    "isometric view", "isometric", "no background",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


_B  = "\033[1m"       # bold
_C  = "\033[36m"      # cyan
_G  = "\033[32m"      # green
_RD = "\033[31m"      # red
_DM = "\033[2m"       # dim
_R  = "\033[0m"       # reset


def print_prompt(prompt: str, previous: str = "") -> None:
    prev_words = {w.strip().lower() for w in previous.replace(",", " ").split() if w.strip()}
    parts = [p.strip() for p in prompt.split(",")]
    colored = []
    for part in parts:
        words = {w.strip().lower() for w in part.split() if w.strip()}
        if not prev_words or (words - prev_words):
            colored.append(f"{_RD}{part}{_R}")
        else:
            colored.append(part)
    print(f"  {', '.join(colored)}", flush=True)


def log(message: str) -> None:
    print(f"  {message}", flush=True)


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m {rest:.1f}s"


@contextmanager
def timed_stage(label: str):
    start = time.perf_counter()
    print(f"\n{_B}{_C}{label}{_R}", flush=True)
    try:
        yield
    finally:
        print(f"  {_DM}done in {format_seconds(time.perf_counter() - start)}{_R}", flush=True)


def parse_template_output(raw: str) -> str:
    obj, desc = "", ""
    for line in raw.splitlines():
        line = line.strip().replace("**", "").replace("*", "").replace("`", "").strip()
        upper = line.upper()
        if upper.startswith(("NEW OBJECT:", "OBJECT:")):
            obj = line.split(":", 1)[1].strip()
        elif (upper.startswith("NEW DESCRI") or upper.startswith("DESCRI")) and ":" in line:
            desc = line.split(":", 1)[1].strip()
    if not obj:
        obj = raw.strip().splitlines()[0].strip()

    obj_parts = [p.strip() for p in obj.split(",") if p.strip()]
    if len(obj_parts) > 1:
        obj = obj_parts[0]
        spill = obj_parts[1:]
        if desc:
            desc_parts = [p.strip() for p in desc.split(",") if p.strip()]
            desc_lower_set = {p.lower() for p in desc_parts}
            desc_parts = [p for p in spill if p.lower() not in desc_lower_set] + desc_parts
            desc = ", ".join(desc_parts)
        else:
            desc = ", ".join(spill)

    if desc:
        desc_parts = [p.strip() for p in desc.split(",") if p.strip()]
        desc_parts = [
            p for p in desc_parts
            if not any(kw in p.lower() for kw in _SUFFIX_KEYWORDS)
        ]
        obj_lower = obj.lower()
        seen: set[str] = set()
        unique = []
        for p in desc_parts:
            key = p.lower()
            if key and key not in seen and key != obj_lower:
                seen.add(key)
                unique.append(p)
        desc = ", ".join(unique) if unique else "natural texture"
    else:
        desc = "natural texture"
    return PROMPT_TEMPLATE.format(object=obj, descriptors=desc)


def strip_prompt_suffix(prompt: str) -> str:
    suffix = ", " + PROMPT_SUFFIX
    return prompt[:-len(suffix)] if prompt.endswith(suffix) else prompt


def with_prompt_suffix(prompt_core: str) -> str:
    prompt_core = strip_prompt_suffix(prompt_core).strip().rstrip(" ,")
    return f"{prompt_core}, {PROMPT_SUFFIX}" if prompt_core else PROMPT_SUFFIX


def clean_prompt_output(raw: str) -> str:
    lines = []
    for line in raw.replace("**", "").replace("*", "").replace("`", "").splitlines():
        line = line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith(("PROMPT:", "REVISED PROMPT:", "NEW PROMPT:", "FINAL:", "ANSWER:")):
            line = line.split(":", 1)[1].strip()
        if upper.startswith(("RETURN ", "DO NOT ", "CURRENT PROMPT:", "EDIT:")):
            continue
        lines.append(line)
    return sanitize_controlnet_prompt(" ".join(lines).strip())


def is_bad_rewrite(revised_core: str, current_core: str) -> bool:
    revised_norm = " ".join(revised_core.lower().split())
    if not revised_norm:
        return True
    bad_fragments = [
        "return only",
        "rewrite the current prompt",
        "current prompt:",
        "edit:",
        "prompt:",
    ]
    return any(fragment in revised_norm for fragment in bad_fragments)


def sanitize_controlnet_prompt(prompt: str) -> str:
    cleaned = " ".join(prompt.replace("\n", " ").split())
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:])\1+", r"\1", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    return cleaned.strip(" ,.;:")


def get_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_qwen(device: str, model_id: str = QWEN_MODEL_ID):
    import transformers
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    from transformers import AutoModelForImageTextToText, AutoProcessor

    log(f"Loading Qwen model: {model_id} on {device}")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def qwen_generate(model, processor, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = generated[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


def make_controlnet_prompt(
    sketch_path: Path,
    output_dir: Path,
    device: str,
    qwen_model_id: str,
    max_new_tokens: int,
) -> str:
    with timed_stage("[1] Qwen prompt generation"):
        model, processor = load_qwen(device, qwen_model_id)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(sketch_path)},
                    {
                        "type": "text",
                        "text": QWEN_SKETCH_INSTRUCTION,
                    },
                ],
            }
        ]
        raw = qwen_generate(model, processor, messages, max_new_tokens=max_new_tokens)
        prompt = parse_template_output(raw)
        save_text(output_dir / "01_qwen_controlnet_prompt.txt", prompt)
        del model, processor
    return prompt


_marian_model = None
_marian_tokenizer = None

def translate_to_english(text: str) -> str:
    global _marian_model, _marian_tokenizer
    if _marian_model is None:
        from transformers import MarianMTModel, MarianTokenizer
        log("Loading translator: Helsinki-NLP/opus-mt-ko-en")
        _marian_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
        _marian_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
        _marian_model.eval()
    inputs = _marian_tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = _marian_model.generate(**inputs)
    return _marian_tokenizer.decode(out[0], skip_special_tokens=True)


def needs_korean_translation(text: str) -> bool:
    import re
    return bool(re.search(r"[가-힣]", text))


def rewrite_prompt_with_edit(
    current_prompt: str,
    edit: str,
    output_dir: Path,
    device: str,
    qwen_model_id: str,
    max_new_tokens: int,
) -> tuple[str, str]:
    core = strip_prompt_suffix(current_prompt)

    # Translate only Korean input; English stays untouched.
    edit_en = edit
    if needs_korean_translation(edit):
        try:
            edit_en = translate_to_english(edit)
            log(f"  Translated: {edit_en}")
        except Exception as exc:
            log(f"  Translation failed; using original edit ({type(exc).__name__})")

    raw = ""
    revised_core = core
    try:
        with timed_stage("[1] Qwen prompt rewrite"):
            model, processor = load_qwen(device, qwen_model_id)
            try:
                instruction = (
                    "Rewrite the current prompt to satisfy the edit.\n"
                    "Return only one short comma-separated English prompt.\n\n"

                    f"CURRENT PROMPT: {core}\n"
                    f"EDIT: {edit_en}\n"
                    "PROMPT:"
                )
                messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
                raw = qwen_generate(model, processor, messages, max_new_tokens=80).strip()

                revised_core = clean_prompt_output(raw).strip().rstrip(" ,")
                if is_bad_rewrite(revised_core, core):
                    retry_instruction = (
                        "You are editing an image generation prompt.\n"
                        "Apply EDIT to CURRENT PROMPT.\n"
                        "Output the final prompt only. No explanations. No labels.\n"
                        "Do not copy these instructions.\n\n"
                        f"CURRENT PROMPT: {core}\n"
                        f"EDIT: {edit_en}\n"
                        "FINAL:"
                    )
                    messages = [{"role": "user", "content": [{"type": "text", "text": retry_instruction}]}]
                    raw_retry = qwen_generate(model, processor, messages, max_new_tokens=80).strip()
                    retry_core = clean_prompt_output(raw_retry).strip().rstrip(" ,")
                    if not is_bad_rewrite(retry_core, core):
                        raw = raw_retry
                        revised_core = retry_core
            finally:
                del model, processor
    except Exception as exc:
        log(f"  Qwen rewrite failed; keeping previous prompt ({type(exc).__name__})")

    if is_bad_rewrite(revised_core, core):
        log("  Qwen rewrite was not clean; keeping previous prompt")
        revised_core = core

    revised = with_prompt_suffix(revised_core)
    save_text(output_dir / "01_qwen_controlnet_prompt.txt", revised)
    save_json(output_dir / "edit_request.json", {"edit": edit, "edit_en": edit_en,
                                                  "previous": current_prompt, "raw": raw,
                                                  "revised": revised})
    return revised, edit_en


def qwen_evaluate_image(
    image: Image.Image,
    current_prompt: str,
    user_intent: str,
    device: str,
    qwen_model_id: str,
    max_new_tokens: int,
) -> str | None:
    tmp = "/tmp/qwen_eval_image.png"
    image.save(tmp)

    core = strip_prompt_suffix(current_prompt)
    core_parts = [p.strip() for p in core.split(",") if p.strip()]
    current_object = core_parts[0] if core_parts else core
    current_desc = ", ".join(core_parts[1:]) if len(core_parts) > 1 else ""

    model, processor = load_qwen(device, qwen_model_id)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": tmp},
            {"type": "text", "text": (
                f"The image should show: {current_object}\n"
                f"With these properties: {current_desc}\n\n"
                f"Does the image match? Reply with exactly one of these formats:\n"
                f"OK\n"
                f"DESCRIPTORS: concise comma-separated fixes\n\n"
                f"Rules:\n"
                f"- Do NOT output OBJECT line\n"
                f"- Do NOT describe what you see — only fix what is wrong vs the target\n"
                f"- Use positive visual phrases only; never write phrases starting with 'no '\n"
            )},
        ],
    }]

    raw = qwen_generate(model, processor, messages, max_new_tokens=max_new_tokens)
    del model, processor
    if device == "mps":
        torch.mps.empty_cache()

    raw_upper = raw.strip().upper()
    if raw_upper.startswith("OK") or "DESCRIPTORS:" not in raw_upper:
        return None

    new_desc = ""
    for line in raw.splitlines():
        line = line.strip().replace("**", "").replace("*", "").strip()
        if line.upper().startswith("DESCRIPTORS:"):
            new_desc = line.split(":", 1)[1].strip()
            break
    new_desc = clean_prompt_output(new_desc).strip().rstrip(" ,")
    if not new_desc:
        return None
    desc_norm = new_desc.lower()
    if "<" in new_desc or ">" in new_desc or "if color" in desc_norm:
        return None

    parts = [p.strip() for p in new_desc.split(",") if p.strip()]
    parts = [p for p in parts if not any(kw in p.lower() for kw in _SUFFIX_KEYWORDS)]
    parts = [p for p in parts if not p.lower().startswith("no ")]
    if not parts:
        return None
    new_desc = ", ".join(parts)

    refined = PROMPT_TEMPLATE.format(object=current_object, descriptors=new_desc)
    return refined if refined != current_prompt else None


def sketch_to_scribble_control(sketch: Image.Image, size: int = 512, pad_ratio: float = 0.15) -> Image.Image:
    log("Preparing ControlNet scribble image")
    arr = np.array(sketch.convert("L"))
    if arr.mean() > 128:
        arr = 255 - arr
    _, arr = cv2.threshold(arr, 45, 255, cv2.THRESH_BINARY)
    # Add padding so objects near edges aren't cropped by ControlNet
    h, w = arr.shape
    pad = int(max(h, w) * pad_ratio)
    arr = np.pad(arr, pad, mode="constant", constant_values=0)
    return Image.fromarray(arr).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def load_controlnet_pipe(device: str):
    import diffusers
    diffusers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.disable_progress_bar()
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    dtype = torch.float16 if device == "cuda" else torch.float32
    log(f"Loading ControlNet: {CONTROLNET_ID}")
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        CONTROLNET_BASE_ID, controlnet=controlnet,
        torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def run_controlnet_inference(pipe, sketch: Image.Image, prompt: str, output_dir: Path,
                             steps: int, seed: int, stage_label: str = "[2] ControlNet",
                             guidance_scale: float = 9.0,
                             conditioning_scale: float = 0.75) -> Image.Image:
    negative = "drawing, sketch, line art, illustration, cartoon, anime, 2d, flat, paper cutout"
    control = sketch_to_scribble_control(sketch)
    control.save(output_dir / "02_control_scribble.png")
    with timed_stage(f"{stage_label} ({steps} steps)"):
        image = pipe(
            prompt=prompt, negative_prompt=negative, image=control,
            num_inference_steps=steps, guidance_scale=guidance_scale,
            controlnet_conditioning_scale=conditioning_scale,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]
    image.save(output_dir / "02_controlnet_image_raw.png")
    return image


def run_controlnet(
    sketch: Image.Image,
    prompt: str,
    output_dir: Path,
    device: str,
    steps: int,
    seed: int,
    stage_label: str = "[2] ControlNet",
    pipe=None,
    guidance_scale: float = 9.0,
    conditioning_scale: float = 0.75,
) -> Image.Image:
    own_pipe = pipe is None
    if own_pipe:
        pipe = load_controlnet_pipe(device)

    image = run_controlnet_inference(pipe, sketch, prompt, output_dir, steps, seed, stage_label,
                                     guidance_scale=guidance_scale,
                                     conditioning_scale=conditioning_scale)

    if own_pipe:
        del pipe

    with timed_stage("[3] Background removal"):
        import rembg
        # onnxruntime CoreML warning goes to C-level stderr; redirect fd 2 to suppress it
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        _saved_fd = os.dup(2)
        os.dup2(_devnull_fd, 2)
        try:
            session = rembg.new_session("birefnet-general")
        finally:
            os.dup2(_saved_fd, 2)
            os.close(_saved_fd)
            os.close(_devnull_fd)
        rgba = rembg.remove(image, session=session)
    return rgba


def generate_controlnet_image(
    sketch: Image.Image,
    prompt: str,
    output_dir: Path,
    device: str,
    args: argparse.Namespace,
    seed: int,
    user_intent: str,
    stage_prefix: str = "[2] ControlNet",
) -> tuple[Image.Image, str, int]:
    pipe = load_controlnet_pipe(device)
    last_image = None
    last_prompt = prompt
    last_seed = seed
    try:
        if args.no_refine:
            hq_image = run_controlnet(
                sketch, prompt, output_dir, device, args.controlnet_steps, seed,
                stage_label=stage_prefix, pipe=pipe,
                guidance_scale=args.controlnet_guidance_scale,
                conditioning_scale=args.controlnet_conditioning_scale,
            )
            return hq_image, prompt, seed

        for attempt in range(1, args.max_refine_attempts + 1):
            prompt_for_image = prompt
            seed_for_image = seed
            hq_image = run_controlnet(
                sketch, prompt, output_dir, device, args.controlnet_steps, seed,
                stage_label=f"{stage_prefix} (attempt {attempt})", pipe=pipe,
                guidance_scale=args.controlnet_guidance_scale,
                conditioning_scale=args.controlnet_conditioning_scale,
            )
            last_image = hq_image
            last_prompt = prompt_for_image
            last_seed = seed_for_image

            with timed_stage(f"[4] Qwen image evaluation (attempt {attempt})"):
                white = Image.new("RGBA", hq_image.size, (255, 255, 255, 255))
                white.paste(hq_image, mask=hq_image.split()[3])
                try:
                    refined_prompt = qwen_evaluate_image(
                        white.convert("RGB"), prompt, user_intent,
                        device, args.qwen_model, args.qwen_tokens,
                    )
                except Exception as exc:
                    log(f"  Qwen evaluation failed; using current image ({type(exc).__name__})")
                    refined_prompt = None
            if refined_prompt is None or refined_prompt == prompt:
                print(f"  {_G}✓ image OK{_R}", flush=True)
                return hq_image, prompt, seed

            log("Qwen validation refined:")
            print_prompt(refined_prompt, previous=prompt)
            prompt = refined_prompt
            seed = random.randint(0, 9999)
            save_text(output_dir / "01_qwen_controlnet_prompt.txt", prompt)
        log(f"  Qwen validation reached {args.max_refine_attempts} attempt(s); using last image")
        if last_image is not None:
            save_text(output_dir / "01_qwen_controlnet_prompt.txt", last_prompt)
            return last_image, last_prompt, last_seed
    finally:
        del pipe
        if device == "mps":
            torch.mps.empty_cache()
    return run_controlnet(
        sketch, prompt, output_dir, device, args.controlnet_steps, seed,
        stage_label=stage_prefix,
        guidance_scale=args.controlnet_guidance_scale,
        conditioning_scale=args.controlnet_conditioning_scale,
    ), prompt, seed


def decimate_mesh(mesh, target_faces: int = 10000):
    if len(mesh.faces) <= target_faces:
        return mesh
    log(f"  Decimating: {len(mesh.faces)} → {target_faces} faces")
    has_colors = mesh.visual.kind == "vertex" and mesh.visual.vertex_colors is not None
    orig_verts = mesh.vertices.copy()
    orig_colors = mesh.visual.vertex_colors.copy() if has_colors else None
    target_reduction = 1.0 - (target_faces / len(mesh.faces))
    try:
        import fast_simplification
        verts, faces = fast_simplification.simplify(mesh.vertices, mesh.faces, target_reduction=target_reduction)
        mesh = tm_lib.Trimesh(vertices=verts, faces=faces)
    except Exception:
        import open3d as o3d
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(orig_verts)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_faces)
        mesh = tm_lib.Trimesh(vertices=np.asarray(o3d_mesh.vertices), faces=np.asarray(o3d_mesh.triangles))
    if has_colors and orig_colors is not None:
        from scipy.spatial import cKDTree
        tree = cKDTree(orig_verts)
        _, idx = tree.query(mesh.vertices)
        mesh.visual.vertex_colors = orig_colors[idx]
    return mesh


def soften_lighting(rgb: np.ndarray, alpha: np.ndarray, strength: float = 0.35) -> np.ndarray:
    if strength <= 0:
        return rgb
    fg = alpha[..., 0] > 0.02
    if not np.any(fg):
        return rgb

    luminance = (
        0.2126 * rgb[..., 0] +
        0.7152 * rgb[..., 1] +
        0.0722 * rgb[..., 2]
    )
    target = float(np.median(luminance[fg]))
    scale = np.ones_like(luminance)
    scale[fg] = (target / np.clip(luminance[fg], 0.08, 1.0)) ** strength
    return np.clip(rgb * scale[..., None], 0, 1)


def run_triposr(
    image: Image.Image,
    output_dir: Path,
    device: str,
    resolution: int = 256,
    tripo_input_size: int = 512,
    threshold: float = 25.0,
    foreground_ratio: float = 0.85,
    delight_strength: float = 0.35,
    fast: bool = False,
    target_faces: int = 10000,
    viewer_faces: int = 3000,
):
    import transformers
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    from tsr.system import TSR
    from tsr.utils import resize_foreground

    with timed_stage(f"[5] TripoSR 3D reconstruction (res={resolution})"):
        model = TSR.from_pretrained(TRIPOSR_MODEL_ID, config_name="config.yaml", weight_name="model.ckpt")
        model.renderer.set_chunk_size(2048 if fast else 8192)
        model.to(device)
        if fast:
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

        # Match TripoSR preprocessing: center/crop foreground, then composite onto neutral gray.
        rgba_input = image.convert("RGBA")
        if rgba_input.getchannel("A").getbbox() is not None:
            rgba_input = resize_foreground(rgba_input, foreground_ratio)
        arr = np.array(rgba_input).astype(np.float32) / 255.0
        rgb, alpha = arr[:, :, :3], arr[:, :, 3:4]
        rgb = soften_lighting(rgb, alpha, delight_strength)
        processed = rgb * alpha + 0.5 * (1 - alpha)
        processed = Image.fromarray((processed * 255.0).astype(np.uint8))
        processed = processed.resize((tripo_input_size, tripo_input_size), Image.LANCZOS)
        processed.save(output_dir / "03_triposr_input.png")

        with torch.no_grad():
            codes = model([processed], device=device)
        meshes = model.extract_mesh(codes, True, resolution=resolution, threshold=threshold)
        mesh = meshes[0]
        log(f"  Raw mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

        mesh_file = decimate_mesh(mesh, target_faces=target_faces)
        mesh_path = output_dir / "03_mesh.obj"
        mesh_file.export(str(mesh_path))
        log(f"  File mesh : {len(mesh_file.vertices)} verts, {len(mesh_file.faces)} faces → {mesh_path}")

        mesh_viewer = decimate_mesh(mesh, target_faces=viewer_faces)
        log(f"  View mesh : {len(mesh_viewer.vertices)} verts, {len(mesh_viewer.faces)} faces (for interactive viewer)")

        del model
        if device == "mps":
            torch.mps.empty_cache()
    return mesh_viewer


def _prepare_mesh_for_display(mesh, light_dir=None):
    m = tm_lib.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(),
                       visual=mesh.visual)
    m.apply_transform(tm_lib.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    m.apply_transform(tm_lib.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    verts = m.vertices - m.vertices.mean(0)
    max_r = np.abs(verts).max()
    if max_r > 0:
        verts /= max_r

    faces = m.faces
    has_vc = (hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None
              and np.array(mesh.visual.vertex_colors).shape[0] == len(mesh.vertices))
    base_fc = (np.array(mesh.visual.vertex_colors)[:, :3][mesh.faces].mean(1).astype(float) / 255
               if has_vc else np.full((len(faces), 3), [0.65, 0.72, 0.85]))

    # Gamma correction to brighten without changing hue
    base_fc = np.power(np.clip(base_fc, 1e-6, 1), 0.65)

    # Lambertian shading: compute face normals → dot with light
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0).astype(float)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(norms > 0, norms, 1)

    if light_dir is None:
        light_dir = np.array([0.6, 0.8, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)

    diffuse = np.clip(normals @ light_dir, 0, 1)[:, None]
    ambient = 0.65
    shading = ambient + (1 - ambient) * diffuse      # [0.65, 1.0]
    fc = np.clip(base_fc * shading, 0, 1)

    return verts, faces, fc, has_vc



def show_interactive_viewer(mesh, title: str = "TripoSR 3D Viewer", block: bool = True,
                            init_elev: float = 37, init_azim: float = -74, init_roll: float = 57):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts, faces, fc, has_vc = _prepare_mesh_for_display(mesh)

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(f"{title}  —  drag to rotate", fontsize=11)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"{len(faces):,} faces", fontsize=9)

    tri = Poly3DCollection(verts[faces], alpha=1.0, linewidths=0)
    tri.set_facecolor(fc)

    ax.add_collection3d(tri)
    margin = 1.2
    ax.set_xlim(-margin, margin); ax.set_ylim(-margin, margin); ax.set_zlim(-margin, margin)
    ax.set_box_aspect([1, 1, 1]); ax.set_axis_off()
    ax.view_init(elev=init_elev, azim=init_azim, roll=init_roll)

    plt.tight_layout()
    plt.show(block=block)
    if not block:
        plt.pause(0.1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sketch -> Qwen -> ControlNet -> rembg -> TripoSR")
    parser.add_argument("sketch", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("demo_outputs"))
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--qwen-model", default=QWEN_MODEL_ID)
    parser.add_argument("--prompt", default=None, help="Manual ControlNet prompt; skips Qwen")
    parser.add_argument("--edit", default=None, help="Natural-language edit applied to previous prompt")
    parser.add_argument("--skip-triposr", action="store_true", help="Stop after background removal")
    parser.add_argument("--no-viewer", action="store_true", help="Skip interactive 3D viewer")
    parser.add_argument("--no-refine", action="store_true", help="Skip Qwen validation after ControlNet")
    parser.add_argument("--max-refine-attempts", type=int, default=3)
    parser.add_argument("--controlnet-steps", type=int, default=10)
    parser.add_argument("--controlnet-guidance-scale", type=float, default=9.0)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=0.75)
    parser.add_argument("--triposr-resolution", type=int, default=256, help="256=fast(~10s), 512=hq(~60s)")
    parser.add_argument("--hq", action="store_true", help="TripoSR resolution 512 (high quality, slower)")
    parser.add_argument("--triposr-input-size", type=int, default=512)
    parser.add_argument("--triposr-threshold", type=float, default=25.0)
    parser.add_argument("--triposr-foreground-ratio", type=float, default=0.85)
    parser.add_argument("--triposr-delight-strength", type=float, default=0.35)
    parser.add_argument("--triposr-target-faces", type=int, default=10000)
    parser.add_argument("--triposr-viewer-faces", type=int, default=8000)
    parser.add_argument("--fast", action="store_true", help="TripoSR fast mode (smaller chunks)")
    parser.add_argument("--qwen-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    if not args.sketch.is_file():
        raise FileNotFoundError(args.sketch)
    if not args.no_refine and args.max_refine_attempts < 1:
        log("  max-refine-attempts < 1; using 1")
        args.max_refine_attempts = 1

    # Auto-number runs: demo_outputs/001/, 002/, ...
    base_dir = args.output_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    run_idx = 1
    while (base_dir / f"{run_idx:02d}").exists():
        run_idx += 1
    output_dir = ensure_dir(base_dir / f"{run_idx:02d}")

    device = get_device(args.device)
    sketch = Image.open(args.sketch).convert("RGB")
    sketch.save(output_dir / "00_input_sketch.png")
    log(f"Input sketch: {args.sketch}")
    log(f"Output dir: {output_dir}")
    log(f"Device: {device}")

    if args.prompt:
        prompt = sanitize_controlnet_prompt(args.prompt)
        save_text(output_dir / "01_qwen_controlnet_prompt.txt", prompt)
        log("[1] Manual prompt provided")
    elif args.edit:
        prompt_path = output_dir / "01_qwen_controlnet_prompt.txt"
        current = prompt_path.read_text(encoding="utf-8").strip()
        prompt, _ = rewrite_prompt_with_edit(current, args.edit, output_dir, device, args.qwen_model, args.qwen_tokens)
        if args.seed == 7:
            args.seed = random.randint(0, 9999)
            log(f"  Auto seed for edit: {args.seed}")
    else:
        prompt = make_controlnet_prompt(args.sketch, output_dir, device, args.qwen_model, args.qwen_tokens)

    log(f"[1] Prompt -> {output_dir / '01_qwen_controlnet_prompt.txt'}")
    _prev_prompt = ""
    if args.edit:
        # show diff against the prompt before the edit
        try:
            _prev_prompt = (output_dir / "01_qwen_controlnet_prompt.txt").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            pass
    print_prompt(prompt, previous=_prev_prompt)

    user_intent = args.edit or prompt
    hq_image, prompt, args.seed = generate_controlnet_image(
        sketch, prompt, output_dir, device, args, args.seed, user_intent
    )
    log("ControlNet foreground accepted in memory")

    if args.skip_triposr:
        log("[5] TripoSR skipped")
        log(f"Total elapsed: {format_seconds(time.perf_counter() - total_start)}")
        return

    resolution = 512 if args.hq else args.triposr_resolution
    mesh_viewer = run_triposr(
        hq_image, output_dir, device,
        resolution=resolution,
        tripo_input_size=args.triposr_input_size,
        threshold=args.triposr_threshold,
        foreground_ratio=args.triposr_foreground_ratio,
        delight_strength=args.triposr_delight_strength,
        fast=args.fast,
        target_faces=args.triposr_target_faces,
        viewer_faces=args.triposr_viewer_faces,
    )
    print(f"\n{_DM}Total elapsed: {format_seconds(time.perf_counter() - total_start)}{_R}")

    if args.no_viewer:
        return

    # ── Interactive edit loop ─────────────────────────────────────────────────
    show_interactive_viewer(mesh_viewer, title="Sketch → 3D", block=False)

    while True:
        try:
            print(f"\n{_B}Edit{_R} {_DM}(press Enter to quit){_R}")
            edit_input = input("▶ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not edit_input:
            break

        plt.close("all")  # close previous viewer before processing
        iter_start = time.perf_counter()

        current_prompt = (output_dir / "01_qwen_controlnet_prompt.txt").read_text(encoding="utf-8").strip()
        prompt, edit_en = rewrite_prompt_with_edit(current_prompt, edit_input, output_dir, device,
                                                   args.qwen_model, args.qwen_tokens)
        print_prompt(prompt, previous=current_prompt)
        seed = random.randint(0, 9999)

        hq_image, prompt, seed = generate_controlnet_image(
            sketch, prompt, output_dir, device, args, seed, edit_input
        )

        mesh_viewer = run_triposr(
            hq_image, output_dir, device,
            resolution=args.triposr_resolution,
            tripo_input_size=args.triposr_input_size,
            threshold=args.triposr_threshold,
            foreground_ratio=args.triposr_foreground_ratio,
            delight_strength=args.triposr_delight_strength,
            fast=args.fast,
            target_faces=args.triposr_target_faces,
            viewer_faces=args.triposr_viewer_faces,
        )
        print(f"  {_DM}done in {format_seconds(time.perf_counter() - iter_start)}{_R}")
        viewer_edit_title = edit_en if len(edit_en) <= 80 else edit_en[:77].rstrip() + "..."
        show_interactive_viewer(mesh_viewer, title=f"Edit: {viewer_edit_title}", block=False)

    plt.close("all")
    plt.show(block=True)

if __name__ == "__main__":
    main()
