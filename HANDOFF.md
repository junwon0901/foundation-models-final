# Handoff Notes

This repository contains a sketch-to-3D demo built as a simple model pipeline:

```text
sketch image
-> Qwen prompt generation
-> ControlNet Scribble image generation
-> rembg foreground extraction
-> Qwen image validation / prompt refinement
-> TripoSR mesh reconstruction
-> interactive mesh viewer
```

The main entry point is `demo.py`.

## Current Models

- Prompt generation / edit / validation: `Qwen/Qwen3.5-0.8B`
- Image generation: `runwayml/stable-diffusion-v1-5`
- ControlNet: `lllyasviel/sd-controlnet-scribble`
- Background removal: `rembg` with `birefnet-general`
- 3D reconstruction: `stabilityai/TripoSR`

Model IDs are defined near the top of `demo.py`.

## Important Behavior

- The Qwen prompt is intentionally short.
- Python assembles the final ControlNet prompt with:

```python
PROMPT_TEMPLATE = "{object}, {descriptors}, isometric view, no background"
```

- Qwen validation runs up to `--max-refine-attempts`, default `3`.
- If Qwen validation fails or does not improve the prompt, the demo continues instead of crashing.
- ControlNet saves only the raw generated image:

```text
02_controlnet_image_raw.png
```

- The rembg RGBA foreground is kept in memory and passed to TripoSR.
- The demo does not save separate white-background or RGBA viewing images.
- The interactive edit loop is part of the intended demo flow.

## Model Replacement Points

### Replace Qwen

Functions:

- `load_qwen`
- `qwen_generate`
- `make_controlnet_prompt`
- `rewrite_prompt_with_edit`
- `qwen_evaluate_image`

Expected behavior:

- `make_controlnet_prompt` should produce a short object prompt.
- `rewrite_prompt_with_edit` should preserve the current prompt unless the edit clearly changes it.
- `qwen_evaluate_image` should return `None` for OK, or a refined prompt string.

Do not add object/color-specific fallback dictionaries. The project should stay general.

### Replace ControlNet / Image Generator

Functions:

- `load_controlnet_pipe`
- `run_controlnet_inference`
- `run_controlnet`
- `generate_controlnet_image`

Expected input:

- A PIL sketch image.
- A short text prompt.

Expected output:

- A PIL image that can be sent to rembg.

Current important settings:

- `--controlnet-steps`: default `10`
- `--controlnet-guidance-scale`: default `9.0`
- `--controlnet-conditioning-scale`: default `0.75`

### Replace Background Removal

Function:

- `run_controlnet`

Current behavior:

- Calls `rembg.remove(...)`.
- Returns an RGBA foreground image.

Any replacement should return a PIL RGBA image.

### Replace TripoSR

Function:

- `run_triposr`

Expected input:

- PIL RGBA image.

Expected output:

- A mesh object for the viewer.
- `03_mesh.obj` saved to the output directory.

The current preprocessing composites the foreground onto neutral gray before reconstruction.

## Output Directory

Each run creates a numbered directory under:

```text
demo_outputs/
```

This directory is ignored by git.

Useful files inside a run:

- `00_input_sketch.png`
- `01_qwen_controlnet_prompt.txt`
- `02_control_scribble.png`
- `02_controlnet_image_raw.png`
- `03_triposr_input.png`
- `03_mesh.obj`
- `edit_request.json`

## Cross-Platform Notes

- CUDA is recommended.
- Apple MPS is supported through `--device mps`.
- CPU can run but will be slow.
- Use `--no-viewer` on headless machines.
- The first run downloads model weights from Hugging Face.

## Things To Avoid

- Do not reintroduce `draw_pipeline.py`.
- Do not save extra white-background ControlNet preview images unless needed for debugging.
- Do not add hardcoded object/color replacement dictionaries.
- Do not make the prompt template long; SD1.5 is sensitive to prompt length.
