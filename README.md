# Deep Learning Final Assignment

Sketch-to-3D foundation model demo: draw a sketch and get a 3D mesh using a multi-model pipeline.

```text
sketch image
→ Qwen prompt generation
→ ControlNet Scribble image generation
→ rembg foreground extraction
→ Qwen image validation / prompt refinement
→ TripoSR mesh reconstruction
→ interactive mesh viewer
```

## Setup

```bash
git clone https://github.com/junwon0901/foundation-models-final.git
cd foundation-models-final
```

Using venv:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirement.txt
```

For CUDA, install the matching PyTorch build from the official PyTorch site first, then run `pip install -r requirement.txt`.

## Run

```bash
python demo.py samples/sample_01.png
```

Useful options:

```bash
python demo.py samples/sample_01.png --device cuda
python demo.py samples/sample_01.png --device mps
python demo.py samples/sample_01.png --no-viewer
python demo.py samples/sample_01.png --max-refine-attempts 3
```

After the mesh is shown, type an edit prompt in the terminal and press Enter. Press Enter on an empty line to stop.

Example session:

```text
$ python demo.py samples/sample_01.png --device cuda
[Qwen] prompt: "chair"
[ControlNet] generating image...
[rembg] removing background...
[Qwen] validating image... OK
[TripoSR] reconstructing mesh...
Mesh saved to demo_outputs/001/03_mesh.obj

Edit prompt (or Enter to stop): make it wooden
[Qwen] rewriting prompt: "wooden chair"
[ControlNet] regenerating...
...
```

## Models

| Stage | Model |
|---|---|
| Prompt generation / validation | `Qwen/Qwen3.5-0.8B` |
| Image generation | `runwayml/stable-diffusion-v1-5` |
| ControlNet | `lllyasviel/sd-controlnet-scribble` |
| Background removal | `rembg` with `birefnet-general` |
| 3D reconstruction | `stabilityai/TripoSR` |

## Outputs

Each run writes to a numbered directory under `demo_outputs/`:

- `00_input_sketch.png`
- `01_qwen_controlnet_prompt.txt`
- `02_control_scribble.png`
- `02_controlnet_image_raw.png`
- `03_triposr_input.png`
- `03_mesh.obj`

Generated outputs are ignored by git.

## Notes

- First run downloads model weights from Hugging Face.
- CUDA is recommended; Apple MPS (`--device mps`) and CPU are also supported (CPU will be slow).
- The interactive viewer requires a display. Use `--no-viewer` on headless servers.
- TripoSR code is vendored under `TripoSR/`; weights are still loaded from Hugging Face.
