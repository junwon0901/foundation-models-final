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

## Example Session

```text
$ python demo.py samples/sample_01.png --device mps

  Input sketch: samples/sample_01.png
  Output dir: demo_outputs/01
  Device: mps

[1] Qwen prompt generation
  Loading Qwen model: Qwen/Qwen3.5-0.8B on mps
  done in 13.3s
  Prompt: red apple, natural skin texture, small stem with leaf, isometric view, no background

[2] ControlNet (attempt 1) (10 steps)
  done in 19.3s

[3] Background removal
  done in 8.6s

[4] Qwen image evaluation (attempt 1)
  ✓ image OK

[5] TripoSR 3D reconstruction (res=256)
    File mesh: 5011 verts, 10000 faces → demo_outputs/01/03_mesh.obj
  done in 9.2s

Total elapsed: 1m 12.5s

Edit (press Enter to quit)
▶ 파란색으로 변경해 줘.
  Translated: Switch to blue.

[1] Qwen prompt rewrite
  blue apple, natural skin texture, small stem with leaf, isometric view, no background

[2] ControlNet (attempt 1) (10 steps)
  done in 20.0s

[3] Background removal
  done in 7.1s

[4] Qwen image evaluation (attempt 1)
  ✓ image OK

[5] TripoSR 3D reconstruction (res=256)
    File mesh: 5003 verts, 9999 faces → demo_outputs/01/03_mesh.obj
  done in 9.2s

Total elapsed: 1m 12.8s

Edit (press Enter to quit)
▶ 
```

## Results

### Input Sketch

![Input sketch](results/00_input_sketch.png)

### Before Edit — red apple

| ControlNet output | 3D mesh |
|---|---|
| ![ControlNet before](results/02_controlnet_before.png) | ![Mesh before](results/Tripo_result_before.png) |

### After Edit — blue apple

| ControlNet output | 3D mesh |
|---|---|
| ![ControlNet after](results/02_controlnet_after.png) | ![Mesh after](results/Tripo_result_after.png) |

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
- Korean edit prompts are supported via Helsinki-NLP/opus-mt-ko-en translation.
