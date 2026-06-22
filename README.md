# Sketch-to-3D Foundation Model Demo

This demo turns a simple sketch into a 3D mesh using a multi-model pipeline:

1. Qwen generates a short image prompt from the sketch.
2. ControlNet Scribble generates an object image from the sketch and prompt.
3. rembg removes the background.
4. Qwen validates the generated image and can refine the prompt.
5. TripoSR reconstructs a 3D mesh and opens an interactive viewer.

The project is intended to run on macOS, Linux, or Windows. CUDA is recommended; Apple MPS and CPU are also supported, but CPU will be slow.

## Setup

```bash
git clone https://github.com/junwon0901/foundation-models-midterm.git
cd foundation-models-midterm

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirement.txt
```

For CUDA, install the PyTorch build that matches your driver from the official PyTorch instructions, then install the rest of the requirements.

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

After the first mesh is shown, type an edit prompt in the terminal, for example:

```text
make it blue
```

Press Enter on an empty edit prompt to stop.

## Outputs

Each run writes to a numbered directory under `demo_outputs/`.

Main files:

- `00_input_sketch.png`
- `01_qwen_controlnet_prompt.txt`
- `02_control_scribble.png`
- `02_controlnet_image_raw.png`
- `03_triposr_input.png`
- `03_mesh.obj`

Generated outputs are ignored by git.

## Notes

- The first run downloads model weights from Hugging Face.
- The interactive viewer requires a GUI display. Use `--no-viewer` on headless servers.
- TripoSR code is vendored under `TripoSR/` for the demo wrapper; model weights are still loaded from Hugging Face.
