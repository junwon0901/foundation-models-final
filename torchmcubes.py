import numpy as np
import torch
from skimage import measure


def marching_cubes(volume, level):
    input_is_torch = isinstance(volume, torch.Tensor)

    if input_is_torch:
        device = volume.device
        volume_np = volume.detach().float().cpu().numpy()
    else:
        device = torch.device("cpu")
        volume_np = np.asarray(volume, dtype=np.float32)

    vertices, faces, normals, values = measure.marching_cubes(
        volume_np,
        level=level
    )

    vertices = torch.from_numpy(vertices[:, [2, 1, 0]].astype(np.float32)).to(device=device)
    faces = torch.from_numpy(faces.astype(np.int64)).to(device=device)

    return vertices, faces
