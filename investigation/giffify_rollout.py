import os

import imageio
import numpy as np
from PIL import Image, ImageDraw

from investigation.rollout import compute_rollout
from investigation.util import ExperimentConfig, ExperimentResult


image_size = 16
mass_offset = 1
border_offset = int(50 / 10.0)



def giffify_rollout(out_dir: str, config: ExperimentConfig, result: ExperimentResult):
    _, (obs_rollout, _) = compute_rollout(config, result)

    phis = obs_rollout[:, 0]
    with imageio.get_writer('%s/rollout.gif' % out_dir, mode = 'I', duration = config.h) as gif_writer:
        for j, phi in enumerate(phis):
            tau = j * config.h
            image_path = _save_pendulum_image(out_dir, tau, phi, prefix = 'rollout')
            gif_writer.append_data(imageio.imread(image_path))
            os.remove(image_path)



def _save_pendulum_image(out_dir: str, tau: float, phi: float, prefix: str) -> str:
    # Calculate the x/y coordinates using basic trigonometry.
    x = np.sin(phi)
    y = -np.cos(phi)
    # Scale the length of the stick to use more space of the screen.
    x *= (image_size / 2 - border_offset)
    y *= (image_size / 2 - border_offset)
    # Move the pendulum base to the center of the screen.
    x += image_size / 2
    y += image_size / 2
    # Clip the values to integers as screens are so natural...
    x = int(x)
    y = int(y)

    image = Image.new('1', (image_size, image_size), color = 1)
    draw = ImageDraw.Draw(image)
    draw.ellipse(((x - mass_offset, y - mass_offset), (x + mass_offset, y + mass_offset)), fill = 0)
    image_path = '%s/%s-%06.3f.bmp' % (out_dir, prefix, tau)
    image.save(image_path, format = 'BMP')
    return image_path
