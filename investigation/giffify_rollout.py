import os
from typing import Optional, Tuple

import imageio
from PIL import Image, ImageDraw

from investigation.plot_util import tuda
from investigation.rollout import compute_rollout
from investigation.util import ExperimentConfig, ExperimentResult


image_size_multiplier = 2
image_size = image_size_multiplier * 128
mass_offset = image_size_multiplier * 4
border_offset = image_size_multiplier * 6
mount_offset = image_size_multiplier * 2
pole_width = image_size_multiplier * 1
fps = 24



def giffify_rollout(out_dir: str, config: ExperimentConfig, result: ExperimentResult):
    _, (obs_rollout, _) = compute_rollout(config, result)

    with imageio.get_writer('%s/rollout.gif' % out_dir, mode = 'I', duration = 1 / fps) as gif_writer:
        for j, xy in enumerate(obs_rollout):
            true_xy = tuple(result.observations[:, j].flatten()) if j < config.T else None
            tau = j * config.h
            # noinspection PyTypeChecker
            image_path = _save_pendulum_image(out_dir, tau, xy, true_xy, j < config.T_train, prefix = 'rollout')
            gif_writer.append_data(imageio.imread(image_path))
            os.remove(image_path)



def _save_pendulum_image(out_dir: str, tau: float, xy: Tuple[float, float], true_xy: Optional[Tuple[float, float]], is_train: bool, prefix: str) -> str:
    xy_img = _compute_image_position(*xy)
    true_xy_img = None if true_xy is None else _compute_image_position(*true_xy)

    image = Image.new('RGB', (image_size, image_size), color = tuda('white'))
    draw = ImageDraw.Draw(image)
    if true_xy_img is not None:
        _draw_pendulum(draw, *true_xy_img, color = tuda('black'))
    _draw_pendulum(draw, *xy_img, color = tuda('blue') if is_train else tuda('orange'))
    draw.ellipse(((image_size / 2 - mount_offset, image_size / 2 - mount_offset), (image_size / 2 + mount_offset, image_size / 2 + mount_offset)), fill = tuda('black'))
    image_path = '%s/%s-%06.3f.bmp' % (out_dir, prefix, tau)
    image.save(image_path, format = 'BMP')
    return image_path



def _compute_image_position(x: float, y: float) -> Tuple[int, int]:
    # Scale the length of the stick to use more space of the screen.
    x *= (image_size / 2 - border_offset)
    y *= (image_size / 2 - border_offset)
    # Move the pendulum base to the center of the screen.
    x += image_size / 2
    y += image_size / 2
    # Clip the values to integers as screens are so natural...
    return int(x), int(y)



def _draw_pendulum(draw: ImageDraw.Draw, x: int, y: int, color: str) -> None:
    draw.ellipse(((x - mass_offset, y - mass_offset), (x + mass_offset, y + mass_offset)), fill = color)
    draw.line(((image_size / 2, image_size / 2), (x, y)), width = pole_width, fill = color)
