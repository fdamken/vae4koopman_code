import argparse
import os
import shutil
from typing import Tuple

import imageio
import numpy as np
import scipy.integrate as sci
from PIL import Image, ImageDraw


d = 0.1

t_final = 2 * 50.0  # s
h = 0.1
T = int(t_final / h)
N = 1
initial_value_mean = np.array([0.0872665, 0.0])
initial_value_cov = np.diag([np.pi / 8.0, 0.0])
observation_cov = 0.0

image_size = 16
mass_offset = 1
border_offset = int(50 / 10.0)



def sample_dynamics() -> Tuple[np.ndarray, np.ndarray]:
    ode = lambda t, x: np.asarray([x[1], np.sin(x[0]) - d * x[1]])
    sequences = []
    for _ in range(0, N):
        initial_value = np.random.multivariate_normal(initial_value_mean, initial_value_cov)
        sequences.append(sci.solve_ivp(ode, (0, t_final), initial_value, t_eval = np.arange(0, t_final, h), method = 'Radau').y.T)
    sequences = np.asarray(sequences)
    sequences_noisy = sequences + np.random.multivariate_normal(np.array([0.0]), np.array([[observation_cov]]), size = sequences.shape).reshape(sequences.shape)
    return sequences, sequences_noisy



def save_pendulum_image(tau: float, phi: float, prefix: str) -> str:
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_directory', help = 'The name of the output directory.')
    parser.add_argument('-f', '--force', action = 'store_true', help = f'Overwrite output directory directory if it exists.')
    parser.add_argument('-s', '--silent', action = 'store_false', help = 'Disable showing of the plots (but still exporting the figures).')
    parser.add_argument('-c', '--seed', type = int, help = 'The seed use this to generate multiple deterministic pendulum sequences.')
    args = parser.parse_args()
    out_dir = args.output_directory
    show_plots = args.silent

    if os.path.exists(out_dir):
        if args.force:
            shutil.rmtree(out_dir)
        else:
            raise Exception('Output directory %s exists!' % out_dir)
    os.makedirs(out_dir)

    sequences, sequences_noisy = sample_dynamics()
    for i, (sequence, sequence_noisy) in enumerate(zip(sequences, sequences_noisy)):
        gif_writer = imageio.get_writer('%s/sequence-%05d.gif' % (out_dir, i), mode = 'I', duration = h)
        gif_writer_noisy = imageio.get_writer('%s/sequence-%05d_noisy.gif' % (out_dir, i), mode = 'I', duration = h)
        phis = sequence[:, 0]
        phis_noisy = sequence_noisy[:, 0]
        for j, (phi, phi_noisy) in enumerate(zip(phis, phis_noisy)):
            tau = j * h
            image_path = save_pendulum_image(tau, phi, prefix = 'sequence-%05d' % i)
            image_path_noisy = save_pendulum_image(tau, phi_noisy, prefix = 'sequence-%05d_noisy' % i)
            gif_writer.append_data(imageio.imread(image_path))
            gif_writer_noisy.append_data(imageio.imread(image_path_noisy))
        gif_writer.close()
        gif_writer_noisy.close()
