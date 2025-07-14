import numpy as np
import imageio.v3 as iio
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def gen_diag_stripes(res, roughness_values):
    """
    Generate a roughness map in diagonal stripes with the given roughness values.
    ----------------------------
    |  1     /        /        |
    |     /        /        /  |
    |  /    2   /        /     |
    |        /        /        |
    |     /    3   /        /  |
    |  /        /   4    /     |
    |        /        /        |
    |     /        /    5   /  |
    |  /        /        /  6  |
    ----------------------------
    """
    part_res = res // 3

    img = np.zeros((res, res), dtype=np.float32)

    # Coordinate grid
    x = np.arange(res)
    y = np.arange(res)
    x, y = np.meshgrid(x, y)

    # Define the roughness map
    parts = []
    parts.append(x + y < part_res)
    parts.append(((x + y) >= part_res) & ((x + y) < 2 * part_res))
    parts.append(((x + y) >= 2 * part_res) & ((x + y) < 3 * part_res))
    parts.append(((x + y) >= 3 * part_res) & ((x + y) < 4 * part_res))
    parts.append(((x + y) >= 4 * part_res) & ((x + y) < 5 * part_res))
    parts.append((x + y) >= 5 * part_res)

    for i, part in enumerate(parts):
        img[part] = roughness_values[i]

    return img


def gen_blocks(res, roughness_values):
    """
    Generate a roughness map in blocks with the given roughness values.
    ----------------------------
    |  1     |  2     |  3     |
    |--------------------------|
    |  4     |  5     |  6     |
    ----------------------------
    """
    res_y = res // 2
    res_x = res // 3

    img = np.zeros((res, res), dtype=np.float32)

    for i in range(2):
        for j in range(3):
            img[i * res_y:(i + 1) * res_y, j * res_x:(j + 1) * res_x] = \
                roughness_values[3 * i + j]

    return img

            


if __name__ == "__main__":
    # Define the resolution and roughness values
    res = 600
    roughness_values = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

    # # Generate the stripe map
    # img = gen_diag_stripes(res, roughness_values)
    # iio.imwrite("diagStripe.exr", img, extension=".exr")

    # Generate the block map
    img = gen_blocks(res, roughness_values)
    iio.imwrite("blockMap.exr", img, extension=".exr")