import os
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.colors import NoNorm
from matplotlib.cm import ScalarMappable

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # Enable OpenEXR support in OpenCV


def visualize_t_std(input_path, output_path):
    """
    Visualize the standard deviation of the interaction distance from the input EXR files.
    """
    img = cv2.imread(input_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img = np.mean(img[..., :3], axis=-1)  # Average across channels

    cmap = cm.get_cmap('plasma')
    norm=LogNorm(vmin=1e-3, vmax=1e0)
    # plt.imshow(img, cmap=cmap, norm=norm)
    # plt.colorbar()
    # plt.show()
    colored_img = (cmap(norm(img))[..., :3] * 255).astype(np.uint8)

    print(f"Visualizing: {input_path} -> {output_path}")
    cv2.imwrite(output_path, cv2.cvtColor(colored_img, cv2.COLOR_RGB2BGR))

    sm = ScalarMappable(norm=norm, cmap=cmap)
    fig, ax = plt.subplots(figsize=(6, 0.3))

    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    plt.savefig(output_path.replace('.png', '_colorbar.png'), dpi=400, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Path to input EXR files")
    args = parser.parse_args()

    visualize_t_std(args.input_dir, args.input_dir.replace('.exr', '_vis_t.png'))