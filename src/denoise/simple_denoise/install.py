import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(CURRENT_DIR, "../"))
sys.path.append(os.path.join(CURRENT_DIR, "../../"))

from denoise.config import check_user_settings
check_user_settings()
import shutil

# gen *.pyd at . location
cmd = "pip install {}".format(CURRENT_DIR)
# cmd = "pip install {}".format(CURRENT_DIR)

os.system(cmd)

# from simple_denoise.prepare_shaders import generate_OpenGL_shaders
# generate_OpenGL_shaders()
