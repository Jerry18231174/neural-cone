import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# download oidn
cmd = "python {}/download_resources.py".format(CURRENT_DIR)
os.system(cmd)

# install oidn
cmd = "pip install {}".format(CURRENT_DIR)
os.system(cmd)