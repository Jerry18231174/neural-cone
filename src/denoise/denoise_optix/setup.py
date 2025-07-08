import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "../../"))
from denoise.config import _C as optix_cfg

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1"

os.environ['PATH'] = os.environ['PATH'] + os.pathsep + optix_cfg.PATH

# compile flags
c_flags = []
if sys.platform == "win32":
    c_flags.extend(["/DEBUG:NONE", "/O2"])
elif sys.platform == "linux":
    c_flags.extend(["-O3"])

ld_flags = []
if sys.platform == "win32":
    ld_flags.append("/NODEFAULTLIB:LIBCMT")
    ld_flags.append("/LIBPATH:{}/lib/x64/".format(optix_cfg.CUDA_PATH))
    ld_flags.append('cuda.lib')
    ld_flags.append('cudart_static.lib')
    ld_flags.append('advapi32.lib')
elif sys.platform == "linux":
    ld_flags.append("-L{}/lib64/stubs/".format(optix_cfg.CUDA_PATH))
    ld_flags.append("-lcuda")

ext_modules = [
    CUDAExtension(
        name='setup_optix_example',
        sources=['bind.cpp', 'denoiser.cpp', "optix_helper.cpp"],
        include_dirs=optix_cfg.INCLUDE_PATHS.split(";"),
        extra_compile_args={'cxx': c_flags},
        extra_link_args=ld_flags,
    ),
]

setup(
    name="setup_optix_example",
    version=__version__,
    author="banbao990",
    description="optix denoiser example",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.10",
)
