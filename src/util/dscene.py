import numpy as np
import mitsuba as mi
mi.set_variant("cuda_rgb")


def set_anim_vars(params: mi.SceneParameters, anim_dict: dict, v: dict):
    """
    Set the rho values for different materials
    """
    result = {}

    for key, val in anim_dict.items():
        assert isinstance(val, list)

        if len(val) == 2:
            # Linear mapping
            rho = v[key] * (val[1] - val[0]) + val[0]
        elif len(val) == 3:
            # Exponential mapping
            rho = val[0] * val[1] ** (v[key] * val[2])
        else:
            raise ValueError("Rho value must be a list of length 2 or 3.")
        
        result[key] = rho

        params[key] = mi.Float(rho)

    params.update()

    return result

