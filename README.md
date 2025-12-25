# neural-cone
Neural Cone Tracing for Interactive Glossy Global Illumination

## Method Pipeline

* Construct a **hash grid** representing diffuse outgoing radiance and rough part of glossy shading points (like in NR/NRC).
* Trace a cone for glossy reflection lobe.
  * A **tri-plane** model is optimized to represent prefiltered radiance information. (Require a bilateral prefilter technique to prevent light leakage, considering geometry information)
  * Contributions along the cone is integrated in a (NeRF-like) volume rendering manner.
  * ~~We trace multiple rays in stationary relative direction (like in Unscented Kalman Filter), to simulate cone tracing. Any ray hit is considered a partial occlusion.~~
  * ~~Conduct SDF sphere tracing along main reflection direction, each local minimum less than radius is considered partial occlusion.~~
  * We trace multiple RHS rays according to BSDF, and aggregate them into stationary number of points (KMeans).
* Merge cone color (the smoother, the better) and model color (the rougher, the better) according to roughness for glossy shading points.

## Advantage scenarios

Glossy objects in complex incident radiance distributions.

Test cases:

1. Glossy objects in an environment map.
2. Veach door scene with arbitrarily glossy floor.

## How to render a scene

* Choose a `[config_name]`, default value: `ncr-4-2`
* Choose a `[scene_name]` from `['bathroom', 'cornell-box', 'living-room', 'kitchen', 'veach-ajar']`
* Compile our customized version of Mitsuba 3.5.2
  * Clone the [repo](https://github.com/Jerry18231174/mitsuba3-old) to ../official-submodules/
  * Build mitsuba3 according to this [tutorial](https://mitsuba.readthedocs.io/en/v3.5.2/src/developer_guide/compiling.html)
* Run `source scripts/activate_mitsuba.sh`
* Run `python render.py -c [config_name] -s [scene_name]`
* Select render mode: LHS
* * Preferred hyper-parameters for rendering:
  * `"n_glossy_rhs": 32`
  * `"n_kmeans_iter": 3`

## How to train a model for a new scene

* If a large part of the scene's surface is occluded (which leads to un-illuminated surface), please save several camera poses manually:
  * Move the camera to an unoccluded pose, click "save camera config"
  * Save all pose configs to `scenes/[scene_name]/camera_poses`
* Run `python train.py -c [config_name] -s [scene_name]`
* Preferred hyper-parameters for training:
  * `"n_glossy_rhs": 128`
  * `"n_kmeans_iter": 10`