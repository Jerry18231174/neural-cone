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