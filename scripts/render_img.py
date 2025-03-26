import mitsuba as mi
import drjit as dr
import numpy as np
import json
import argparse


mi.set_variant("cuda_rgb")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dynamic Neural Radiosity")
    parser.add_argument("-t", type=str, default="path")
    parser.add_argument("-s", type=int, default=1)
    parser.add_argument("-c", type=str, default="config.json")
    parser.add_argument("-m", type=str, default="")
    parser.add_argument("-o", type=str, default="hello.exr")

    args = parser.parse_args()
    # config = json.load(open(args.c, "r"))

    # prepare scene

    scene: mi.Scene = mi.load_file("scenes/veach-ajar/scene.xml")
    bbox = scene.bbox()

    # rendering
    
    size = scene.sensors()[0].film().size()
    img: mi.TensorXf = dr.zeros(mi.TensorXf, (size[1], size[0], 3))
    
    total_spp = 10240
    max_spp_per_iter = 1024 if args.t == "path" else 1
    iter_num = (total_spp + max_spp_per_iter - 1) // max_spp_per_iter
    spp = total_spp if iter_num == 1 else max_spp_per_iter
    
    # for j in range(10):
    from tqdm import tqdm
    for i in tqdm(range(iter_num)):
        # img += mi.render(dscene.scene, integrator=integrator, spp=spp, seed=i + 2)
        img += mi.render(scene, spp=spp, seed=np.random.randint(0, 1000000))
        dr.flush_malloc_cache()
    img = img / iter_num
    
    img = mi.Bitmap(img)
    img.write(args.o)
