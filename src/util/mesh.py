
import os
from tqdm import tqdm
from typing import List, Dict

import torch
import numpy as np

import pymeshlab
import pymeshlab.pmeshlab

import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")


"""
Utility functions
"""
def rectangle_to_mesh(shape: mi.Shape):
    # Generate a rectangle mesh in xy plane
    vertex_pos = np.array(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32
    )
    face_indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    vertex_pos = mi.Point3f(vertex_pos)
    face_indices = mi.Point3u(face_indices)
    
    mesh = mi.Mesh(shape.id(),
        vertex_count=4,
        face_count=2,
        has_vertex_normals=True,
        has_vertex_texcoords=False,
    )
    mesh_params = mi.traverse(mesh)
    shape_params = mi.traverse(shape)
    
    # Apply transformation
    if shape_params.get("to_world") is None:
        mesh_params["vertex_positions"] = dr.ravel(vertex_pos)
    else:
        mesh_params["vertex_positions"] = dr.ravel(shape_params["to_world"] @ vertex_pos)
    mesh_params["faces"] = dr.ravel(face_indices)
    mesh_params.update()
    
    return mesh

def merge_mesh(shapes: List[mi.Shape]) -> mi.Mesh:
    """
    Merge multiple meshes into a single mesh.
    """
    vertices = []
    faces = []
    vertex_count = 0
    face_count = 0

    for shape in shapes:
        if isinstance(shape, mi.Mesh):
            param = mi.traverse(shape)
            vertices.append(param["vertex_positions"].numpy())
            faces.append(param["faces"].numpy() + vertex_count)
            vertex_count += shape.vertex_count()
            face_count += shape.face_count()

    vertices = np.concatenate(vertices, axis=0)
    faces = np.concatenate(faces, axis=0)

    mesh = mi.Mesh(
        "veach-ajar",
        vertex_count=vertex_count,
        face_count=face_count,
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )
    mesh_params = mi.traverse(mesh)
    mesh_params["vertex_positions"] = mi.Float32(vertices)
    mesh_params["faces"] = mi.UInt32(faces)
    mesh_params.update()
    
    return mesh

"""
Main functions
"""
def meshify(scene_dir: str):
    mi_scene = mi.load_file(os.path.join(scene_dir, "scene.xml"))
    mi_param = mi.traverse(mi_scene)

    mesh_dir = os.path.join(scene_dir, "raw_meshes")
    if not os.path.exists(mesh_dir):
        os.mkdir(mesh_dir)
    
    shapes = mi_scene.shapes()
    meshes = []
    print("Meshifying...")
    for shape in shapes:
        shape_type = shape.shape_type()
        # Check shape types, turn them into meshes
        if shape_type == int(mi.ShapeType.Mesh):
            meshes.append((shape.id(), shape))
        elif shape_type == int(mi.ShapeType.BSplineCurve):
            raise NotImplementedError("BSplineCurve cannot be meshified.")
        elif shape_type == int(mi.ShapeType.Cylinder):
            raise NotImplementedError("Cylinder cannot be meshified.")
        elif shape_type == int(mi.ShapeType.Disk):
            raise NotImplementedError("Disk cannot be meshified.")
        elif shape_type == int(mi.ShapeType.LinearCurve):
            raise NotImplementedError("LinearCurve cannot be meshified.")
        elif shape_type == int(mi.ShapeType.Rectangle):
            meshes.append((shape.id(), rectangle_to_mesh(shape)))
        elif shape_type == int(mi.ShapeType.SDFGrid):
            raise NotImplementedError("SDFGrid cannot be meshified.")
        elif shape_type == int(mi.ShapeType.Sphere):
            raise NotImplementedError("Sphere cannot be meshified.")
        elif shape_type == int(mi.ShapeType.Other):
            raise NotImplementedError("Other shape cannot be meshified.")
        else:
            raise ValueError("Unknown shape type:", shape_type)
    
    # Write meshes to files
    surface_areas = {}
    for name, mesh in meshes:
        surface_areas[name] = mesh.surface_area().numpy()[0]
        mesh.write_ply(os.path.join(mesh_dir, name + ".ply"))
    
    # Merge meshes
    meshes = []
    for file in os.listdir(mesh_dir):
        if file.endswith(".ply"):
            mesh = mi.load_dict({
                "type": "ply",
                "filename": os.path.join(mesh_dir, file),
            })
            meshes.append(mesh)
    merge_mesh(meshes).write_ply(os.path.join(mesh_dir, "merged.ply"))

    return surface_areas

def remesh(scene_dir: str, surface_areas: Dict[str, float]):
    mesh_dir = os.path.join(scene_dir, "raw_meshes")
    """
    Note: The following directory name is "meshes",
    since the same name is hardcoded in mi.dict_to_xml(),
    Naming the same to avoid duplicate files.
    """
    remesh_dir = os.path.join(scene_dir, "meshes")
    if not os.path.exists(remesh_dir):
        os.mkdir(remesh_dir)
    
    # Isotropic Explicit Remeshing
    print("Isotropic Explicit Remeshing...")
    mesh = pymeshlab.MeshSet()
    for file in tqdm(os.listdir(mesh_dir)):
        print(file)
        if file.endswith(".ply"):
            mesh.load_new_mesh(os.path.join(mesh_dir, file))
            targetlen = pymeshlab.PercentageValue(0.5 + 10 / (surface_areas[file[:-4]] ** 0.5 + 1))
            mesh.meshing_isotropic_explicit_remeshing(targetlen=targetlen)
            try:
                mesh.meshing_repair_non_manifold_edges(method="Split Vertices")
                mesh.meshing_cut_along_crease_edges(angledeg=70)
            except pymeshlab.pmeshlab.PyMeshLabException as e:
                print("Error:", e)
            else:
                pass
            mesh.save_current_mesh(os.path.join(remesh_dir, "IER_" + file))
    
    # Merge meshes
    meshes = []
    for file in os.listdir(remesh_dir):
        if file.endswith(".ply"):
            mesh = mi.load_dict({
                "type": "ply",
                "filename": os.path.join(remesh_dir, file),
            })
            meshes.append(mesh)
    merge_mesh(meshes).write_ply(os.path.join(remesh_dir, "merged.ply"))
    
    return None

def gen_feature_scene(scene_dir: str):
    if not os.path.exists(os.path.join(scene_dir, "meshes")):
        raise FileNotFoundError("Remeshed meshes not found.")
    
    scene_dict = {
        "type": "scene",
        "integrator": {"type": "path"},
        "light": {"type": "constant"},
        "sensor": {
            "type": "perspective",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0, -5, 5], target=[0, 0, 0], up=[0, 0, 1]
            ),
        },
        "merged_mesh": {
            "type": "ply",
            "filename": os.path.join(scene_dir, "meshes", "merged.ply"),
        },
    }

    feature_scene = mi.load_dict(scene_dict)
    mi.xml.dict_to_xml(scene_dict, os.path.join(scene_dir, "feature_scene.xml"))
    return feature_scene

def gen_remeshed_scene(scene_dir: str):
    """
    Replace shapes from original scene with remeshed meshes.
    Directly modify the XML file.
    """
    identity_str = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
    if not os.path.exists(os.path.join(scene_dir, "meshes")):
        raise FileNotFoundError("Remeshed meshes not found.")
    
    import xml.etree.ElementTree as ET
    tree = ET.parse(os.path.join(scene_dir, "scene.xml"))
    root = tree.getroot()

    for shape in root.findall("shape"):
        print(shape.get("id"))
        original_type = shape.get("type")
        shape.set("type", "ply")

        if original_type == "ply" or original_type == "obj":
            filename = shape.find("string")
        else:
            filename = ET.Element("string")
            filename.set("name", "filename")
            shape.append(filename)
        filename.set("value", os.path.join("meshes", "IER_" + shape.get("id") + ".ply"))

        shape.find("transform").find("matrix").set("value", identity_str)
    
    tree.write(os.path.join(scene_dir, "remeshed_scene.xml"))
    remeshed_scene = mi.load_file(os.path.join(scene_dir, "remeshed_scene.xml"))

    return remeshed_scene


if __name__ == "__main__":
    scene_dir = "scenes/veach-ajar"
    surface_areas = meshify(scene_dir)
