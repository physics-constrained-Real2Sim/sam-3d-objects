import sys
import os
import zarr
import numpy as np
import trimesh
from icecream import ic
import trimesh

import torch
# # import inference code
from sam3d_objects.pipeline.inference_utils import compose_transform,decompose_transform
from notebook.inference import Inference 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_reader import read_scene_zarr

from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, Transform3d

import cv2
from copy import deepcopy

from pytorch3d.transforms import quaternion_to_matrix
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

from tqdm import tqdm

_R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
_R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T

T_sam_to_real_world = np.array([
    [-0.545447, -0.664139,  0.511280,  0.782866],
    [-0.837612,  0.453687, -0.304259, -0.528839],
    [-0.029891, -0.594211, -0.803753,  0.745482],
    [ 0.0,       0.0,       0.0,       1.0     ]
], dtype=np.float64)


def make_scene_untextured_mesh(*outputs, in_place=False):
    import trimesh

    if not in_place:
        outputs = [deepcopy(output) for output in outputs]

    all_meshes = []
    for output in outputs:
        mesh = output["glb"]
        if mesh is None:
            continue

        # GLB is Y-up, transforms are Z-up; convert, apply, convert back
        vertices = mesh.vertices.astype(np.float32) @ _R_YUP_TO_ZUP
        vertices_tensor = torch.from_numpy(vertices).float().to(output["rotation"].device)
        R_l2c = quaternion_to_matrix(output["rotation"])
        l2c_transform = compose_transform(
            scale=output["scale"],
            rotation=R_l2c,
            translation=output["translation"],
        )
        vertices = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
        mesh.vertices = vertices.squeeze(0).cpu().numpy() @ _R_ZUP_TO_YUP
        all_meshes.append(mesh)

    if not all_meshes:
        return None

    if len(all_meshes) == 1:
        return all_meshes[0]

    return trimesh.util.concatenate(all_meshes)


def make_mesh_to_sam_scene(output, in_place=False):

    if not in_place:
        output = deepcopy(output)

    mesh = output["glb"]
    if mesh is None:
        return None

    # GLB is Y-up, transforms are Z-up; convert, apply, convert back
    vertices = mesh.vertices.astype(np.float32) @ _R_YUP_TO_ZUP
    vertices_tensor = torch.from_numpy(vertices).float().to(output["rotation"].device)
    R_l2c = quaternion_to_matrix(output["rotation"])
    l2c_transform = compose_transform(
        scale=output["scale"],
        rotation=R_l2c,
        translation=output["translation"],
    )
    vertices = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
    mesh.vertices = vertices.squeeze(0).cpu().numpy() @ _R_ZUP_TO_YUP
    
    return output


def make_mesh_to_real_world(output, in_place=False):

    output = make_mesh_to_sam_scene(output, in_place=in_place)
    mesh = output["glb"]
    if mesh is None:
        return None
    mesh.apply_transform(T_sam_to_real_world)
    
    return output

if __name__ == "__main__":
    
    dataset_dir = "../dataset/google5"
    zarr_path = "../dataset/google5/scene.zarr"  # 修改成你的路径
    output_dir = f"{dataset_dir}/SAM3D_recon"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


    data = read_scene_zarr(zarr_path)
    width, height, cam_position, fov_y_deg, target_look_at, cam_up_direction, near, far = data.get("metadata_camera", (None,)*7)
    rgb, depth, seg_mask, intrinsic, extrinsic_world_to_cam, extrinsic_cam_to_world = data.get("camera", (None,)*6)
    partial_points, partial_colors, partial_mask = data.get("partial", (None,)*3)
    complete_points, complete_mask = data.get("complete", (None,)*2)
    GT_objs = data.get("objects", [])
    # load model
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    depth[depth <= 0] = np.nan  

    H, W = depth.shape

    K = np.array(intrinsic)


    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # ------------------ PIXEL GRID ------------------
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    Z = depth
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    # ---------------------------------------------------------------------
    # Convert image coordinates (x→right, y→down) into PyTorch3D coordinates:
    #   PyTorch3D expects a right-handed camera frame with:
    #       +x → right, +y → UP, +z → forward.
    #   So we flip both X and Y:
    #       -Y  converts image Y-down into Y-up,
    #       -X  keeps the coordinate system right-handed.
    # ---------------------------------------------------------------------

    pointmap = np.stack([-X, -Y, Z], axis=-1)
    pointmaP = torch.tensor(pointmap, dtype=torch.float32)

    outputs = []
    for obj_idx, obj in enumerate(tqdm(GT_objs)):
        body_id = obj["bid"]
        object_mask = (seg_mask == body_id)
        output = inference(rgb, object_mask, seed=42, pointmap=pointmaP)
        # output = inference(rgb, object_mask, seed=42)
        body_reconstructed_path = f"{output_dir}/obj{body_id}"
        output_path = f"{body_reconstructed_path}/obj{body_id}.obj"
        
        if not os.path.exists(body_reconstructed_path):
            os.makedirs(body_reconstructed_path, exist_ok=True)
        ic(output.keys())
        output = make_mesh_to_real_world(output, in_place=True)
        output["glb"].export(output_path)
        # baked_mesh_path = bake_color_vertex_to_texture(filename = output_path)
        
        print(f"[SAM3D reconstruction]: Body {body_id} Mesh saved to {output_path}")
        # print(f"Baked mesh saved to {baked_mesh_path}")
        ## TODO: export texture mesh instead of vertices color mesh
        outputs.append(output)


    all_meshes = [output["glb"] for output in outputs if output["glb"] is not None]

    scene = trimesh.Scene()
    for m in all_meshes:
        scene.add_geometry(m)

    scene.export(f"{output_dir}/scene.glb")
    scene.export(f"{output_dir}/scene.obj")