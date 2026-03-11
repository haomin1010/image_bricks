import os
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf

output_path = "/home/user/桌面/image2bricks/image_bricks/assets/dataset_v2/bordered_blue_block.usda"
stage = Usd.Stage.CreateNew(output_path)

# Root Xform
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root_path = "/Block"
root = UsdGeom.Xform.Define(stage, root_path)
stage.SetDefaultPrim(root.GetPrim())

# Add RigidBody API to Root
UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())

# Collision Cube (0.05m = 50mm)
col_path = f"{root_path}/collision"
col_cube = UsdGeom.Cube.Define(stage, col_path)
col_cube.GetSizeAttr().Set(0.05)
col_cube.CreatePurposeAttr(UsdGeom.Tokens.guide) # Guide means invisible to render
UsdPhysics.CollisionAPI.Apply(col_cube.GetPrim())

# Visual Cube (0.046m = 46mm, leaves 2mm gap on each side)
vis_path = f"{root_path}/visual"
vis_cube = UsdGeom.Cube.Define(stage, vis_path)
vis_cube.GetSizeAttr().Set(0.046)
vis_cube.CreatePurposeAttr(UsdGeom.Tokens.render)

# Create Material and Shader for the visual cube
material_path = f"{root_path}/material"
material = UsdShade.Material.Define(stage, material_path)

# OmniPBR MDL Shader
shader_path = f"{material_path}/shader"
shader = UsdShade.Shader.Define(stage, shader_path)
shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

# Deep Sky Blue explicitly passed into diffuse_color_constant using GfVec3f
diffuse_color = Gf.Vec3f(0.0, 0.75, 1.0)
shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(diffuse_color)
shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.9)
shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)

# Connect Material outputs to the shader
material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
material.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")

# Bind material to visual cube
UsdShade.MaterialBindingAPI(vis_cube).Bind(material)

stage.GetRootLayer().Save()
simulation_app.close()
print(f"Successfully created {output_path}")
