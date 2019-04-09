from firedrake import *

base_mesh = CubedSphereMesh(radius=6400.e6, refinement_level=5,
                            degree=2)
mesh = ExtrudedMesh(base_mesh, layers=20, layer_height=10,
                    extrusion_type="radial")
