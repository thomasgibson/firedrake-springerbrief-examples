from firedrake import *

N = 20
base_mesh = UnitIntervalMesh(N)
mesh = ExtrudedMesh(base_mesh, layers=N, layer_height=1./N)

P2 = FiniteElement("CG", interval, 2)
P3 = FiniteElement("CG", interval, 3)
H1_element = TensorProductElement(P2, P3)
H1 = FunctionSpace(mesh, H1_element)

dP1 = FiniteElement("DG", interval, 1)
dP2 = FiniteElement("DG", interval, 2)
L2_element = TensorProductElement(dP1, dP2)
L2 = FunctionSpace(mesh, L2_element)

S = TensorProductElement(P2, dP1)
Hcurl = FunctionSpace(mesh, HCurl(S))
Hdiv = FunctionSpace(mesh, HDiv(S))
