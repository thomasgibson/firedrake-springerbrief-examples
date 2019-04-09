from firedrake import *
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(16, 16)

U = VectorFunctionSpace(mesh, "CG", 1)
V = FunctionSpace(mesh, "DG", 0)
W = U * V

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)
f = Function(V)
x, y = SpatialCoordinate(mesh)

f.interpolate(-2*(x-1)*x - 2*(y-1)*y)

a = (dot(tau, sigma) - u*div(tau) + div(sigma)*v)*dx
L = f*v*dx
w = Function(W, name="Solution")

solve(a == L, w, solver_parameters={'ksp_rtol': 1e-3})
sigma_h, u_h = w.split()

plot(u_h)
plt.show()
