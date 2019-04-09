from firedrake import *
mesh = UnitSquareMesh(64, 64)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V*Q

v, q = TestFunctions(W)

w = Function(W)
u, p = split(w)

nu = Constant(0.0001)

F = nu*inner(grad(u), grad(v))*dx - p*div(v)*dx - div(u)*q*dx

x, y = SpatialCoordinate(mesh)
forcing = as_vector([0.25 * x**2 * (2-x)**2 *y**2, 0])

bcs = [DirichletBC(W.sub(0), forcing, 4),
       DirichletBC(W.sub(0), zero(mesh.geometric_dimension()),
                   (1, 2, 3))]

const_basis = VectorSpaceBasis(constant=True)
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), const_basis])

solver_parameters = {
    "snes_type": "ksponly",
    "ksp_rtol": 1e-8,
    "ksp_type": "gmres",
    "ksp_monitor": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
    "fieldsplit_1": {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "__main__.MassMatrix",
        "mass_pc_type": "bjacobi",
        "mass_sub_pc_type": "ilu",
    }
}


class MassMatrix(AuxiliaryOperatorPC):
    _prefix = "mass_"
    def form(self, pc, test, trial):
        nu = self.get_appctx(pc)["nu"]
        a = -1/nu*inner(test, trial)*dx
        return (a, None)


appctx = {"nu": nu}
solve(F == 0, w,
      bcs=bcs,
      nullspace=nullspace,
      appctx=appctx,
      solver_parameters=solver_parameters)

u_h, p_h = w.split()
plot(u_h)
import matplotlib.pyplot as plt
plt.show()
