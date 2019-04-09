from firedrake import *

mesh = UnitSquareMesh(128, 128, quadrilateral=True)

V = FunctionSpace(mesh, "DQ", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

x, y = SpatialCoordinate(mesh)

velocity = as_vector((0.5 - y, x - 0.5))
u = Function(W).interpolate(velocity)

bell_r0 = 0.15
bell_x0 = 0.25
bell_y0 = 0.5
cone_r0 = 0.15
cone_x0 = 0.5
cone_y0 = 0.25
cyl_r0 = 0.15
cyl_x0 = 0.5
cyl_y0 = 0.75
slot_left = 0.475
slot_right = 0.525
slot_top = 0.85

bell = 0.25*(1+cos(
    pi*min_value(sqrt(pow(x-bell_x0, 2) +
                      pow(y-bell_y0, 2))/bell_r0, 1.0))
)
cone = 1.0 - min_value(
    sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cyl_r0, 1.0
)
slot_cyl = conditional(
    sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
    conditional(And(And(x > slot_left,
                        x < slot_right), y < slot_top),
                0.0, 1.0), 0.0
)
q = Function(V).interpolate(1.0 + bell + cone + slot_cyl)
q_init = Function(V).assign(q)

outfile = File("results/DGadv.pvd")
outfile.write(q)

T = 2*pi
dt = T/1800.0
dtc = Constant(dt)
q_in = Constant(1.0)

dq_trial = TrialFunction(V)
phi = TestFunction(V)
a = phi*dq_trial*dx

n = FacetNormal(mesh)
un = 0.5*(dot(u, n) + abs(dot(u, n)))

L1 = dtc*(
    q*div(phi*u)*dx
    - conditional(dot(u, n) < 0, phi*dot(u, n)*q_in, 0.0)*ds
    - conditional(dot(u, n) > 0, phi*dot(u, n)*q, 0.0)*ds
    - (phi('+')-phi('-'))*(un('+')*q('+')-un('-')*q('-'))*dS
)
q1 = Function(V)
q2 = Function(V)
L2 = replace(L1, {q: q1})
L3 = replace(L1, {q: q2})

dq = Function(V)

params = {'ksp_type': 'preonly',
          'pc_type': 'bjacobi',
          'sub_pc_type': 'ilu'}
prb1 = LinearVariationalProblem(a, L1, dq)
solv1 = LinearVariationalSolver(prb1, solver_parameters=params)
prb2 = LinearVariationalProblem(a, L2, dq)
solv2 = LinearVariationalSolver(prb2, solver_parameters=params)
prb3 = LinearVariationalProblem(a, L3, dq)
solv3 = LinearVariationalSolver(prb3, solver_parameters=params)

t = 0.0
step = 0
while t < T - 0.5*dt:
    solv1.solve()
    q1.assign(q + dq)

    solv2.solve()
    q2.assign(0.75*q + 0.25*(q1 + dq))

    solv3.solve()
    q.assign((1.0/3.0)*q + (2.0/3.0)*(q2 + dq))

    step += 1
    t += dt

    if step % 20 == 0:
        outfile.write(q)
        print("t=", t)

L2_err = errornorm(q, q_init, norm_type="L2")
L2_init = norm(q_init, norm_type="L2")
print(L2_err/L2_init)
