from firedrake import *
R0 = 6371220.0
R = Constant(R0)             # Radius of the earth [m]
H = Constant(5960.0)         # Mean depth [m]
day = 24. * 60. * 60.        # Time in a day [s]
Omega = Constant(7.292E-5)   # Angular rotation rate [rads]
g = Constant(9.80616)        # Accel. due to gravity [m/s^s]
mesh_degree = 3    # Cubic coordinate field
refinements = 5    # Number of refinements
mesh = IcosahedralSphereMesh(R0, refinements,
                             degree=mesh_degree)
x = SpatialCoordinate(mesh)
global_normal = as_vector([x[0], x[1], x[2]])
mesh.init_cell_orientations(global_normal)
model_degree = 2    # Degree of the finite element complex
V1 = FunctionSpace(mesh, "BDM", model_degree)
V2 = FunctionSpace(mesh, "DG", model_degree - 1)
V = V1 * V2    # Mixed space for velocity and depth
f_expr = 2 * Omega * x[2] / R
Vf = FunctionSpace(mesh, "CG", mesh_degree)
f = Function(Vf).interpolate(f_expr)    # Coriolis frequency

u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = Constant(u_0)
u_exp = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
h_exp = H-((R*Omega*u_max+u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g
un = Function(V1).project(u_exp)
hn = Function(V2).interpolate(h_exp)

# Topography
rl = pi/9.0
lambda_x = atan_2(x[1]/R0, x[0]/R0)
lambda_c = -pi/2.0
phi_x = asin(x[2]/R0)
phi_c = pi/6.0
minarg = Min(pow(rl, 2),
             pow(phi_x - phi_c, 2) +
             pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - sqrt(minarg)/rl)
b = Function(V2, name="Topography").interpolate(bexpr)
hn -= b

# Max time to run the simulation [s]
tmax = 15 * day      # Run for 15 days
Dt = 450.0
dt = Constant(Dt)    # Time-step size [s]
up = Function(V1)
hp = Function(V2)

##############################################################################
# Set up depth advection solver (DG upwinded scheme)
##############################################################################

hps = Function(V2)
h = TrialFunction(V2)
phi = TestFunction(V2)
hh = 0.5 * (hn + h)
uh = 0.5 * (un + up)
n = FacetNormal(mesh)
uup = 0.5 * (dot(uh, n) + abs(dot(uh, n)))
Heqn = ((h - hn)*phi*dx - dt*inner(grad(phi), uh*hh)*dx
        + dt*jump(phi)*(uup('+')*hh('+')-uup('-')*hh('-'))*dS)
Hproblem = LinearVariationalProblem(lhs(Heqn), rhs(Heqn), hps)
lu_params = {'ksp_type': 'preonly',
             'pc_type': 'lu',
             'pc_factor_mat_solver_type': 'mumps'}
Hsolver = LinearVariationalSolver(Hproblem,
                                  solver_parameters=lu_params,
                                  options_prefix="H-advection")

##############################################################################
# Velocity advection (Natale et. al (2016) extended to SWE)
##############################################################################

ups = Function(V1)
u = TrialFunction(V1)
v = TestFunction(V1)
hh = 0.5 * (hn + hp)
ubar = 0.5 * (un + up)
uup = 0.5 * (dot(ubar, n) + abs(dot(ubar, n)))
uh = 0.5 * (un + u)
Upwind = 0.5 * (sign(dot(ubar, n)) + 1)
K = 0.5 * (inner(0.5 * (un + up), 0.5 * (un + up)))
both = lambda u: 2*avg(u)
outward_normals = CellNormal(mesh)
perp = lambda arg: cross(outward_normals, arg)
Ueqn = (inner(u - un, v)*dx + dt*inner(perp(uh)*f, v)*dx
        - dt*inner(perp(grad(inner(v, perp(ubar)))), uh)*dx
        + dt*inner(both(perp(n)*inner(v, perp(ubar))),
                   both(Upwind*uh))*dS
        - dt*div(v)*(g*(hh + b) + K)*dx)
Uproblem = LinearVariationalProblem(lhs(Ueqn), rhs(Ueqn), ups)
Usolver = LinearVariationalSolver(Uproblem,
                                  solver_parameters=lu_params,
                                  options_prefix="U-advection")

##############################################################################
# Linear solver for incremental updates
##############################################################################

HU = Function(V)
deltaU, deltaH = HU.split()
w, phi = TestFunctions(V)
du, dh = TrialFunctions(V)
alpha = 0.5
HUlhs = (inner(w, du + alpha*dt*f*perp(du))*dx
         - alpha*dt*div(w)*g*dh*dx
         + phi*(dh + alpha*dt*H*div(du))*dx)
HUrhs = -inner(w, up - ups)*dx - phi*(hp - hps)*dx
HUproblem = LinearVariationalProblem(HUlhs, HUrhs, HU)
params = {'ksp_type': 'preonly',
          'mat_type': 'aij',
          'pc_type': 'lu',
          'pc_factor_mat_solver_type': 'mumps'}
HUsolver = LinearVariationalSolver(HUproblem,
                                   solver_parameters=params,
                                   options_prefix="impl-solve")

# Write out initial fields
u_out = Function(V1, name="Velocity").assign(un)
h_out = Function(V2, name="Surface elevation").assign(hn + b)
outfile = File("results/W5/nonlinear_w5.pvd")
outfile.write(u_out, h_out)

# Write out orography field for visualization
File("results/W5/W5_b.pvd").write(b)

# Start time-loop
t = 0.0
dumpfreq = 10    # Dump output every 10 steps
counter = 1
k_max = 4        # Maximum number of Picard iterations
while t < tmax:
    t += Dt
    up.assign(un)
    hp.assign(hn)

    # Start picard cycle
    for i in range(k_max):
        # Advect to get candidates
        Hsolver.solve()
        Usolver.solve()

        # Linear solve for updates
        HUsolver.solve()

        # Increment updates
        up += deltaU
        hp += deltaH

    # Update fields for next time step
    un.assign(up)
    hn.assign(hp)

    counter += 1
    if counter > dumpfreq:
        u_out.assign(un)
        h_out.assign(hn + b)
        outfile.write(u_out, h_out)
        counter -= dumpfreq
