from firedrake import *

R0 = 6371220.0
R = Constant(R0)             # Radius of the earth [m]
H = Constant(5960.0)         # Mean depth [m]
day = 24. * 60. * 60.        # Seconds in a day [s]
Omega = Constant(7.292E-5)   # Angular rotation rate [rads]
g = Constant(9.80616)        # Accel. due to gravity [m/s^2]
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
u_0 = 2*pi*R0/(12*day)
u_max = Constant(u_0)  # Maximum amplitude of the zonal wind
u_exp = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
h_exp = H-((R*Omega*u_max+u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g
u0 = Function(V1).project(u_exp)
h0 = Function(V2).interpolate(h_exp)
un = Function(V1).assign(u0)  # Fields at time-step n
hn = Function(V2).assign(h0)
tmax = 5 * day
Dt = 1000.0
dt = Constant(Dt)                          # Timestep size [s]
alpha = Constant(0.5)                      # Midpoint method
u, h = TrialFunctions(V)
w, phi = TestFunctions(V)
outward_normal = CellNormal(mesh)
perp = lambda u: cross(outward_normal, u)  # Perp operator
uh_eqn = (inner(w, u) + alpha*dt*inner(w, f*perp(u))
          - alpha*dt*g*div(w)*h
          - inner(w, un)
          + alpha*dt*inner(w, f*perp(un))
          - alpha*dt*g*div(w)*hn
          + phi*h + alpha*dt*H*phi*div(u)
          - phi*hn + alpha*dt*H*phi*div(un))*dx
wn1 = Function(V)         # mixed func. for both fields (n+1)
un1, hn1 = wn1.split()    # Split func. for individual fields
uh_problem = LinearVariationalProblem(lhs(uh_eqn),
                                      rhs(uh_eqn), wn1)
params = {'mat_type': 'aij',
          'ksp_type': 'preonly',
          'pc_type': 'lu',
          'pc_factor_mat_solver_type': 'mumps'}
uh_solver = LinearVariationalSolver(uh_problem,
                                    solver_parameters=params)
# Write out initial fields
u_out = Function(V1, name="Velocity").assign(un)
h_out = Function(V2, name="Depth").assign(hn)
outfile = File("results/W2/linear_w2.pvd")
outfile.write(u_out, h_out)
t = 0.0
dumpfreq = 10    # Dump output every 10 steps
counter = 1
Uerrors = []
Herrors = []
t_array = []
# Energy functional
energy = 0.5*inner(un, un)*H*dx + 0.5*g*hn*hn*dx
energy_0 = assemble(energy)
energy_t = []
while t < tmax:  # Start time loop
    t += Dt
    t_array.append(t)
    uh_solver.solve()  # Solve for updated fields

    # Fields are solved in physical coordinates,
    # so we normalise errors by norm of the initial fields
    u0norm = norm(u0, norm_type="L2")
    h0norm = norm(h0, norm_type="L2")
    Uerr = errornorm(un1, un, norm_type="L2")/u0norm
    Herr = errornorm(hn1, hn, norm_type="L2")/h0norm
    Uerrors.append(Uerr)
    Herrors.append(Herr)

    # Use updated fields in the RHS for the next timestep
    un.assign(un1)
    hn.assign(hn1)

    # If energy is conserved, this should be close to 0
    energy_t.append((assemble(energy) - energy_0)/energy_0)

    counter += 1
    if counter > dumpfreq:
        u_out.assign(un)
        h_out.assign(hn)
        outfile.write(u_out, h_out)
        counter -= dumpfreq


# plotting
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 20})
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))


def make_a_plot(x_array, y_array, xlabel, ylabel,
                xlim, ylim, data_label, name):

    fig, (axes,) = plt.subplots(1, 1, figsize=(7, 7), squeeze=False)
    ax, = axes
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.plot(x_array, y_array, label=data_label)
    ax.legend(loc="upper right")
    fig.savefig(name,
                orientation="landscape",
                format="pdf",
                transparent=True,
                bbox_inches="tight")


t_array = [sec/day for sec in t_array]

# Plot velocity errors
c = 1.25
make_a_plot(t_array, Uerrors,
            xlabel="Time (days)", ylabel="Normalised $L^2$ errors",
            xlim=[0, 5],
            ylim=[min(Uerrors)/c, max(Uerrors)*c],
            data_label="Velocity errors",
            name="velocity_errors.pdf")

# Plot depth errors
make_a_plot(t_array, Herrors,
            xlabel="Time (days)", ylabel="Normalised $L^2$ errors",
            xlim=[0, 5],
            ylim=[min(Herrors)/c, max(Herrors)*c],
            data_label="Depth errors",
            name="depth_errors.pdf")

# Plot energy
# t_array.insert(0, 0.0)
make_a_plot(t_array, energy_t,
            xlabel="Time (days)", ylabel="Relative energy diff.",
            xlim=[0, 5],
            ylim=None,
            data_label="Relative energy diff.",
            name="energy.pdf")
