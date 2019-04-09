from firedrake import *
nlayers = 20                 # Number of extrusion layers
R = 6.371E6/125.0            # Scaled radius [m]: R_earth/125.0
thickness = 1.0E4            # Thickness [m] of the domain
degree = 1                   # Degree of finite element complex
refinements = 4              # Number of horizontal refinements
c = Constant(343.0)          # Speed of sound
N = Constant(0.01)           # Buoyancy frequency
Omega = Constant(7.292E-5)   # Angular rotation rate
dt = 36.0                    # Time-step size
tmax = 3600.0                # End time

# Horizontal base mesh (cubic coordinate field)
base = IcosahedralSphereMesh(R,
                             refinement_level=refinements,
                             degree=3)

# Extruded mesh
mesh = ExtrudedMesh(base,
                    extrusion_type='radial',
                    layers=nlayers,
                    layer_height=thickness/nlayers)

# Create tensor product complex:
# Horizontal elements
U1 = FiniteElement('RT', triangle, degree)
U2 = FiniteElement('DG', triangle, degree - 1)

# Vertical elements
V0 = FiniteElement('CG', interval, degree)
V1 = FiniteElement('DG', interval, degree - 1)

# HDiv element
W2_ele_h = HDiv(TensorProductElement(U1, V1))
W2_ele_v = HDiv(TensorProductElement(U2, V0))
W2_ele = W2_ele_h + W2_ele_v

# L2 element
W3_ele = TensorProductElement(U2, V1)

# Charney-Phillips element
Wb_ele = TensorProductElement(U2, V0)

# Resulting function spaces
W2 = FunctionSpace(mesh, W2_ele)
W3 = FunctionSpace(mesh, W3_ele)
Wb = FunctionSpace(mesh, Wb_ele)

x = SpatialCoordinate(mesh)

# Initial condition for velocity
u0 = Function(W2)
u_max = Constant(20.0)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
u0.project(uexpr)

# Initial condition for the buoyancy perturbation
lamda_c = 2.0*pi/3.0
phi_c = 0.0
W_CG1 = FunctionSpace(mesh, "CG", 1)
z = Function(W_CG1).interpolate(
    sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) - R
)
lat = Function(W_CG1).interpolate(
    asin(x[2]/sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]))
)
lon = Function(W_CG1).interpolate(atan_2(x[1], x[0]))
b0 = Function(Wb)
L_z = 20000.0
d = 5000.0
sin_tmp = sin(lat) * sin(phi_c)
cos_tmp = cos(lat) * cos(phi_c)
q = R*acos(sin_tmp + cos_tmp*cos(lon-lamda_c))
s = (d**2)/(d**2 + q**2)
bexpr = s*sin(2*pi*z/L_z)
b0.interpolate(bexpr)

# Initial condition for pressure
p0 = Function(W3).assign(0.0)

# Set up linear variational solver for u-p
# (After eliminating buoyancy)
W = W2 * W3
u, p = TrialFunctions(W)
w, phi = TestFunctions(W)

# radial unit vector
khat = interpolate(x/sqrt(dot(x, x)),
                   mesh.coordinates.function_space())

a_up = (dot(w, u)
        - 0.5*dt*p*div(w)
        # Appears after eliminating b
        + (0.5*dt*N)**2*dot(w, khat)*dot(u, khat)
        + phi*p + 0.5*dt*c**2*phi*div(u))*dx

L_up = (dot(w, u0)
        + 0.5*dt*dot(w, khat*b0)
        + phi*p0)*dx

bcs = [DirichletBC(W.sub(0), 0.0, "bottom"),
       DirichletBC(W.sub(0), 0.0, "top")]

w = Function(W)
up_problem = LinearVariationalProblem(a_up, L_up, w, bcs=bcs)
params = {
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'ksp_type': 'fgmres',
    'ksp_rtol': 1e-6,
    'ksp_monitor_true_residual': None,
    'pc_fieldsplit_schur': {
        'fact_type': 'FULL',
        'precondition': 'selfp'
    },
    'fieldsplit_0': {
        'ksp_type': 'preonly',
        'pc_type': 'bjacobi',
        'sub_pc_type': 'ilu'
    },
    'fieldsplit_1': {
        'ksp_type': 'cg',
        'pc_type': 'gamg',
        'pc_gamg_sym_graph': None,
        'mg_levels': {
            'ksp_type': 'chebyshev',
            'ksp_chebyshev_esteig': None,
            'ksp_max_it': 5,
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu'
        }
    }
}
up_solver = LinearVariationalSolver(up_problem,
                                    solver_parameters=params)

# Buoyancy solver
gamma = TestFunction(Wb)
b = TrialFunction(Wb)

a_b = gamma*b*dx
L_b = dot(gamma*khat, u0)*dx

b_update = Function(Wb)
b_problem = LinearVariationalProblem(a_b, L_b, b_update)
ilu_params = {
    'ksp_type': 'cg',
    'pc_type': 'bjacobi',
    'sub_pc_type': 'ilu'
}
b_solver = LinearVariationalSolver(b_problem,
                                   solver_parameters=ilu_params)

# Time-loop
t = 0
state = Function(W2 * W3 * Wb, name="state")
un1, pn1, bn1 = state.split()
un1.assign(u0)
pn1.assign(p0)
bn1.assign(b0)
output = File("results/gravity_waves.pvd")
output.write(un1, pn1, bn1)
count = 1
dumpfreq = 50
while t < tmax:
    t += dt

    # Solve for velocity and pressure updates
    up_solver.solve()
    un1.assign(w.sub(0))
    pn1.assign(w.sub(1))
    u0.assign(un1)
    p0.assign(pn1)

    # Reconstruct buoyancy
    b_solver.solve()
    bn1.assign(assemble(b0 - 0.5*dt*N**2*b_update))
    b0.assign(bn1)

    count += 1
    if count > dumpfreq:
        # Write output
        output.write(un1, pn1, bn1)
        count -= dumpfreq
