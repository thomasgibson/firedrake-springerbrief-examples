from firedrake import *
nlayers = 16                # Number of extrusion layers
R = 6.371E6/125.0           # Scaled radius [m]: R_earth/125.0
thickness = 1.0E4           # Thickness [m] of the domain
degree = 1                  # Degree of finite element complex
refinements = 4             # Number of horizontal refinements
c = Constant(343.0)         # Speed of sound
N = Constant(0.01)          # Buoyancy frequency
g = Constant(9.810616)      # Accel. due to gravity (m/s^2)
N = Constant(0.01)          # Brunt-Vaisala frequency (1/s)
p_0 = Constant(1000.0*100)  # Reference pressure (Pa, not hPa)
c_p = Constant(1004.5)      # SHC of dry air at const. pressure
R_d = Constant(287.0)       # Gas constant for dry air (J/kg/K)
kappa = 2.0/7.0             # R_d/c_p
T_eq = 300.0                # Isothermal atmospheric temp. (K)
p_eq = 1000.0 * 100.0       # Ref surface pressure at equator
u_max = 20.0                # Maximum amp. of zonal wind (m/s)
d = 5000.0                  # Width parameter for Theta'
lamda_c = 2.0*pi/3.0        # Longitudinal centerpoint of Theta'
phi_c = 0.0                 # Latitudinal centerpoint of Theta'
deltaTheta = 1.0            # Maximum amplitude of Theta' (K)
L_z = 20000.0               # Vert. wave len. of the Theta' pert.

# Horizontal base mesh (cubic coordinate field)
base = CubedSphereMesh(R,
                       refinement_level=refinements,
                       degree=3)

# Extruded mesh
mesh = ExtrudedMesh(base,
                    extrusion_type='radial',
                    layers=nlayers,
                    layer_height=thickness/nlayers)

# Horizontal elements
U1 = FiniteElement('RTCF', quadrilateral, degree)
U2 = FiniteElement('DQ', quadrilateral, degree - 1)

# Vertical elements
V0 = FiniteElement('CG', interval, degree)
V1 = FiniteElement('DG', interval, degree - 1)

# HDiv element (vertical only)
W2_ele_v = HDiv(TensorProductElement(U2, V0))

# L2 element
W3_ele = TensorProductElement(U2, V1)

# Charney-Phillips element
Wt_ele = TensorProductElement(U2, V0)

# Resulting function spaces
W2v = FunctionSpace(mesh, W2_ele_v)
W3 = FunctionSpace(mesh, W3_ele)
Wtheta = FunctionSpace(mesh, Wt_ele)

theta0 = Function(Wtheta)
x = SpatialCoordinate(mesh)

# Create polar coordinates:
# Since we use a CG1 field, this is constant on layers
W_Q1 = FunctionSpace(mesh, "CG", 1)
z_expr = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) - R
z = Function(W_Q1).interpolate(z_expr)
lat_expr = asin(x[2]/sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]))
lat = Function(W_Q1).interpolate(lat_expr)
lon = Function(W_Q1).interpolate(atan_2(x[1], x[0]))

# Surface temperature
G = g**2/(N**2*c_p)
Ts_expr = G
Ts_expr += (T_eq-G)*exp(
    -(u_max*N**2/(4*g*g))*u_max*(cos(2.0*lat)-1.0)
)
Ts = Function(W_Q1).interpolate(Ts_expr)

# Surface pressure
ps_expr = p_eq*exp(
    (u_max/(4.0*G*R_d))*u_max*(cos(2.0*lat)-1.0)
)*(Ts/T_eq)**(1.0/kappa)
ps = Function(W_Q1).interpolate(ps_expr)

# Background pressure
p_expr = ps*(1 + G/Ts*(exp(-N**2*z/g)-1))**(1.0/kappa)
p = Function(W_Q1).interpolate(p_expr)

# Background temperature
Tb_expr = G*(1 - exp(N**2*z/g)) + Ts*exp(N**2*z/g)
Tb = Function(W_Q1).interpolate(Tb_expr)

# Background potential temperature
thetab_expr = Tb*(p_0/p)**kappa
thetab = Function(W_Q1).interpolate(thetab_expr)
theta_b = Function(Wtheta).interpolate(thetab)
sin_tmp = sin(lat) * sin(phi_c)
cos_tmp = cos(lat) * cos(phi_c)
r = R*acos(sin_tmp + cos_tmp*cos(lon-lamda_c))
s = (d**2)/(d**2 + r**2)
theta_pert = deltaTheta*s*sin(2*pi*z/L_z)
theta0.interpolate(theta_b)

khat = interpolate(x/sqrt(dot(x, x)),
                   mesh.coordinates.function_space())


# Calculate hydrostatic Pi
W = W2v * W3
v, pi = TrialFunctions(W)
dv, dpi = TestFunctions(W)

n = FacetNormal(mesh)

a = (c_p*inner(v, dv) - c_p*div(dv*theta0)*pi
     + dpi*div(v*theta0))*dx
L = (-c_p*inner(dv, n)*theta0*(p/p_0)**kappa*ds_t
     - c_p*g*inner(dv, khat)*dx)

bcs = [DirichletBC(W.sub(0), 0.0, "bottom")]

w = Function(W)
piproblem = LinearVariationalProblem(a, L, w, bcs=bcs)

params = {
    'mat_type': 'matfree',
    'pc_type': 'python',
    'pc_python_type': 'scpc.VerticalHybridizationPC',
    'ksp_type': 'preonly',
    'vert_hybridization': {
        'ksp_type': 'cg',
        'ksp_rtol': 1e-8,
        'pc_type': 'gamg',
        'pc_gamg_sym_graph': None,
        'mg_levels': {
            'ksp_type': 'richardson',
            'ksp_max_it': 5,
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu'
        }
    }
}
pisolver = LinearVariationalSolver(piproblem,
                                   solver_parameters=params)

pisolver.solve()
v, exner = w.split()

# Print the norm of v: should be close to zero
print(sqrt(assemble(inner(v, v)*dx)))
File("pressure.pvd").write(exner)
