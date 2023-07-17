"""
The derivation is done using the Lagrange formalism. The DC motor is modeled with P behaviour (infinitely fast PT1).
The shafts are modeled with a thin rod and an addtional mass (m3) for the sensor at the end of the first shaft is
considered.

creates the following text files:
- 'dynamics_phi1_dd.txt', which contains the analytical equation of the second derivative of phi one
- 'dynamics_phi2_dd.txt', which contains the analytical equation of the second derivative of phi two
- 'dynamics_linearized_A.txt', which contains the analytical system matrix of the linearization of the system
- 'dynamics_linearized_b.txt', which contains the analytical input vector of the linearization of the system

generate_latex_equations can be set True to generate the latex equations as output.
This will generate the following textfiles:
- 'lagrange_system_A.txt', which contains the matrix A which results from using the lagrange formalism (Not to be confused with the linearized system matrix A!)
- 'lagrange_system_b.txt', which contains the vector b which results from using the lagrange formalism (Not to be confused with the linearized output vector b!)
"""
import sympy as smp
import sys
from sympy.tensor.array import derive_by_array
from sympy import latex

generate_latex_equations = False
if generate_latex_equations:
    t, g = smp.symbols('t g')
    # inverted pendulum kart
    b = smp.symbols('b')
    m, M = smp.symbols('m M')
    l = smp.symbols('l')
    # motor and transmission
    F = smp.symbols('F')
    """
    voltage = smp.symbols('u')
    J_motor = smp.symbols(r'J_m')  # motor inertia with respect to phi one (already multiplied with transmission_ratio^2)
    resistance = smp.symbols('R')
    inductance = smp.symbols('L')
    transmission_ratio = smp.symbols(r'\mathrm{tr}')
    ku = smp.symbols('k_u')
    km = smp.symbols('k_m')
    eta = smp.symbols(r'\eta')
    """
    # define states (functions of time)
    phi_f, s_f = smp.symbols(r'\phi, \s', cls=smp.Function)
else:
    # define parameter (constant values) names must not be changed sice eval() is used on them!
    # however, they can be temporarily changed into a Latex kompatible version for example
    t, g = smp.symbols('t g')
    # pendubot
    b = smp.symbols('b')
    m, M = smp.symbols('m M')
    l = smp.symbols('l')
    # motor and transmission
    F = smp.symbols('F')
    """
    voltage = smp.symbols('voltage')
    J_motor = smp.symbols('J_motor')  # motor inertia with respect to phi one (already multiplied with transmission_ratio^2)
    resistance = smp.symbols('resistance')
    inductance = smp.symbols('inductance')
    transmission_ratio = smp.symbols('transmission_ratio')
    ku = smp.symbols('ku')
    km = smp.symbols('km')
    eta = smp.symbols('eta')
    """
    # define states (functions of time)
    phi_f, s_f = smp.symbols('phi, s', cls=smp.Function)

phi_f = phi_f(t)
s_f = s_f(t)

# define the derivatives
dphi_f = smp.diff(phi_f, t)
ds_f = smp.diff(s_f, t)
ddphi_f = smp.diff(dphi_f, t)
dds_f = smp.diff(ds_f, t)

# motor equations, where the dynamics are assumed to be infinitely fast
#current = -1 / resistance * (ku * transmission_ratio * dphi1_f + voltage)
#M = transmission_ratio * eta * km * current

# get the cartesian coordinates and the center of gravity for second shaft
#x1 = L1 * smp.sin(phi1_f)
#y1 = -L1 * smp.cos(phi1_f)
#x2 = L1 * smp.sin(phi1_f) + L2 * smp.sin(phi2_f)
#y2 = -L1 * smp.cos(phi1_f) - L2 * smp.cos(phi2_f)

#s2_x = (x1 + x2) * smp.Rational(1, 2)
#s2_y = (y1 + y2) * smp.Rational(1, 2)

# Kinetic
J = smp.Rational(1, 3) * m * l**2
T = smp.Rational(1, 2) * (M + m) * ds_f ** 2 - smp.Rational(1, 2)* m * l * ds_f * dphi_f * smp.cos(phi_f) + smp.Rational(1, 8)*m*l**2*dphi_f**2

# Lagrangian

LE1 = ((M+m)*dds_f + b*ds_f + m*l*ddphi_f*smp.cos(phi_f) - m*l*dphi_f**2*smp.sin(phi_f)-F).simplify()
LE2 = ((J+m*l**2)*ddphi_f + m*g*l*smp.sin(phi_f) + m*l*dds_f*smp.cos(phi_f)).simplify()

if generate_latex_equations:
    A, b = smp.linear_eq_to_matrix([LE1, LE2], [ddphi_f, dds_f])
    with open('lagrange_system_A.txt', 'w') as f:
        f.write(latex(A))
    with open('LE1.txt', 'w') as f:
        f.write(latex(LE1))
    with open('LE2.txt', 'w') as f:
        f.write(latex(LE2))
    with open('lagrange_system_b.txt', 'w') as f:
        f.write(latex(b))
    print('generated latex equations.')
    print('If you want to generate the dynamics used in dynamics.py set generate_latex_equations to False')
    sys.exit()

# solve system using the fact that it is linear in the second derivatives

# without replacement of functions (is slower) names would not match and would need to be adjusted
# print('start linear solve')
# start = time()
# A, b = smp.linear_eq_to_matrix([LE1, LE2], [ddphi1_f, ddphi2_f])
# sol = A.solve(b)
# end = time()
# print(f'time needed: {end - start}')

# exchange time dependent functions with symbols (is faster)
phi, s, dphi, ds, ddphi, dds = smp.symbols('phi s dphi ds ddphi dds')
LE1_sym = LE1.subs({phi_f: phi, s_f: s, dphi_f: dphi, ds_f: ds, ddphi_f: ddphi, dds_f: dds})
LE2_sym = LE2.subs({phi_f: phi, s_f: s, dphi_f: dphi, ds_f: ds, ddphi_f: ddphi, dds_f: dds})


A, b = smp.linear_eq_to_matrix([LE1_sym, LE2_sym], [ddphi, dds])

print('start linear solve sym')
sol = A.solve(b)

# solve equations without linear information. Is slower and takes forever if functions are not replaced with symbols.
# print('start solve sym')
# start = time()
# sols = smp.solve([LE1_sym, LE2_sym], (ddphi1, ddphi2), simplify=False, rational=False)
# end = time()
# print(f'time needed: {end - start}')
print('save dynamics.')
with open('dynamics_phi_dd.txt', 'w') as f:
    f.write(str(sol[0].simplify()))
with open('dynamics_x_dd.txt', 'w') as f:
    f.write(str(sol[1].simplify()))

print('generate linear dynamics.')
dxdt_symbolic = [dphi, sol[0], ds, sol[1]]
A = derive_by_array(dxdt_symbolic, [phi, dphi, s, ds])
b = derive_by_array(dxdt_symbolic, [F])
A_simple = A.simplify()
print('save linear dynamics.')
with open('dynamics_linearized_A.txt', 'w') as f:
    f.write(str(A_simple))
with open('dynamics_linearized_b.txt', 'w') as f:
    f.write(str(b.simplify()))
print('done.')