#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root
from scipy.spatial import ConvexHull

from data import G_hcp, G_fcc, G_liq, G_CuMg2, G_Cu2Mg, X, X_CuMg2, X_Cu2Mg
from phase import Phase, calculate_eq3, calculate_eq2, calculate_eqline, Eq2Point

x = np.linspace(0, 1, 500)
y1, y2 = np.random.uniform(0, 1, size=(2, 100000))

fcc = Phase("fcc", X, G_fcc)
hcp = Phase("hcp", X, G_hcp)
liq = Phase("liq", X, G_liq)
cumg2 = Phase("CuMg2", X_CuMg2, G_CuMg2)
cu2mg = Phase("Cu2Mg", X_Cu2Mg, G_Cu2Mg)

phases = [fcc, hcp, liq, cumg2, cu2mg]

T0 = 273.15
Trange = np.arange(200, 1200, 10)


def at2wt(at):
    at = np.array(at)
    AMG, ACU = 24.305, 63.546
    return at * AMG / (at * AMG + (1 - at) * ACU)


from PIL import Image

plt.imshow(np.array(Image.open("CuMg_diagram.png"))[86:681, 123:915], extent=(0, 1, 200, 1200), aspect="auto")

ctx = {"X": x, "Y1": y1, "Y2": y2}

N = sum(np.array(p.X.eval(ctx)).size for p in phases)
points = np.empty((N, 2))
phase_id = np.empty(N, dtype=int)

for T in Trange:
    print(".", end="", flush=True)

    o = 0
    for i, phase in enumerate(phases):
        c = np.array(phase.X.eval(ctx))
        points[o:o + c.size, 0] = c
        points[o:o + c.size, 1] = phase.G.eval(ctx | {"T": T0 + T})
        phase_id[o:o + c.size] = i
        o += c.size

    hull = ConvexHull(points)
    mask = hull.equations[:, 1] < 0
    for a, b in hull.simplices[mask]:
        if phase_id[a] != phase_id[b] or abs(points[a, 0] - points[b, 0]) > 0.05:
            plt.plot(at2wt([points[a, 0], points[b, 0]]), [T, T], "C0.-", lw=0.5)
print()
plt.xlim(0, 1)


def solve_eq2_cu2mg(p: Phase, t0: float, x0: float):
    def fun(z):
        t, y1, y2, x, l = z
        ctx = {"T": t, "Y1": y1, "Y2": y2}
        return (
            p.G.eval({"T": t, "X": x}) - cu2mg.G.eval(ctx),
            p.G.derivative("X").eval({"T": t, "X": x}) - l,
            cu2mg.G.derivative("Y1").eval(ctx) - l*cu2mg.X.derivative("Y1").eval(ctx),
            cu2mg.G.derivative("Y2").eval(ctx) - l*cu2mg.X.derivative("Y2").eval(ctx),
            cu2mg.X.eval(ctx) - x
        )

    sol = root(fun, (float(t0), 0.05, 0.9, x0, -20000), options=dict(factor=1e-4))
    print(sol.message)
    print(sol.x)
    return Eq2Point(sol.x[3], sol.x[0])


eq2 = [
    calculate_eq2(fcc, liq, 1000, 0),
    solve_eq2_cu2mg(liq, 800 + T0, 1 / 3),
    calculate_eq2(cumg2, liq, 800, 2 / 3),
    calculate_eq2(hcp, liq, 1000, 1),
]

# Initial conditions for Eq3 calculations
fcc0, liq0, cu2mg0 = 0.1, 0.2, 0.33
t0 = T0 + 700
a0 = fcc.a0().eval({"T": t0, "X": fcc0})
b0 = fcc.b0().eval({"T": t0, "X": fcc0})
x0 = (a0, b0, t0, fcc0, liq0, cu2mg0, cu2mg0)

cu2mg1, liq1 = 0.33, 0.56
t1 = T0 + 550
a1 = liq.a0().eval({"T": t1, "X": liq1})
b1 = liq.b0().eval({"T": t1, "X": liq1})
x1 = (a1, b1, t1, cu2mg1, cu2mg1, liq1)

hcp2, liq2 = 0.99, 0.85
t2 = T0 + 400
a2 = hcp.G.derivative("X").eval({"T": t2, "X": hcp2})
b2 = hcp.G.eval({"T": t2, "X": hcp2}) - a2 * hcp.X.eval({"X": hcp2})
x2 = (a2, b2, t2, liq2, hcp2)

eq3 = [
    calculate_eq3(fcc, liq, cu2mg, x0),
    calculate_eq3(cu2mg, liq, cumg2, x1),
    calculate_eq3(cumg2, liq, hcp, x2),
]

eqlines = [
    calculate_eqline(liq, fcc, (eq3[0].T, eq2[0].T), eq=eq3[0]),
    calculate_eqline(cu2mg, fcc, (eq3[0].T, T0 + 200), eq=eq3[0]),
    calculate_eqline(liq, cu2mg, (eq3[0].T, eq2[1].T), eq=eq3[0]),

    calculate_eqline(liq, cu2mg, (eq3[1].T, eq2[1].T), eq=eq3[1]),
    calculate_eqline(cu2mg, cumg2, (eq3[1].T, 200 + T0), eq=eq3[1]),
    calculate_eqline(liq, cumg2, (eq3[1].T, eq2[2].T), eq=eq3[1]),

    calculate_eqline(liq, cumg2, (eq3[2].T, eq2[2].T), eq=eq3[2]),
    calculate_eqline(hcp, cumg2, (eq3[2].T, 200 + T0), eq=eq3[2]),
    calculate_eqline(liq, hcp, (eq3[2].T, eq2[3].T), eq=eq3[2])
]

for p in eq2:
    plt.plot(at2wt(p.x), p.T - T0, "C3o")

for p in eq3:
    plt.plot([at2wt(i[0]) for i in p.phases.values()], [p.T - T0, p.T - T0, p.T - T0], "C1o-")

for l in eqlines:
    plt.plot(at2wt(l.x1), l.t - T0, "C2-")
    plt.plot(at2wt(l.x2), l.t - T0, "C2-")

plt.savefig("CuMg_diagram-withCH.png")
plt.show()
