#!/usr/bin/env python3
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np

from old.data import G_hcp, G_fcc, G_liq, G_CuMg2, G_Cu2Mg

y = np.linspace(0, 1, 500)
y1, y2 = np.random.uniform(0, 1, size=(2, 100000))

phases = [
    ("fcc", y, lambda T: G_fcc(T, 1 - y, y)),
    ("hcp", y, lambda T: G_hcp(T, 1 - y, y)),
    ("liq", y, lambda T: G_liq(T, 1 - y, y)),
    ("CuMg2", np.array([2 / 3]), G_CuMg2),
    ("Cu2Mg", (2 * y1 + y2) / 3, lambda T: G_Cu2Mg(T, 1 - y1, y1, 1 - y2, y2)),
]

T0 = 273.15
Trange = np.arange(200, 1200, 10)


def at2wt(x):
    x = np.array(x)
    AMG, ACU = 24.305, 63.546
    return x * AMG / (x * AMG + (1 - x) * ACU)


from PIL import Image

plt.imshow(np.array(Image.open("../CuMg_diagram.png"))[86:681, 123:915], extent=(0, 1, 200, 1200), aspect="auto")

N = sum(x.size for name, x, G in phases)
points = np.empty((N, 2))
phase_id = np.empty(N, dtype=int)

for T in Trange:
    print(".", end="", flush=True)

    o = 0
    for i, (name, x, G) in enumerate(phases):
        points[o:o + x.size, 0] = x
        points[o:o + x.size, 1] = G(T0 + T)
        phase_id[o:o + x.size] = i
        o += x.size

    hull = ConvexHull(points)
    mask = hull.equations[:, 1] < 0
    for a, b in hull.simplices[mask]:
        if phase_id[a] != phase_id[b] or abs(points[a, 0] - points[b, 0]) > 0.05:
            plt.plot(at2wt([points[a, 0], points[b, 0]]), [T, T], "C0.-", lw=0.5)
print()
plt.xlim(0, 1)
plt.savefig("CuMg_diagram.pdf")
plt.show()
