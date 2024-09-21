import numpy as np

import data as d
import old.data as od

y = np.linspace(0, 1, 500)
y1, y2 = np.random.uniform(0, 1, size=(2, 100000))
T0 = 273.15
Trange = np.arange(200 + T0, 1200 + T0, 10)

# Temperature only
assert np.isclose(d.GHSERCU.eval({"T": Trange}), od.GHSERCU(Trange)).all()
assert np.isclose(d.GHSERMG.eval({"T": Trange}), od.GHSERMG(Trange)).all()
assert np.isclose(d.GFCCMG.eval({"T": Trange}), od.GFCCMG(Trange)).all()
assert np.isclose(d.GLIQMG.eval({"T": Trange}), od.GLIQMG(Trange)).all()
assert np.isclose(d.GLIQCU.eval({"T": Trange}), od.GLIQCU(Trange)).all()

assert np.isclose(d.G_CuMg2.eval({"T": Trange}), od.G_CuMg2(Trange)).all()

# Simple concentration
for T in Trange:
    assert np.isclose(d.G_hcp.eval({"T": T, "X": y}), od.G_hcp(T, 1 - y, y)).all()
    assert np.isclose(d.G_fcc.eval({"T": T, "X": y}), od.G_fcc(T, 1 - y, y)).all()
    assert np.isclose(d.G_liq.eval({"T": T, "X": y}), od.G_liq(T, 1 - y, y)).all()

# Double lattice
for T in Trange:
    assert np.isclose(d.G_Cu2Mg.eval({"T": T, "Y1": y1, "Y2": y2}), od.G_Cu2Mg(T, 1 - y1, y1, 1 - y2, y2)).all()
