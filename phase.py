from functools import reduce

from scipy.integrate import solve_ivp
from scipy.optimize import root

from terms import *


@dataclass(frozen=True)
class Phase:
    """
    A Class representing a Phase.

    X - Term representing composition
    G - Term representing the Gibbs function
    """

    name: str
    X: Term
    G: Term

    def eq3_terms(self, a: Term, b: Term) -> list[Term]:
        """
        Returns the Terms needed to calculate the Eq3 point

        :param Term a: The slope of the tangent line (usually Param("A"))
        :param Term b: The intercept of the tangent line (usually Param("B"))
        :return: list of Terms for Eq3 calculation
        """
        x_params = self.X.get_param_names()
        return ([SumList([Mult(a, self.X), b, neg(self.G)])]
                + [Sum(Mult(a, self.X.derivative(p)), neg(self.G.derivative(p))) for p in x_params])

    def eqline_terms(self, da: Term):
        pars = self.X.get_param_names()

        if len(pars) == 1:
            dtx = self.G.derivative("T").derivative(pars[0])
            dxx = self.G.derivative(pars[0]).derivative(pars[0])
            return Div(Sum(da, neg(dtx)), dxx)
        if len(pars) == 2:
            dxx = self.G.derivative(pars[0]).derivative(pars[0])
            dxy = self.G.derivative(pars[0]).derivative(pars[1])
            dyy = self.G.derivative(pars[1]).derivative(pars[1])

            dtx = self.G.derivative("T").derivative(pars[0])
            dty = self.G.derivative("T").derivative(pars[1])

            t1 = Sum(Mult(da, self.X.derivative(pars[0])), neg(dtx))
            t2 = Sum(Mult(da, self.X.derivative(pars[1])), neg(dty))
            return [(dxx, dxy), (dxy, dyy)], (t1, t2)

    def a0(self) -> Term:
        """
        Returns a Term representing the initial slope for Eq3 calculation

        :return: The initial slope for Eq3 calculation
        """
        pars = self.X.get_param_names()
        if pars:
            return self.G.derivative(pars[0])

    def b0(self) -> Term:
        """
        Returns a Term representing the initial intercept for Eq3 calculation
        :return: The initial intercept for Eq3 calculation
        """
        a0 = self.a0()
        if a0:
            return Sum(self.G, neg(Mult(a0, self.X)))


def calculate_eq2(p1: Phase, p2: Phase, t0: float, x: float) -> float:
    """
    Calculates an equilibrium point between two phases.

    :param p1: Phase 1
    :param p2: Phase 2
    :param t0: starting temperature for the solver
    :param x: composition fraction
    :return: Temperature
    """

    def fun(t):
        ctx = {"T": t, "X": x}
        return p1.G.eval(ctx) - p2.G.eval(ctx)

    print(f"Now Solving EQ2: {p1.name}, {p2.name}.")

    sol = root(fun, float(t0))
    print(sol.message)
    print(sol.x)
    return sol.x[0]


@dataclass(frozen=True)
class Eq3Point:
    phases: dict[str, (float, tuple)]
    T: float


def calculate_eq3(p1: Phase, p2: Phase, p3: Phase, x0: tuple) -> Eq3Point:
    """
    Calculates the equilibrium line between three phases.

    :param p1: Phase 1
    :param p2: Phase 2
    :param p3: Phase 3
    :param x0: initial parameters for the solver
    :return: Eq3Point containing the temperature and concentration of the eutectic points
    """

    A = Param("A")
    B = Param("B")

    phases = [p1, p2, p3]

    pars = [p.X.get_param_names() for p in phases]
    lens = [len(par) for par in pars]
    indexes = reduce(lambda x, y: x + [x[-1] + y], lens, [0])

    params = 3 + sum(lens)

    if params != len(x0):
        raise Exception(f"initial value must have length {params}")

    def split(y):
        return [y[start:stop] for start, stop in zip(indexes[:-1], indexes[1:])]

    def fun(x):
        a, b, t, *y = x
        ctxs = [{"A": a, "B": b, "T": t} | {k: v for k, v in zip(par, z)} for par, z in zip(pars, split(y))]
        return [f.eval(ctx) for p, ctx in zip(phases, ctxs) for f in p.eq3_terms(A, B)]

    print(f"Now Solving EQ3: {p1.name}, {p2.name}, {p3.name}")

    sol = root(fun, x0, options=dict(factor=1e-4))
    print(sol.message)
    print(sol.x)

    _, _, sol_t, *sol_y = sol.x
    ctxs = [{k: v for k, v in zip(par, z)} for par, z in zip(pars, split(sol_y))]
    return Eq3Point({p.name: (p.X.eval(ctx), z) for p, ctx, z in zip(phases, ctxs, split(sol_y))}, sol_t)


@dataclass(frozen=True)
class EqLine:
    x1: np.ndarray
    x2: np.ndarray
    t: np.ndarray


def calculate_eqline(p1: Phase,
                     p2: Phase,
                     t_range: tuple[float, float],
                     c: tuple | None = None,
                     eq: Eq3Point | None = None
                     ):
    """
    Calculates boundaries between phases within the supplied temperature range with initial concentrations.

    :param p1: Phase 1
    :param p2: Phase 2
    :param t_range: The temperature range of integration
    :param c: Initial concentrations (optional)
    :param eq: Initial Eq3 point (optional - used instead of c)
    :return: EqLine object representing the phase boundaries
    """

    if not c:
        c = (*eq.phases[p1.name][1], *eq.phases[p2.name][1])

    phases = [p1, p2]
    pars = [p.X.get_param_names() for p in phases]
    lens = [len(par) for par in pars]
    indexes = reduce(lambda x, y: x + [x[-1] + y], lens, [0])

    def split(y):
        return [y[start:stop] for start, stop in zip(indexes[:-1], indexes[1:])]

    def eval_eqline_terms(tr, ctx):
        if isinstance(tr, Term):
            return (tr.eval(ctx),)
        if not tr:
            return tuple()
        else:
            mat, t = tr
            return np.linalg.solve(list(tuple(c.eval(ctx) for c in r) for r in mat), tuple(x.eval(ctx) for x in t))

    def fun(t, x):
        c1, c2 = [{"T": t} | {k: v for k, v in zip(par, z)} for par, z in zip(pars, split(x))]
        da = (p1.G.derivative("T").eval(c1) - p2.G.derivative("T").eval(c2)) / (p1.X.eval(c1) - p2.X.eval(c2))
        A = Param("A")
        # UglyHack: this could be solved much more elegantly with a Param rewriting rule, allowing for proper Term mixing, requiring less .eval() calls

        ret = (
            *eval_eqline_terms(p1.eqline_terms(A), c1 | {"A": da}),
            *eval_eqline_terms(p2.eqline_terms(A), c2 | {"A": da}),
        )
        return ret

    print(f"Now Solving EQLine: {p1.name}, {p2.name}.")
    sol = solve_ivp(fun, t_range, c, max_step=5)
    print(sol.message)
    print(sol.y)
    print(sol.t)

    ctx1, ctx2 = [{k: v for k, v in zip(par, z)} for par, z in zip(pars, split(sol.y))]
    x1, x2 = np.atleast_1d(p1.X.eval(ctx1)), np.atleast_1d(p2.X.eval(ctx2))
    l = len(sol.t)
    return EqLine(x1 if len(x1) != 1 else np.repeat(x1, l), x2 if len(x2) != 1 else np.repeat(x2, l), sol.t)
