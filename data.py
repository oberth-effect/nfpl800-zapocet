from terms import *

R = Const(8.314)

T = Param("T")


def xlnx(t: Term) -> Term:
    """
    Returns (term) * ln(term).

    :param t: The Term to be used
    :return: Term representing (term) * ln(term)
    """

    return Mult(t, Log(t))


def cxlnx(coef: float, t: Term) -> Term:
    """
    Returns (coef) * (term) * ln(term).

    :param coef: A numeric coefficient
    :param t: The Term to be used
    :return: Term representing (coef) * (term) * ln(term)
    """

    return Mult(Const(coef), xlnx(t))


def one_minus(t: Term) -> Term:
    """
    Returns (1 - term)

    :param t: The Term to be used
    :return: Term representing (1 - term)
    """

    return Sum(Const(1), neg(t))


X = Param("X")
yCu = one_minus(X)
yMg = X

Y1 = Param("Y1")
yCu1 = one_minus(Y1)
yMg1 = Y1

Y2 = Param("Y2")
yCu2 = one_minus(Y2)
yMg2 = Y2

X_CuMg2 = Const(2 / 3)
X_Cu2Mg = Mult(
    Sum(Mult(Const(2), Y1), Y2),
    Const(1 / 3)
)

ENTROPY = Mult(R, Mult(T, Sum(xlnx(yCu), xlnx(yMg))))

GHSERCU = UglyBranch(
    SumList([
        Const(-7770.458),
        Poly(T, 1, 130.485235),
        cxlnx(-24.112392, T),
        Poly(T, 2, -.00265684),
        Poly(T, 3, 1.29223E-07),
        Poly(T, -1, 52478)
    ]),
    SumList([
        Const(-13542.026),
        Poly(T, 1, 183.803828),
        cxlnx(-31.38, T),
        Poly(T, -9, 3.64167E+29)
    ]),
    T,
    1357.77
)

GHSERMG = UglyBranch(
    SumList([
        Const(-8367.34),
        Poly(T, 1, 143.675547),
        cxlnx(-26.1849782, T),
        Poly(T, 2, 4.858E-04),
        Poly(T, 3, -1.393669E-06),
        Poly(T, -1, 78950)
    ]),
    SumList([
        Const(-14130.185),
        Poly(T, 1, 204.716215),
        cxlnx(-34.3088, T),
        Poly(T, -9, 1.038192E+28)
    ]),
    T,
    923
)

GFCCMG = SumList([
    Const(2600),
    Poly(T, 1, -0.9),
    GHSERMG
])

GLIQMG = UglyBranch(
    SumList([
        Const(8202.243),
        Poly(T, 1, -8.83693),
        Poly(T, 7, -8.01759E-20),
        GHSERMG
    ]),
    SumList([
        Const(-5439.869),
        Poly(T, 1, 195.324057),
        cxlnx(-34.3088, T)
    ]),
    T,
    923
)

GLIQCU = UglyBranch(
    SumList([
        Const(5194.277),
        Poly(T, 1, 120.973331),
        cxlnx(-24.112392, T),
        Poly(T, 2, -0.00265684),
        Poly(T, 3, 1.29223E-07),
        Poly(T, -1, 52478),
        Poly(T, 7, -5.8489E-21)
    ]),
    SumList([
        Const(-46.545),
        Poly(T, 1, 173.881484),
        cxlnx(-31.38, T)
    ]),
    T,
    1357.76
)

G_hcp = SumList([
    Mult(
        yCu,
        SumList([
            Const(600),
            Poly(T, 1, 0.2),
            GHSERCU
        ])
    ),
    Mult(yMg, GHSERMG),
    Mult(yCu, Mult(yMg, Sum(Const(22500), Poly(T, 1, -3)))),
    ENTROPY
])

G_fcc = SumList([
    Mult(yCu, GHSERCU),
    Mult(yMg, SumList([
        Const(2600),
        Poly(T, 1, -0.9),
        GHSERMG
    ])),
    Mult(yCu, Mult(yMg, Sum(Const(-22059.61), Poly(T, 1, 5.63232)))),
    ENTROPY
])

G_liq = SumList([
    Mult(yCu, GLIQCU),
    Mult(yMg, GLIQMG),
    Mult(yCu, Mult(yMg, Sum(Const(-36962.71), Poly(T, 1, 4.74394)))),
    Mult(yCu, Mult(yMg, Mult(Sum(yCu, neg(yMg)), Const(-8182.19)))),
    ENTROPY
])

G_CuMg2 = Mult(
    SumList([
        Const(-28620),
        Poly(T, 1, 1.86456),
        GHSERCU,
        Mult(Const(2), GHSERMG)
    ]),
    Const(1 / 3)
)

GCU2MG = UglyBranch(
    SumList([
        Const(-54690.99),
        Poly(T, 1, 364.73085),
        cxlnx(-69.276417, T),
        Poly(T, 2, -5.19246E-04),
        Poly(T, -1, 143502),
        Poly(T, 3, -5.65953E-06)
    ]),
    SumList([
        Const(-84928.79),
        Poly(T, 1, 679.00124),
        cxlnx(-111.269753, T)
    ]),
    T,
    1100
)

G_Cu2Mg = Mult(
    SumList([
        Mult(yCu1, Mult(yCu2, Sum(
            Const(15000),
            Mult(Const(3), GHSERCU)
        ))),
        Mult(yCu1, Mult(yMg2, GCU2MG)),
        Mult(yMg1, Mult(yCu2, SumList([
            Const(104970.96),
            Poly(T, 1, -16.46448),
            Mult(Const(2), GHSERMG),
            GHSERCU
        ]))),
        Mult(yMg1, Mult(yMg2, Sum(
            Const(15000),
            Mult(Const(3), GHSERMG)
        ))),
        Mult(yCu1, Mult(yMg1, Mult(yCu2, Const(13011.35)))),
        Mult(yCu1, Mult(yMg1, Mult(yMg2, Const(13011.35)))),
        Mult(yCu1, Mult(yCu2, Mult(yMg2, Const(6599.45)))),
        Mult(yMg1, Mult(yCu2, Mult(yMg2, Const(6599.45)))),
        Mult(R, Mult(T, SumList([
            Mult(Const(2), xlnx(yCu1)),
            Mult(Const(2), xlnx(yMg1)),
            xlnx(yCu2),
            xlnx(yMg2)
        ])))
    ]),
    Const(1 / 3)
)
