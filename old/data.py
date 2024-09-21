import numpy as np

R = 8.314
LN = np.log


def xlnx(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x, x * np.log(x), 0)


# FUN GHSERCU 298.15 -7770.458+130.485235*T-24.112392*T*LN(T)
#     -.00265684*T**2+1.29223E-07*T**3+52478*T**(-1); 1357.77 Y
#      -13542.026+183.803828*T-31.38*T*LN(T)
#     +3.64167E+29*T**(-9); 3200 N !
# FUN GHSERMG 298.15 -8367.34+143.675547*T-26.1849782*T*LN(T)
#     +4.858E-04*T**2-1.393669E-06*T**3+78950*T**(-1); 923 Y
#      -14130.185+204.716215*T-34.3088*T*LN(T)
#     +1.038192E+28*T**(-9); 3000 N !
# FUN GFCCMG 298.15 +2600-.9*T+GHSERMG#; 6000 N !
# FUN GLIQMG 298.15 +8202.243-8.83693*T-8.01759E-20*T**7
#     +GHSERMG#; 923 Y
#      -5439.869+195.324057*T-34.3088*T*LN(T); 3000 N !
# FUN GLIQCU 298.15 +5194.277+120.973331*T-24.112392*T*LN(T)
#     -.00265684*T**2+1.29223E-07*T**3+52478*T**(-1)
#     -5.8489E-21*T**7;  1357.76 Y
#      -46.545+173.881484*T-31.38*T*LN(T); 3200 N !

def GHSERCU(T):
    return np.where(
        T < 1357.77,
        -7770.458 + 130.485235 * T - 24.112392 * T * LN(T)
        - .00265684 * T ** 2 + 1.29223E-07 * T ** 3 + 52478 * T ** (-1),
        -13542.026 + 183.803828 * T - 31.38 * T * LN(T) + 3.64167E+29 * T ** (-9)
    )


def GHSERMG(T):
    return np.where(
        T < 923,
        -8367.34 + 143.675547 * T - 26.1849782 * T * LN(T) + 4.858E-04 * T ** 2
        - 1.393669E-06 * T ** 3 + 78950 * T ** (-1),
        -14130.185 + 204.716215 * T - 34.3088 * T * LN(T) + 1.038192E+28 * T ** (-9)
    )


def GFCCMG(T):
    return +2600 - .9 * T + GHSERMG(T)


def GLIQMG(T):
    return np.where(
        T < 923,
        +8202.243 - 8.83693 * T - 8.01759E-20 * T ** 7 + GHSERMG(T),
        -5439.869 + 195.324057 * T - 34.3088 * T * LN(T)
    )


def GLIQCU(T):
    return np.where(
        T < 1357.76,
        +5194.277 + 120.973331 * T - 24.112392 * T * LN(T)
        - .00265684 * T ** 2 + 1.29223E-07 * T ** 3 + 52478 * T ** (-1) - 5.8489E-21 * T ** 7,
        -46.545 + 173.881484 * T - 31.38 * T * LN(T)
    )


# PHASE HCP_A3  %F  2 1   .5 !
# CONST HCP_A3 : CU, MG : VA : !
#
# PARAM G(HCP_A3,CU:VA;0) 298.15 +600+.2*T+GHSERCU#; 3200 N 91Din !
# PARAM G(HCP_A3,MG:VA;0) 298.15 +GHSERMG#; 3000 N 01Din !
# PARAM G(HCP_A3,CU,MG:VA;0) 298.15 +22500-3*T; 6000 N 98Cou2 !

def G_hcp(T, yCu, yMg):
    return (
        # reference surface
            + yCu * (+600 + .2 * T + GHSERCU(T))
            + yMg * GHSERMG(T)
            # excess energy (interaction)
            + yCu * yMg * (+22500 - 3 * T)
            # entropy
            + R * T * (xlnx(yCu) + xlnx(yMg))
    )


# PHASE FCC_A1  %F  2 1   1 !
# CONST FCC_A1   : CU, MG : VA : !

# PARAM G(FCC_A1,CU:VA;0) 298.15 +GHSERCU#; 3200 N 91Din !
# PARAM G(FCC_A1,MG:VA;0) 298.15 +GFCCMG#; 3000 N 91Din !
# PARAM G(FCC_A1,CU,MG:VA;0) 298.15 -22059.61+5.63232*T; 6000 N 98Cou2 !

def G_fcc(T, yCu, yMg):
    return (
        # reference surface
            + yCu * GHSERCU(T)
            + yMg * (+2600 - .9 * T + GHSERMG(T))
            # excess energy (interaction)
            + yCu * yMg * (-22059.61 + 5.63232 * T)
            # entropy
            + R * T * (xlnx(yCu) + xlnx(yMg))
    )


# PHASE LIQUID:L %  1  1.0
#  > Metallic Liquid Solution: Redlich-Kister_Muggianu Model. !
# CONST LIQUID:L : CU, MG : !

# PARAM G(LIQUID,CU;0) 298.15 +GLIQCU#; 3200 N 91Din !
# PARAM G(LIQUID,MG;0) 298.15 +GLIQMG#; 3000 N 91Din !
# PARAM G(LIQUID,CU,MG;0) 298.15 -36962.71+4.74394*T; 6000 N 98Cou2 !
# PARAM G(LIQUID,CU,MG;1) 298.15 -8182.19; 6000 N 98Cou2 !

def G_liq(T, yCu, yMg):
    return (
        # reference surface
            + yCu * GLIQCU(T)
            + yMg * GLIQMG(T)
            # excess energy (interaction)
            + yCu * yMg * (-36962.71 + 4.74394 * T)
            + yCu * yMg * (yCu - yMg) * (-8182.19)
            # entropy
            + R * T * (xlnx(yCu) + xlnx(yMg))
    )


# PHASE CB_CUMG2  %  2 1   2 !
# CONST CB_CUMG2  : CU : MG : !

# PARAM G(CB_CUMG2,CU:MG;0) 298.15 -28620+1.86456*T+GHSERCU#+2*GHSERMG#;
#    6000 N 98Cou2 !

def G_CuMg2(T):
    return (-28620 + 1.86456 * T + GHSERCU(T) + 2 * GHSERMG(T)) / 3

# PHASE C15_LAVES  %  2 2   1 !
# CONST C15_LAVES : CU%,MG : CU, MG% : !

# PARAM G(C15_LAVES,CU:CU;0) 298.15 +15000+3*GHSERCU#; 6000 N 98Cou2 !
# PARAM G(C15_LAVES,MG:CU;0) 298.15 +104970.96-16.46448*T
#    +2*GHSERMG#+GHSERCU#; 6000 N 98Cou2 !
# PARAM G(C15_LAVES,CU:MG;0) 298.15 -54690.99+364.73085*T
#    -69.276417*T*LN(T)-5.19246E-04*T**2+143502*T**(-1)
#    -5.65953E-06*T**3;  1100 Y
#   -84928.79+679.00124*T-111.269753*T*LN(T); 6000 N 98Cou2 !
# PARAM G(C15_LAVES,MG:MG;0) 298.15 +15000+3*GHSERMG#; 6000 N 98Lia2 !

# PARAM G(C15_LAVES,CU,MG:CU;0) 298.15 13011.35; 6000 N 98Cou2 !
# PARAM G(C15_LAVES,CU,MG:MG;0) 298.15 13011.35; 6000 N 98Cou2 !
# PARAM G(C15_LAVES,CU:CU,MG;0) 298.15 6599.45; 6000 N 98Cou2 !
# PARAM G(C15_LAVES,MG:CU,MG;0) 298.15 6599.45; 6000 N 98Cou2 !

def GCU2MG(T):
    return np.where(
        T < 1100,
        -54690.99
        + 364.73085 * T
        - 69.276417 * T * LN(T)
        - 5.19246E-04 * T ** 2
        + 143502 * T ** (-1)
        - 5.65953E-06 * T ** 3,
        -84928.79
        + 679.00124 * T
        - 111.269753 * T * LN(T)
    )


def G_Cu2Mg(T, yCu1, yMg1, yCu2, yMg2):
    return (
        # referece
            + yCu1 * yCu2 * (+15000 + 3 * GHSERCU(T))
            + yCu1 * yMg2 * GCU2MG(T)
            + yMg1 * yCu2 * (+104970.96 - 16.46448 * T + 2 * GHSERMG(T) + GHSERCU(T))
            + yMg1 * yMg2 * (+15000 + 3 * GHSERMG(T))

            + yCu1 * yMg1 * yCu2 * 13011.35
            + yCu1 * yMg1 * yMg2 * 13011.35
            + yCu1 * yCu2 * yMg2 * 6599.45
            + yMg1 * yCu2 * yMg2 * 6599.45

            + R * T * (2 * xlnx(yCu1) + 2 * xlnx(yMg1) + xlnx(yCu2) + xlnx(yMg2))
    ) / 3
