from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

Value = float | np.ndarray  # Value datatype


class Term(ABC):
    """
    Abstract base class representing a Term.
    """

    @abstractmethod
    def eval(self, ctx: dict[str, Value]) -> Value:
        """
        Evaluates a term with a given context and returns the value.

        :param dict ctx: The evaluation context
        :return: The value of the Term
        :rtype: Value
        :raises KeyError: if a parameter name is not found in the context
        """
        pass

    @abstractmethod
    def derivative(self, param_name: str) -> Term:
        """
        Perform a symbolic derivation on the Term.

        :param str param_name: The parameter name that the Term should be derived with.
        :return: A Term representing a derivation
        :rtype: Term
        """
        pass

    def simplify(self) -> Term:
        """
        Applies simplification rules on the Term.

        :return: The simplified Term
        :rtype: Term
        """
        return self

    def get_param_names(self) -> list[str]:
        """
        Returs a list of all parameter names contined by this Term, sorted in alphabetical order.

        :return: List of parameter names
        """
        return sorted(list(self._get_param_names()))

    @abstractmethod
    def _get_param_names(self) -> set[str]:
        pass

    # @abstractmethod
    # def substitute(self, old, new) -> Term:
    #     pass
    #
    # @abstractmethod
    # def get_function(self) -> Callable:
    #     pass


@dataclass(frozen=True)
class Sum(Term):
    """
    A Term representing a sum of two sub-Terms.
    """

    a: Term
    b: Term

    def eval(self, ctx: dict[str, Value]) -> Value:
        return self.a.eval(ctx) + self.b.eval(ctx)

    def derivative(self, param_name: str) -> Term:
        return Sum(self.a.derivative(param_name), self.b.derivative(param_name))

    def simplify(self) -> Term:
        t = self.a.simplify()
        u = self.b.simplify()
        match (t, u):
            case (Zero(), v):
                return v
            case (v, Zero()):
                return v
            case _:
                return Sum(t, u)

    def _get_param_names(self) -> set[str]:
        return self.a._get_param_names() | self.b._get_param_names()


@dataclass(frozen=True)
class SumList(Term):
    """
    A Term representing a sum of many sub-Terms.
    """

    term_list: list[Term]

    def eval(self, ctx: dict[str, Value]) -> Value:
        return sum(f.eval(ctx) for f in self.term_list)

    def derivative(self, param_name: str) -> Term:
        d_list = [f.derivative(param_name) for f in self.term_list]
        return SumList(d_list)

    def simplify(self) -> Term:
        new_list = [f.simplify() for f in self.term_list]
        return SumList([f for f in new_list if not isinstance(f, Zero)])

    def _get_param_names(self) -> set[str]:
        return set.union(*(t._get_param_names() for t in self.term_list))


@dataclass(frozen=True)
class Mult(Term):
    """
    A Term representing a multiplication of two sub-Terms.
    """

    m: Term
    n: Term

    def eval(self, ctx: dict[str, Value]) -> Value:
        return self.m.eval(ctx) * self.n.eval(ctx)

    def derivative(self, param_name: str) -> Term:
        t1 = Mult(self.m.derivative(param_name), self.n)
        t2 = Mult(self.m, self.n.derivative(param_name))
        return Sum(t1, t2)

    def simplify(self) -> Term:
        t = self.m.simplify()
        u = self.n.simplify()

        match (t, u):
            case (Zero(), _):
                return Zero()
            case (_, Zero()):
                return Zero()
            case (Const(1), v):
                return v
            case (v, Const(1)):
                return v
            case _:
                return Mult(t, u)

    def _get_param_names(self) -> set[str]:
        return self.m._get_param_names() | self.n._get_param_names()


@dataclass(frozen=True)
class Div(Term):
    """
    A Term representing a division of two sub-Terms.
    """

    dividend: Term
    divisor: Term

    def eval(self, ctx: dict[str, Value]) -> Value:
        return self.dividend.eval(ctx) / self.divisor.eval(ctx)

    def derivative(self, param_name: str) -> Term:
        t1 = Mult(self.dividend.derivative(param_name), self.divisor)
        t2 = Mult(self.dividend, self.divisor.derivative(param_name))
        return Div(Sum(t1, neg(t2)), Poly(self.divisor, 2, 1))

    def simplify(self) -> Term:
        t = self.dividend.simplify()
        u = self.divisor.simplify()

        match (t, u):
            case (Zero(), _):
                return Zero()
            case (_, Zero()):
                raise ZeroDivisionError(f"{self.dividend}/{self.divisor} contains division by 0")
            case (v, Const(1)):
                return v
            case _:
                return Div(t, u)

    def _get_param_names(self) -> set[str]:
        return self.dividend._get_param_names() | self.divisor._get_param_names()


@dataclass(frozen=True)
class Zero(Term):
    """
    A Term representing a 0.
    """

    def eval(self, ctx: dict[str, Value]) -> Value:
        return 0.0

    def derivative(self, param_name: str) -> Term:
        return Zero()

    def _get_param_names(self) -> set[str]:
        return set()


@dataclass(frozen=True)
class Const(Term):
    """
    A Term representing a numerical constant.
    """

    value: float

    def eval(self, ctx: dict[str, Value]) -> Value:
        return self.value

    def derivative(self, param_name: str) -> Term:
        return Zero()

    def _get_param_names(self) -> set[str]:
        return set()


@dataclass(frozen=True)
class Param(Term):
    """
    A Term representing a named parameter.
    """

    name: str

    def eval(self, ctx: dict[str, Value]) -> Value:
        if self.name in ctx:
            return ctx[self.name]
        else:
            raise Exception(f"Parameter {self.name} not found in eval context")

    def derivative(self, param_name: str) -> Term:
        if param_name == self.name:
            return Const(1)
        else:
            return Zero()

    def _get_param_names(self) -> set[str]:
        return {self.name}


@dataclass(frozen=True)
class Poly(Term):
    """
    A Term representing a Polynomial part (coef)*[(term)**(degree)] 
    """

    term: Term
    degree: int
    coef: float

    def eval(self, ctx: dict[str, Value]) -> Value:
        return self.coef * (self.term.eval(ctx) ** self.degree)

    def derivative(self, param_name: str) -> Term:
        return Mult(
            Poly(self.term, self.degree - 1, self.coef * self.degree) if self.degree != 0 else Zero(),
            self.term.derivative(param_name)
        )

    def simplify(self) -> Term:
        if self.degree == 0:
            return Const(self.coef)
        else:
            return Poly(self.term.simplify(), self.degree, self.coef)

    def _get_param_names(self) -> set[str]:
        return self.term._get_param_names()


@dataclass(frozen=True)
class Log(Term):
    """
    A Term representing a natural logarithm.
    """

    term: Term

    def eval(self, ctx: dict[str, Value]) -> Value:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(self.term.eval(ctx), np.log(self.term.eval(ctx)), 0)

    def derivative(self, param_name: str) -> Term:
        return Mult(Poly(self.term, -1, 1), self.term.derivative(param_name))

    def simplify(self) -> Term:
        return Log(self.term.simplify())

    def _get_param_names(self) -> set[str]:
        return self.term._get_param_names()


# UglyHack: This yearns for a proper Condition Term
@dataclass(frozen=True)
class UglyBranch(Term):
    """
    A Term representing a branch based on a Param value.
    """

    a: Term
    b: Term
    param: Param
    value: float

    def eval(self, ctx: dict[str, Value]) -> Value:
        return np.where(
            self.param.eval(ctx) < self.value,
            self.a.eval(ctx),
            self.b.eval(ctx)
        )

    def derivative(self, param_name: str) -> Term:
        return UglyBranch(self.a.derivative(param_name), self.b.derivative(param_name), self.param, self.value)

    def simplify(self) -> Term:
        return UglyBranch(self.a.simplify(), self.b.simplify(), self.param, self.value)

    def _get_param_names(self) -> set[str]:
        return self.a._get_param_names() | self.b._get_param_names()


def neg(t: Term) -> Term:
    """
    Returns negation of a given term. ((-1) * term)
    
    :param t: The Term to be negated
    :return: term multiplied by (-1)
    """
    return Mult(Const(-1), t)
