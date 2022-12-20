from typing import Callable
import numpy as np


class CubicSpline:
    def __init__(
        self,
        target: Callable[[float], float],
        n: int,
        mu: float,
        lmbd: float,
        moment_0: int,
        moment_n: int,
    ) -> None:

        self.n = n
        self.h = 1 / n
        self.h_squared = self.h**2
        self.mu = mu
        self.lmbd = lmbd
        self.taget = target
        self.y = [target(self.h * j) for j in range(0, n + 1)]
        self.moment_0 = moment_0
        self.moment_n = moment_n
        self.moments = []

    def rhs(self, j: int) -> float:
        a = self.y[j + 1] - self.y[j]
        b = self.y[j] - self.y[j - 1]

        return 6 * ((a - b) / self.h) / (2 * self.h)

    def build_matrix(self):
        s_matrix = np.array([2, self.lmbd] + [0] * (self.n - 2))
        mat = [s_matrix]

        for i in range(1, self.n - 1):
            mat.append(
                np.array(
                    [0] * (i - 1) + [self.mu, 2, self.lmbd] + [0] * (self.n - i - 2)
                )
            )

        mat.append(np.array([0] * (self.n - 1) + [1]))

        return np.vstack(mat)

    def build_rhs(self):
        d = []
        for j in range(1, self.n):
            d.append(self.rhs(j))

        d.append(self.moment_n)

        return np.array(d).transpose()

    def compute_moments(self):
        a = self.build_matrix()
        d = self.build_rhs()
        print("y:=", cubicspline.y)
        print("d:=", d)
        print("a:=", a)
        self.moments = np.linalg.solve(a, d)
        print("m:=", self.moments)

    def spline_f(self, j: int):
        assert (
            len(self.moments) > 0
        ), "Please call compute_moments before accesing spline_f"

        moment_j_1 = self.moments[j - 1]
        moment_j = self.moments[j]

        x_j = self.h * j
        x_j_1 = self.h * (j - 1)

        def s(x: float):
            a = (moment_j_1 / (6 * self.h)) * (x_j - x) ** 3
            b = (moment_j / (6 * self.h)) * (x - x_j_1) ** 3
            c = (self.y[j - 1] - (moment_j_1 * self.h**2) / 6) * (x_j - x) / self.h
            d = (self.y[j] - (moment_j * self.h**2) / 6) * (x - x_j_1) / self.h

            return a + b + c + d

        return s

    def check_error_for_spline(
        self, spline: Callable[[float], float], value: float
    ) -> float:
        print("t()", self.taget(value))
        return abs(self.taget(value) - spline(value))


if __name__ == "__main__":
    cubicspline = CubicSpline(
        lambda x: x**4 + 1, n=20, mu=0.5, lmbd=0.5, moment_0=0, moment_n=12
    )
    cubicspline.compute_moments()
    s_2 = cubicspline.spline_f(6)
    print("S_2(3/10) :=", s_2(3 / 20))
    err = cubicspline.check_error_for_spline(s_2, 3 / 20)
    print("err:=", err)
