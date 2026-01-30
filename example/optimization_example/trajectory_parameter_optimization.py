"""Trajectory parameter optimization example using polynomial coefficients.

This script shows how to treat the parameters of a motion trajectory as
decision variables. A cubic polynomial trajectory ``p(t)`` is fit to sampled
measurements ``y_target(t_i)`` by minimizing the squared error between the
polynomial and the desired trajectory. The coefficients of the polynomial are
exposed as a ``Variable`` so that the inward/outward utilities can assemble the
least-squares system and perform Gauss-Newton iterations.
"""
import numpy as np

from robokots.inward.problem import Problem
from robokots.outward.term import L2Cost, Variable, VariablePack, VectorSquaredSumResidual


def _polynomial_basis(times: np.ndarray, order: int) -> np.ndarray:
    """Return a Vandermonde-style basis matrix up to ``order`` terms.

    Args:
        times: Sampling times, shape ``(N,)``.
        order: Number of coefficients (polynomial degree is ``order-1``).

    Returns:
        Basis matrix with shape ``(N, order)`` where column ``k`` corresponds to
        ``t**k``.
    """

    times = np.asarray(times, dtype=float).reshape(-1)
    powers = np.arange(order, dtype=float)
    return np.power(times[:, None], powers[None, :])


class PolynomialTrajectoryQuantity:
    """Quantity that measures trajectory tracking error for a polynomial."""

    def __init__(self, times: np.ndarray, target: np.ndarray, coeffs: Variable) -> None:
        self.name = "polynomial_trajectory"
        self.vars = [coeffs]

        self._times = np.asarray(times, dtype=float).reshape(-1)
        self._target = np.asarray(target, dtype=float).reshape(-1)
        if self._times.shape != self._target.shape:
            raise ValueError("times and target must have the same shape")

        self._basis = _polynomial_basis(self._times, coeffs.dim())
        self.out_dim = int(self._times.size)
        self._coeffs = coeffs

    def value(self) -> np.ndarray:
        coeff_vec = np.asarray(self._coeffs.x, dtype=float).reshape(-1)
        predicted = self._basis @ coeff_vec
        return predicted - self._target

    def jacobian_blocks(self):
        # Only one variable; the Jacobian is the basis matrix itself.
        return [self._basis]


def solve_gauss_newton(problem: Problem, variables: VariablePack, max_iters: int = 15) -> None:
    """Run a basic Gauss-Newton loop to minimize the stacked residuals."""

    for k in range(max_iters):
        r_all, J_all = problem.linearize()
        cost = float(r_all @ r_all)
        print(f"[iter {k:02d}] coeffs={variables.get()}  cost={cost:.6f}")

        grad = J_all.T @ r_all
        if np.linalg.norm(grad) < 1e-10:
            break

        lhs = J_all.T @ J_all
        rhs = -grad
        dx, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        if np.linalg.norm(dx) < 1e-12:
            break

        variables.apply_dx(dx)


def main() -> None:
    # Create synthetic measurements from a known smooth trajectory.
    times = np.linspace(0.0, 2.0, num=25)
    y_target = np.sin(1.5 * times) + 0.2 * times  # shape (25,)

    # Cubic coefficients are the optimization variables.
    coeffs = Variable(name="poly_coeffs", x=np.zeros(4, dtype=float))
    variables = VariablePack([coeffs])

    residual = VectorSquaredSumResidual(
        "trajectory_tracking", PolynomialTrajectoryQuantity(times, y_target, coeffs)
    )
    problem = Problem(variables=variables, terms=[(residual, L2Cost())])

    print("Initial coefficients:", variables.get())
    print("Initial cost:", problem.cost_value())

    solve_gauss_newton(problem, variables, max_iters=15)

    print("\nOptimized coefficients:", variables.get())
    y_fit = _polynomial_basis(times, coeffs.dim()) @ variables.get()
    print("Final RMSE:", np.sqrt(np.mean((y_fit - y_target) ** 2)))


if __name__ == "__main__":
    main()
