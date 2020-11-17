# coding = 'utf-8'

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import numpy as np
import copy


class SolverFunctional(ABC):
    @abstractmethod
    def call(self, beta):
        """
        Returns a float/double that represents that function value
        :params beta: The input. Must be a 1d array.
        :returns: function value
        """

    @abstractmethod
    def dir(self, beta):
        """
        Returns the derivatives.
        :params beta:
        :returns: partial derivatives. Must be equal to the length of the input!
        """


@dataclass
class ProximalSolverOptions:
    penalty_type: str = field(
        default='l1',
        metadata={'help':
                      ("The penalty to add to the function."
                       "Currently only support l1 penalty")
                  }
    )

    penalty: float = field(
        default=0,
        metadata={'help': ("The penalty for l1 loss. "
                           "The default value is 0")}
    )
    tol: float = field(
        default=1e-12,
        metadata={'help':
                      ("The tolerance to check the convergence of the algorithm"
                       "Must be larger than 0."
                       "Default value is 1e-12")}
    )
    max_iter: int = field(
        default=100,
        metadata={'help':
                      "The max iteration for the algorithms to run."
                      "Should be larger than 0"
                      "The default value is 100."}
    )

    n_dim: int = field(
        default=1,
        metadata={'help':
                      "The dimension of the solver."
                      "Must be equal to the input."
                      "Default is 1"}
    )

    step_size: float = field(
        default=0.1,
        metadata={'help': "The step size."
                          "Currently it is fixed. "
                          "Default is 0.1."}
    )

    def __post_init__(self):
        if self.tol <= 0:
            logging.warning("The tolerance for convergence must be set to be larger than 0. "
                            "Revert to default value 1e-12")
            self.tol = 1e-12
        if self.max_iter <= 0:
            logging.warning("The max iteration should be a number larger than 0."
                            "Revert to default value 100.")
            self.max_iter = 100
        if self.penalty != 'l1':
            logging.error("The only support form of penalty now is L_1 penalty")


class ProximalSolver(object):
    """
    This class offers functionality for solving general proximal methods. For now, only methods with $L_1$ penalty
    is supported.
    """

    def __init__(self, opt: ProximalSolverOptions = None):
        if opt is None:
            self.opt = ProximalSolverOptions()
        else:
            self.opt = opt
        self.parameter = None

    def solve(self, solver_functional: SolverFunctional, init_params=None):
        """
        Solve the problem using proximal methods.

        params: solver_functional: Must inherit SolverFunctional abstract base class. Specifically, one must offer call and dir functions.
        init_params: The initial parameter. If not given, will be given a small random normal distributed, the size is equal
        to the problem.

        returns: the solution.
        """
        if init_params is None:
            self.parameter = np.random.randn(self.opt.n_dim)
        else:
            self.parameter = init_params

        round = 0
        converged = False
        while not converged and round <= self.opt.max_iter:
            parameter_old = copy.deepcopy(self.parameter)
            v = parameter_old - self.opt.step_size * solver_functional.dir(parameter_old)
            parameter_new = self._soft_thres(v, self.opt.penalty)
            if np.linalg.norm(parameter_new - parameter_old, ord=2) <= self.opt.tol * self.opt.n_dim and np.abs(
                solver_functional.call(parameter_old) - solver_functional.call(parameter_new)) <= self.opt.tol:   :
            self.parameter = parameter_new
            converged = True

            if not converged and round == self.opt.max_iter:
                logging.warning("The algorithms have not converged. Return the value for last round.")

            self.parameter = parameter_new

        return self.parameter


    def _soft_thres(self, v_i: np.ndarray, lambda_i: float):


        def self_thres_impl(v_j):
            if v_j >= lambda_i:
                return v_j - lambda_i
            elif v_j <= -lambda_i:
                return v_j + lambda_i
            else:
                return 0

        result = np.apply_along_axis(self_thres_impl, 0, v_i)
        return result
