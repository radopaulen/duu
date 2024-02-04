from duu.activities.solvers.ds_ns_solver import \
    DesignSpaceSolverUsingNS
from duu.activities.solvers.pe_ns_solver import \
    ParameterEstimationSolverUsingNS


class SolverFactory:
    @classmethod
    def create(cls, name, problem, settings, algorithms):
        assert name is not None, \
            "The solver must be defined."
        assert problem is not None, \
            "The problem must be defined."
        assert settings is not None, \
            "The settings must be defined."
        assert algorithms is not None, \
            "The algorithms must be defined."

        if name == "pe-ns":
            return ParameterEstimationSolverUsingNS(problem, settings, algorithms)
        elif name == "pe-mcmc":
            assert False, "Branch not done yet. :("  # TODO

        elif name == "ds-ns":
            return DesignSpaceSolverUsingNS(problem, settings, algorithms)
        elif name == "ds-mcmc":
            assert False, "Branch not done yet. :("  # TODO
        else:
            assert False, "Solver not found."