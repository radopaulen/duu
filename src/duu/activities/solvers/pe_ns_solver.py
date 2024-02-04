import copy
from math import isnan, nan
import numpy as np
import time

from duu import utils
from duu.activities.interfaces import Subject, Observer
from duu.activities.solvers.algorithms.points import BayesPoint
from duu.activities.solvers.algorithms.composite.factory import \
    CompositeAlgorithmFactory
from duu.activities.solvers.algorithms.composite import \
    NestedSamplingWithGlobalSearch


class ParameterEstimationSolverUsingNS(Subject):
    def __init__(self, problem=None, settings=None, algorithms=None):
        self.problem = None
        if problem is not None:
            self.set_problem(problem)

        self.settings = None
        if settings is not None:
            self.set_settings(settings)

        self.algorithms = None
        if algorithms is not None:
            self.set_algorithms(algorithms)

        self.observers = []
        self.solver_iteration = 0
        self.iteration_cpu_secs = 0.0

        self.output_buffer = []

        # state
        self.status = None
        self.phase = "INITIAL"

        self.nlive = None

        self.nest_idx = 0
        self.log_x = 0.0  # x_hat(0) = 1.0
        self.log_t = None
        self.log_w0 = None

        self.log_zl = -np.inf
        self.log_zd = -np.inf
        self.log_z = -np.inf

        self.hd = 0.0

    def set_problem(self, problem):
        # The checking is already done by the PE Manager.
        self.problem = problem

    def set_settings(self, settings):
        assert isinstance(settings, dict), \
            "The settings must be a dictionary."

        mandatory_keys = ['parallel', 'stop_criteria']
        assert all(mkey in settings.keys() for mkey in mandatory_keys), \
            "The 'settings' keys must be the following:\n" \
            "['parallel', 'stop_criteria']." \
            "Look for typos, white spaces or missing keys."

        specified_stop_criteria = utils.keys_in(settings["stop_criteria"])
        assert 'contribution_to_evidence' in specified_stop_criteria, \
            "'contribution_to_evidence' must be the stop criterion."

        self.settings = settings

    def set_algorithms(self, algos):
        assert self.problem is not None, \
            "Define 'problem' before the algorithms of 'pe_solver'."
        assert self.settings is not None, \
            "Define 'settings' before the algorithms of 'pe_solver'."
        assert isinstance(algos, dict), \
            "algorithms must be a dictionary."

        mandatory_keys = ['sampling']
        assert all(mkey in algos.keys() for mkey in mandatory_keys), \
            "The 'algorithms' of 'pe_solver' should be for the steps:\n" \
            "['sampling']. " \
            "Look for typos, white spaces or missing keys."

        algos_permitted = [NestedSamplingWithGlobalSearch.get_type() + "-" +
                           NestedSamplingWithGlobalSearch.get_ui_name()]

        sampling_algo_name = algos["sampling"]["algorithm"]
        assert sampling_algo_name in algos_permitted, \
            "The algorithm specified for 'sampling' is not permitted."

        sampling_algo_settings = algos["sampling"]["settings"]
        algos_for_sampling_algo = algos["sampling"]["algorithms"]
        algo = CompositeAlgorithmFactory.create(sampling_algo_name,
                                                sampling_algo_settings,
                                                algos_for_sampling_algo)
        self.algorithms = {
            "sampling": {"algorithm": algo}
        }

    def solve(self):
        if self.phase == "INITIAL":
            sampling_algo = self.algorithms["sampling"]["algorithm"]
            lbs, ubs = self.get_parameters_bounds(which="both", as_array=True)
            sampling_algo.set_bounds(lbs, ubs)
            sampling_algo.set_sorting_func(self.phi)
            sampling_algo.initialize_live_points()

            lpoints = sampling_algo.get_live_points()
            self.nlive = len(lpoints)
            self.log_t = -1.0 / self.nlive
            self.log_w0 = np.log(1.0 - np.exp(self.log_t))

            print("Phase INITIAL is over.")
            self.phase = "NESTING"

        while not self.status in ["SAMPLING_FAILED",
                                  "SUBALGORITHM_STOPPED",
                                  "FINISHED"]:
            self.do_one_sampling_round()

    def get_parameters_bounds(self, which, as_array):
        assert isinstance(which, str), "'which' must be a string."
        permitted_keys = ['lower', 'upper', 'both']
        assert which in permitted_keys, \
            "'which' must be one of the following: ['lower', 'upper', 'both']"\
            "Look for typos or white spaces."
        assert isinstance(as_array, bool), "'as_array' must be boolean."

        parameters = self.problem["parameters"]
        if which == "lower":
            lbs = []
            for i, parameter in enumerate(parameters):
                for k, v in parameter.items():
                    lbs.append(copy.deepcopy(v[0]))
            if as_array:
                lbs = np.asarray(lbs)
            return lbs
        elif which == "upper":
            ubs = []
            for i, parameter in enumerate(parameters):
                for k, v in parameter.items():
                    ubs.append(copy.deepcopy(v[1]))
            if as_array:
                ubs = np.asarray(ubs)
            return ubs
        elif which == "both":
            lbs, ubs = [], []
            for i, parameter in enumerate(parameters):
                for k, v in parameter.items():
                    lbs.append(copy.deepcopy(v[0]))
                    ubs.append(copy.deepcopy(v[1]))
            if as_array:
                lbs, ubs = np.asarray(lbs), np.asarray(ubs)
            return lbs, ubs

    def do_one_sampling_round(self):
        # assert False, "Not implemented."

        sampling_algo = self.algorithms['sampling']['algorithm']
        t0 = time.time()
        sampling_algo_status = sampling_algo.run()
        self.iteration_cpu_secs = time.time() - t0
        possible_status = ['success', 'failed', 'stopped']
        assert sampling_algo_status in possible_status, \
            "Got unrecognized status from the sampling algorithm."

        if sampling_algo_status == "success":
            self.solver_iteration += 1
            the_container = dict()
            self.collect_iteration_output(the_container)

            finished_solve, c = self.is_any_stop_criterion_met()
            if finished_solve:
                self.status = "FINISHED"
                print(self.settings["stop_criteria"][c], "is fulfilled.")
                # now we must add the live points to samples
                lpoints = sampling_algo.get_live_points()
                the_container["samples"].extend(lpoints)

            else:
                self.status = "SAMPLING_SUCCEDED"

            self.output_buffer.append(the_container)
            self.print_progress()
            self.notify_observers()

        elif sampling_algo_status == "stopped":
            self.solver_iteration += 1
            the_container = dict()
            self.collect_iteration_output(the_container)
            self.output_buffer.append(the_container)

            self.status = "SUBALGORITHM_STOPPED"
            self.notify_observers()

        elif sampling_algo_status == "failed":
            self.status = "SAMPLING_FAILED"
            assert False, "Not implemented yet."

    # The sorting function for NS algorithm
    def phi(self, p_mat, details = None):
        if self.settings["parallel"] == "no":
            return self.logl_serial(p_mat, details)
        elif self.settings["parallel"] == "multiproc":
            return self.logl_multiprocessing(p_mat, details)
        elif self.settings["parallel"] == "mpi":
            return self.logl_mpi(p_mat, details)

    def logl_serial(self, p_mat, details = None):
        '''
        :param p_mat: array MxP, each row is a point in P-space
        :param details: dictionary
        :return: list of M scalars that are log-likelihood(s)
        '''
        if details is not None:
            t0 = time.time()

        p_num, p_dim = np.shape(p_mat)
        logl_list = [0.]*p_num
        for i, p_vec in enumerate(p_mat):
            logl = self.problem["log_l"](p_vec)
            logl_list[i] = logl

        if details is not None:
            details["n_model_evals"] = len(p_mat)
            details["cpu_secs"] = time.time() - t0

        return logl_list

    def logl_multiprocessing(self, p_mat, details=None):
        assert False, "Not implemented yet."

    def logl_mpi(self, p_mat, details=None):
        assert False, "Not implemented yet."

    # Output Handling  TODO Remove some of these (see DS solver Using NS)
    def print_progress(self):
        print('Solver iteration:', self.solver_iteration)
        print('\t *live points contribution to evidence: %.5f %%' %
              (np.exp(self.log_zl - self.log_z)*100))

        sampling_algo = self.algorithms["sampling"]["algorithm"]
        best_live_point = sampling_algo.get_best_point()
        print('\t * best point:')
        print('\t * \t x:', best_live_point.x)
        print('\t * \t log_l:', best_live_point.f)

    def attach(self, o):
        self.observers.append(o)

    def detach(self, o):
        self.observers.pop(o)

    def notify_observers(self):
        for o in self.observers:
            o.update()

    def clear_output_buffer(self):
        self.output_buffer = []

    def collect_iteration_output(self, container):
        # assert False, "Not implemented yet."
        sampling_algo = self.algorithms['sampling']['algorithm']
        dpoints = sampling_algo.get_dead_points()
        run_details = sampling_algo.run_details

        container.update({
            "iteration": self.solver_iteration,
            "samples": dpoints,
            "nests": [],
            "log_z": {"hat": None},
            "post_prior_kldiv": None,
            "performance": {
                "n_proposals": run_details["n_evals"]["f"],
                "n_replacements": len(dpoints),
                "cpu_secs": {
                    "proposals_generation":
                        run_details["cpu_secs"]["proposals"],
                    "lkhd_evals": run_details["cpu_secs"]["evaluations"],
                    "total": self.iteration_cpu_secs
                }
            }
        })
        new_nests = self.do_nests()
        container["nests"].extend(new_nests)

        self.update_logz_and_kldiv()
        container["log_z"]["hat"] = self.log_z
        container["post_prior_kldiv"] = self.hd

    def do_nests(self):
        sampling_algo = self.algorithms["sampling"]["algorithm"]
        dpoints = sampling_algo.get_dead_points()
        nests = []
        for i, dpoint in enumerate(dpoints):
            self.nest_idx += 1

            log_w = self.log_x + self.log_w0
            log_l = dpoint.f
            log_lw = log_l + log_w
            log_zd_new = np.logaddexp(self.log_zd, log_lw)

            term1 = np.exp(log_lw - log_zd_new) * log_l
            factor1 = np.exp(self.log_zd - log_zd_new)
            factor2 = self.hd + self.log_zd
            if factor1 == 0 and factor2 == -np.inf:
                term2 = 0.0
            else:
                term2 = factor1 * factor2
            # updates
            self.hd = term1 + term2 - log_zd_new
            self.log_zd = log_zd_new
            self.log_x += self.log_t

            # store
            nest = {
                "idx": self.nest_idx,
                "log_x": self.log_x,
                "log_w": log_w,
                "log_lw": log_lw,
                "log_zd": log_zd_new
            }
            nests.append(nest)
            # self.results["log_x"].append(self.log_x)
            # self.results["log_zd"].append(self.log_zd)
            # self.results["hd"].append(self.hd)
        return nests

    def update_logz_and_kldiv(self):
        sampling_algo = self.algorithms['sampling']['algorithm']
        lpoints = sampling_algo.get_live_points()

        self.log_zl = -np.inf
        log_w_live = self.log_x - np.log(self.nlive)
        for i, lpoint in enumerate(lpoints):
            log_l = lpoint.f
            log_lw = log_l + log_w_live
            self.log_zl = np.logaddexp(self.log_zl, log_lw)
        # update log-evidence
        self.log_z = np.logaddexp(self.log_zd, self.log_zl)

        # Update information H
        h = self.hd
        log_z = self.log_zd
        for i, lpoint in enumerate(lpoints):
            log_l = lpoint.f
            log_lw = log_l + log_w_live
            log_z_new = np.logaddexp(log_z, log_lw)

            term1 = np.exp(log_lw - log_z_new) * log_l
            factor1 = np.exp(log_z - log_z_new)
            factor2 = (h + log_z)
            if factor1 == 0 and factor2 == -np.inf:
                term2 = 0.0
            else:
                term2 = factor1 * factor2
            h = term1 + term2 - log_z_new
            log_z = log_z_new

    def is_any_stop_criterion_met(self):
        for c, criterion in enumerate(self.settings["stop_criteria"]):
            for k, v in criterion.items():
                if k == "contribution_to_evidence":
                    live_contrib_frac = np.exp(self.log_zl - self.log_z)
                    if live_contrib_frac < v:
                        return True, c
        return False, 0

    # Post-solve steps
    def do_post_solve_steps(self, om):
        print('Solver executes post-solve steps.')
        # samples weights
        log_lw = om.get_log_lw()
        logl_remainder = om.get_logl_of_last(self.nlive)
        weights = self.compute_samples_weights(log_lw, logl_remainder)
        om.add_samples_weights(weights)
        # log-evidence statistics
        log_l = om.get_logl()
        log_z_mean, log_z_sdev = self.compute_log_z_statistics(log_l)
        om.add_logz_statistics(log_z_mean, log_z_sdev)

    def compute_samples_weights(self, log_lws, logl_remainder):
        log_z = self.log_z
        # dead points
        weights = [np.exp(log_lw - log_z) for log_lw in log_lws]
        # remainder
        n = self.nlive
        log_xm = self.log_x
        t_hat = float(n)/float(n+1)
        log_t_hat = np.log(t_hat)
        factor = np.log(1 - t_hat)
        for i in range(n):
            log_l = logl_remainder[i]
            log_w = factor + log_xm
            log_lw = log_l + log_w
            weight = np.exp(log_lw - log_z)
            weights.append(weight)
            log_xm += log_t_hat

        return weights

    def compute_log_z_statistics(self, log_l):
        nlive = self.nlive
        nsamples = len(log_l)
        nrepeats = 30

        log_z = [-np.inf]*nrepeats
        for r in range(nrepeats):
            unif = np.random.random(nsamples)
            t_samples = [u**(1.0/float(nlive)) for u in unif]
            log_x = 0.0
            for i in range(nsamples):
                log_w = log_x + np.log(1.0 - t_samples[i])
                log_lw = log_l[i] + log_w
                log_z[r] = np.logaddexp(log_z[r], log_lw)
                log_x += np.log(t_samples[i])

        log_z_mean = np.mean(log_z)
        log_z_sdev = np.std(log_z)
        return log_z_mean, log_z_sdev
