import copy
import numpy as np
import time

from duu import utils
from duu.activities.interfaces import Subject, Observer
from duu.activities.solvers.algorithms.points import BayesPoint
from duu.activities.solvers.algorithms.composite.factory import \
    CompositeAlgorithmFactory
from duu.activities.solvers.algorithms.composite import \
    NestedSamplingWithGlobalSearch


class DesignSpaceSolverUsingNS(Subject):
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
        self.worst_alpha = 0.0
        self.a, self.n, self.r = None, None, None

    def set_problem(self, problem):
        # Check is already done by the activity manager.
        self.problem = problem

    def set_settings(self, settings):
        assert isinstance(settings, dict), \
            "The settings must be a dictionary."

        mandatory_keys = ['acceleration', 'parallel', 'points_schedule',
                          'stop_criteria']
        assert all(mkey in settings.keys() for mkey in mandatory_keys), \
            "The 'settings' keys must be the following:\n" \
            "['acceleration', 'parallel', 'points_schedule', " \
            "'stop_criteria']." \
            "Look for typos, white spaces or missing keys."

        pts_schedule = settings['points_schedule']
        assert isinstance(pts_schedule, list), \
            "'points_schedule' must be a list."
        for i, item in enumerate(pts_schedule):
            assert isinstance(item, tuple), \
                "'points_schedule' must contain (a, n, r) tuples, where:" \
                "a - reliability level in [0, 1]; " \
                "n - number of live points; " \
                "r - number of replacements attempts per iteration."
            if len(pts_schedule) > 1 and i < len(pts_schedule) - 1:
                n1, n2 = pts_schedule[i][1], pts_schedule[i+1][1]
                assert (n1 < n2), \
                    "The number of live points must always increase."

        self.settings = settings

    def set_algorithms(self, algos):
        assert self.problem is not None, \
            "Define 'problem' before the algorithms of 'ds_solver'."
        assert self.settings is not None, \
            "Define 'settings' before the algorithms of 'ds_solver'."
        assert isinstance(algos, dict), \
            "algorithms must be a dictionary."

        mandatory_keys = ['sampling']
        assert all(mkey in algos.keys() for mkey in mandatory_keys), \
            "The 'algorithms' of 'pe_solver' should be for the steps:\n" \
            "['sampling']. " \
            "Look for typos, white spaces or missing keys."

        algos_permitted = [NestedSamplingWithGlobalSearch.get_type()
                           + "-" +
                           NestedSamplingWithGlobalSearch.get_ui_name()]

        sampling_algo_name = algos["sampling"]["algorithm"]
        assert sampling_algo_name in algos_permitted, \
            "The algorithm specified for 'sampling' is not permitted."

        permitted_stop_criteria = ['inside_fraction']

        specified_stop_criteria = utils.keys_in(
            self.settings["stop_criteria"])
        for ssc in specified_stop_criteria:
            assert ssc in permitted_stop_criteria,\
                "Stop_criteria '" \
                + ssc + "' is not permitted in this context."

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
            self.sort_parameters_samples(sort_key='w', ascending=False)
            self.p_coords, self.p_weights = \
                self.get_parameters_samples_coords_and_weights()
            self._p_num, self._p_dims = np.shape(self.p_coords)
            self._inside_frac = None

            sampling_algo = self.algorithms["sampling"]["algorithm"]
            lbs, ubs = self.get_design_vars_bounds(which="both", as_array=True)
            sampling_algo.set_bounds(lbs, ubs)
            sampling_algo.set_sorting_func(self.phi)

            self.a, self.n, self.r = self.settings['points_schedule'][0]
            sampling_algo.settings['nlive'] = self.n
            sampling_algo.settings['nreplacements'] = self.r
            self.phase = "DETERMINISTIC"
            sampling_algo.set_sorting_func(self.phi)
            sampling_algo.initialize_live_points()
            print("Phase INITIAL is over.")

        while (self.phase == "DETERMINISTIC"):
            self.do_one_sampling_round()
            if self.is_deterministic_phase_over():
                print("Phase DETERMINISTIC is over.")
                self.phase = "TRANSITION"
                break

        if self.phase == "TRANSITION":
            self.solver_iter_phase1_ended = self.solver_iteration
            self.phase = "PROBABILISTIC"
            self.worst_alpha = 0.0
            sampling_algo.evaluate_live_points_fvalue()
            sampling_algo.live_points.sort()
            self.worst_alpha = sampling_algo.get_live_points()[0].f
            print("Phase TRANSITION is over.")

        while(self.phase == "PROBABILISTIC"):
            if self.status == "SAMPLING_SUCCEDED":
                worst = sampling_algo.get_live_points()[0].f
                self.worst_alpha = copy.deepcopy(worst)
                self.set_live_points_according_schedule()
                self.do_one_sampling_round()
            elif self.status == "SAMPLING_FAILED":
                print("Sampling algorithm failed.")
                break
            elif self.status == "SUBALGORITHM_STOPPED":
                print("A subalgorithm finished.")
                break
            elif self.status == "FINISHED":
                print("Phase PROBABILISTIC is over.")
                print("Solver finished solving the problem.")
                break
            else:
                assert False, "Unrecognizable solver status."

    def get_design_vars_bounds(self, which, as_array):
        assert isinstance(which, str), "'which' must be a string."
        permitted_keys = ['lower', 'upper', 'both']
        assert which in permitted_keys, \
            "'which' must be one of the following: " \
            "['lower', 'upper', 'both']. Look for typos or white spaces."
        assert isinstance(as_array, bool), "'as_array' must be boolean."

        design_vars = self.problem["design_variables"]
        if which == "lower":
            lbs = []
            for i, des_var in enumerate(design_vars):
                for k, v in des_var.items():
                    lbs.append(copy.deepcopy(v[0]))
            if as_array:
                lbs = np.asarray(lbs)
            return lbs
        elif which == "upper":
            ubs = []
            for i, des_var in enumerate(design_vars):
                for k, v in des_var.items():
                    ubs.append(copy.deepcopy(v[1]))
            if as_array:
                ubs = np.asarray(ubs)
            return ubs
        elif which == "both":
            lbs, ubs = [], []
            for i, des_var in enumerate(design_vars):
                for k, v in des_var.items():
                    lbs.append(copy.deepcopy(v[0]))
                    ubs.append(copy.deepcopy(v[1]))
            if as_array:
                lbs, ubs = np.asarray(lbs), np.asarray(ubs)
            return lbs, ubs

    def sort_parameters_samples(self, sort_key, ascending):
        if ascending:
            sorted_samples = sorted(self.problem['parameters_samples'],
                                    key=lambda k: k[sort_key], reverse=False)
        else:
            sorted_samples = sorted(self.problem['parameters_samples'],
                                    key=lambda k: k[sort_key], reverse=True)
        self.problem['parameters_samples'] = sorted_samples

    def get_parameters_samples_coords_and_weights(self):
        p_samples = self.problem['parameters_samples']
        p_num = len(p_samples)
        p_dims = len(p_samples[0]['c'])

        p_coords = np.ndarray((p_num, p_dims))
        p_weights = np.ndarray(p_num)
        for i, sample in enumerate(p_samples):
            p_coords[i, :] = sample['c']
            p_weights[i] = sample['w']
        return p_coords, p_weights

    def set_live_points_according_schedule(self):
        pts_schedule = self.settings["points_schedule"]
        sampling_algo = self.algorithms['sampling']['algorithm']
        for i, configuration in enumerate(pts_schedule):
            a, n, r = configuration
            if i+1 < len(pts_schedule):
                a_next, n_next, r_next = pts_schedule[i + 1]
                if a <= self.worst_alpha < a_next:
                    self.a, self.n, self.r = a, n, r
                    break
            else:
                self.a, self.n, self.r = a, n, r
                break
        sampling_algo.top_up_to(self.n)
        sampling_algo.settings["nreplacements"] = self.r

    def do_one_sampling_round(self):
        sampling_algo = self.algorithms['sampling']['algorithm']
        t0 = time.time()
        sampling_algo_status = sampling_algo.run()
        self.iteration_cpu_secs = time.time() - t0
        possible_status = ['success', 'failed', 'stopped']
        assert sampling_algo_status in possible_status, \
            "Got unrecognised status from the sampling algorithm."

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
    def phi(self, d_mat, details = None):
        if self.phase == "DETERMINISTIC":
            if self.settings["parallel"] is True:
                return self.score_parallel(d_mat, details)
            else:
                return self.score_serial(d_mat, details)

        elif self.phase == "PROBABILISTIC":
            if self.settings["parallel"] is True:
                return self.efp_parallel(d_mat, details)
            else:
                return self.efp_serial(d_mat, details)

        else:
            assert False, "The phi function of the phase not found."

    def score_serial(self, d_mat, details = None):
        '''
        :param d_mat: array MxD, each row is a point in D-space
        :return: list of M scalars that are not fullfilment probabilities
        '''
        if details is not None:
            t0 = time.time()

        theta_best = self.problem['parameters_best_estimate']
        d_num, d_dim = np.shape(d_mat)
        score_list = [0.]*d_num
        for i, d_vec in enumerate(d_mat):

            g_vec = self.problem["constraints"](d_vec, theta_best)
            # The score can be a better function but now KISS.
            score = 0.0  # this is not fullfilment probability
            if np.all(g_vec >= 0.0):
                score = 1.0
            score_list[i] = score

        if details is not None:
            details["n_model_evals"] = len(d_mat)
            details["cpu_secs"] = time.time() - t0

        return score_list

    def score_parallel(self, d, details = None):
        '''
        :param d_mat: array MxD, each row is a point in D-space
        :return: list of M scalars that are not fullfilment probabilities
        '''
        assert False, "phi1_parallel is not implemented yet."

    def efp_serial(self, d_mat, details = None):
        '''
        :param d_mat: array MxD, each row is a point in D-space
        :return: list of M scalars representing estimated constraints
        fulfillment probability
        '''
        if details is not None:
            details["n_model_evals"] = 0
            t0 = time.time()

        p_samples = self.problem['parameters_samples']
        p_weights = np.array(
            [p['w'] for p in self.problem['parameters_samples']])
        # WARNING: We assume here that p_weights are normalized already!!!
        d_num, d_dim = np.shape(d_mat)
        efps_list = [0.0]*d_num
        for i, d_vec in enumerate(d_mat):
            efp = 0.0
            rpr = 1.0
            for j, p_sample in enumerate(p_samples):
                p_vec = p_sample['c']
                g_vec = self.problem["constraints"](d_vec, p_vec)
                if np.all(g_vec >= 0.0):
                    efp = round(efp + p_weights[j], ndigits=15)
                    if efp > 1.0:
                        print("EFP > 1.0:", efp, "j=", j)

                rpr = round(rpr - p_weights[j], ndigits=15)
                if self.settings["acceleration"] is True:
                    if self.worst_alpha > efp + rpr:
                        break

            if not details is None:
                details["n_model_evals"] += j + 1

            efps_list[i] = efp

        if details is not None:
            details["cpu_secs"] = time.time() - t0

        return efps_list

    def efp_parallel(self, d_mat, details = None):
        '''
        :param d_mat: array MxD, each row is a point in D-space
        :return: list of M scalars representing estimated constraints
        fulfillment probability
        '''
        assert False, "not implemented yet."

    # Phase changing checks
    def is_deterministic_phase_over(self):
        sampling_algo = self.algorithms["sampling"]["algorithm"]
        lpts = sampling_algo.get_live_points()
        inside = np.array([point.f >= 1.0 for point in lpts])
        return np.all(inside)

    # Output Handling
    def print_progress(self):
        print("Solver iteration:", self.solver_iteration)

        print("\t *Phase:", self.phase)

        print("\t *lowest F value: %.5f "
              "| live points: %d "
              "| replacement attempts: %d"
              %(self.worst_alpha, self.n, self.r))

        print("\t *Fraction of live points in DS~%.2f%%: %.4f"
              % (self.problem['target_reliability']*100, self._inside_frac))

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
        sampling_algo = self.algorithms['sampling']['algorithm']
        dpoints = sampling_algo.get_dead_points()
        run_details = sampling_algo.run_details

        container.update({
            "iteration": self.solver_iteration,
            "phase": self.phase,
            "samples": dpoints,
            "performance": {
                "n_phi_evaluations": run_details["n_evals"]["f"],
                "n_model_evaluations": run_details["n_evals"]["model"],
                "n_replacements": len(dpoints),
                "cpu_secs": {
                    "proposals_generation":
                        run_details["cpu_secs"]["proposals"],
                    "phi_evaluation": run_details["cpu_secs"]["evaluations"],
                    "total": self.iteration_cpu_secs
                }
            }
        })

    def is_any_stop_criterion_met(self):
        for c, criterion in enumerate(self.settings["stop_criteria"]):
            for k, v in criterion.items():
                if k == "inside_fraction":
                    fraction_wanted = v

                    sampling_algo = self.algorithms["sampling"]["algorithm"]
                    lpoints = sampling_algo.get_live_points()

                    nlive = len(lpoints)
                    num_of_lpoints_inside = 0
                    alpha = self.problem['target_reliability']
                    for lpoint in lpoints:
                        if lpoint.f >= alpha:
                            num_of_lpoints_inside += 1

                    pts_schedule = self.settings['points_schedule']
                    n_list = [item[1] for item in pts_schedule]
                    nlive_max = max(n_list)
                    self._inside_frac = num_of_lpoints_inside / float(nlive_max)

                    if self._inside_frac >= fraction_wanted:
                        return True, c
        return False, 0

    # Post-solve steps
    def do_post_solve_steps(self, om):
        print('Solver has no post-solve steps to do.')