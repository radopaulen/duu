import pickle

from duu.activities import ActivityManager
from duu.activities.output import ParameterEstimationOutputManager
from duu.activities.solvers.factory import SolverFactory


class ParameterEstimationManager(ActivityManager):
    def __init__(self, activity_form):
        super().__init__()
        self.check_activity_type(activity_form["activity_type"])

        self.settings = None
        settings = activity_form["activity_settings"]
        self.check_settings(settings)
        self.settings = settings

        problem = activity_form["problem"]
        self.check_problem(problem)
        self.problem = problem

        solver_name = activity_form["solver"]["name"]
        solver_settings = activity_form["solver"]["settings"]
        solver_algorithms = activity_form["solver"]["algorithms"]
        self.solver = SolverFactory.create(solver_name,
                                           self.problem,
                                           solver_settings,
                                           solver_algorithms)

        self.solver.attach(self)

        cs_path = self.settings["case_path"]
        cs_name = self.settings["case_name"]
        self.cs_folder = cs_path + "/" + cs_name + "/"
        self.output_manager = ParameterEstimationOutputManager(self.cs_folder)

    def check_activity_type(self, a_type):
        assert a_type == "pe", \
            "The activity type must be \"pe\". Recheck the activity form."

    def check_problem(self, problem):
        assert isinstance(problem, dict), \
            "'problem' must be a dictionary."

        mandatory_keys = ['log_pi', 'log_l', 'parameters']
        assert all(mkey in problem.keys() for mkey in mandatory_keys), \
            "The 'problem' keys must include:\n" \
            "['log_pi', 'log_l', 'parameters']." \
            "Look for typos, white spaces or missing keys."

        parameters = problem["parameters"]
        assert isinstance(parameters, list), \
            "'parameters' must be a list of dictionaries."
        for i, item in enumerate(parameters):
            assert isinstance(item, dict), \
                "All items of 'parameters' are dictionaries."
            assert len(item.keys()) == 1, \
                "'Items in 'parameters' must be a single key-value dictionary."
            for k, v in item.items():
                assert isinstance(v, list), \
                    "The value of any 'item' in 'parameters' must be as " \
                    "[<lower_bound>, <upper_bound>]."
                assert len(v) == 2, \
                    "A parameter must have specified exactly one lbound and " \
                    "one ubound."
                assert v[0] < v[1], \
                    "Bad input: The lbound > ubound for parameter '" \
                    + str(k) + "'."

    def solve_problem(self):
        self.solver.solve()

    def update(self):
        if self.is_time_to_save():
            self.output_manager.add(self.solver.output_buffer)
            self.output_manager.write_to_disk()
            self.solver.clear_output_buffer()
            self.save_solution_state()

            solver_status = self.solver.status
            if solver_status in ["FINISHED", "SUBALGORITHM_STOPPED"]:
                self.solver.do_post_solve_steps(self.output_manager)
                self.output_manager.write_to_disk()

    def is_time_to_save(self):
        if self.solver.status in ["finished", "sub_algo_stopped"]:
            return True

        save_period = self.settings["save_period"]
        solver_iter = self.solver.solver_iteration
        return solver_iter % save_period == 0

    def save_solution_state(self):
        with open(self.cs_folder + 'solution_state.pkl', 'wb') as file:
            pickle.dump(self.__dict__, file)
