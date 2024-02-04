import pickle
from os.path import exists

from duu.activities.solvers.algorithms.points import XFPoint


class DesignSpaceOutputManager:
    def __init__(self, cs_folder):
        assert exists(cs_folder)
        self.cs_folder = cs_folder
        self.output_filename = 'output'
        self.output_path_filename_extn = self.cs_folder + \
                                      self.output_filename + '.pkl'
        self.output = {
            "solution": {
                "deterministic_phase": {
                    "samples": {
                        "coordinates": [],
                        "phi": []
                    },
                    "iterations": 1
                },
                "probabilistic_phase": {
                    "samples": {
                        "coordinates": [],
                        "phi": []
                    }
                }
            },
            "performance": []
        }

    def add(self, out_content):
        self.add_to_solution(out_content)
        self.add_to_performance(out_content)

    def add_to_solution(self, out_content):
        for container in out_content:
            phase = container["phase"]
            samples = container["samples"]
            if phase == "DETERMINISTIC":
                root = self.output["solution"]["deterministic_phase"]
                root["iterations"] = container["iteration"]
            elif phase == "PROBABILISTIC":
                root = self.output["solution"]["probabilistic_phase"]

            coords = (XFPoint.coords_of(container["samples"])).tolist()
            root["samples"]["coordinates"].extend(coords)
            fvalues = XFPoint.fvalues_of(container["samples"]).tolist()
            root["samples"]["phi"].extend(fvalues)

    def add_to_performance(self, out_content):
        for container in out_content:
            root = container["performance"]
            element = {
                "iteration": container["iteration"],
                "n_proposals": root["n_phi_evaluations"],
                "n_model_evals": root["n_model_evaluations"],
                "n_replacements": root["n_replacements"],
                "cpu_secs": {
                    "proposals":
                        root["cpu_secs"]["proposals_generation"],
                    "phi_evals": root["cpu_secs"]["phi_evaluation"],
                    "total": root["cpu_secs"]["total"]
                }
            }
            self.output["performance"].append(element)

    def write_to_disk(self):
        with open(self.output_path_filename_extn, 'wb') as file:
            pickle.dump(self.output, file)


if __name__ == "__main__":
    import numpy as np
    from os import getcwd

    ns, nd = 4, 3
    c = np.ones((ns, nd), dtype=float)
    f = np.ones(ns, dtype=float)
    out_test = []
    for i in range(1, 5):
        if i < 3:
            phase = "DETERMINISTIC"
        else:
            phase = "PROBABILISTIC"

        cont = {
            "iteration": i,
            "phase": phase,
            "samples": XFPoint.list_from(i*c, i*f),
            "performance": {
                "n_phi_evaluations": ns,
                "n_model_evaluations": ns,
                "n_replacements": ns/2,
                "cpu_secs": {
                    "proposals_generation": 0.01,
                    "phi_evaluation": 0.04,
                    "total": 0.1
                }
            }
        }
        out_test.append(cont)

    dsom = DesignSpaceOutputManager(getcwd())
    dsom.add(out_test)
    print(dsom.output)
