import matplotlib.pyplot as plt
import numpy as np
from os import getcwd
import pickle
from scipy.stats import multivariate_normal
import time

from duu import DUU


'''
Parameter Estimation - Test 1:
Find the posterior distribution given a uniform prior
and a Gaussian likelihood.
'''


def the_log_prior(xx):
    return -1.0e500


centre = [0, 2]
spread = [[1, 1.5],
          [1.5, 4]]


def the_log_lkhd(xx):
    gaussian = multivariate_normal(mean=centre,
                                   cov=spread)
    xxshape = np.shape(xx)
    if len(xxshape) == 1:
        f = np.log(gaussian.pdf(xx))
    else:
        nx, d = xxshape
        f = np.asarray([np.log(gaussian.pdf(xx[i, :]))
                        for i in range(nx)])
    return f


an_activity_form = {
    "activity_type": "pe",

    "activity_settings": {
        "case_name": "PeTest1",
        "case_path": getcwd(),
        "resume": False,
        "save_period": 1
    },

    "problem": {
        "log_pi": the_log_prior,
        "log_l": the_log_lkhd,
        "parameters": [
            {"a": [-5, 5]},
            {"b": [-10, 10]}
        ]
    },

    "solver": {
        "name": "pe-ns",
        "settings": {
            "parallel": "no",
            "stop_criteria": [
                {"contribution_to_evidence": 0.05}
            ]
        },
        "algorithms": {
            "sampling": {
                "algorithm": "mc_sampling-ns_global",
                "settings": {
                     "nlive": 100,
                     "nreplacements": 50,
                     "prng_seed": 1989,
                     "f0": 0.3,
                     "alpha": 0.2,
                     "stop_criteria": [
                         {"max_iterations": 100000}
                     ],
                     "debug_level": 0,
                     "monitor_performance": False
                 },
                "algorithms": {
                    "replacement": {
                        "sampling": {
                            "algorithm": "suob-ellipsoid"
                        }
                    }
                }
            }
        }
    }
}

the_duu = DUU(an_activity_form)
t0 = time.time()
the_duu.solve()
cpu_time = time.time() - t0
print('CPU seconds', cpu_time)

cs_path = an_activity_form["activity_settings"]["case_path"]
cs_name = an_activity_form["activity_settings"]["case_name"]

with open(cs_path + '/' + cs_name + '/' + 'output.pkl', 'rb')\
        as file:
    output = pickle.load(file)

log_z_mean = output["solution"]["log_z"]["hat"]
log_z_sdev = output["solution"]["log_z"]["sdev"]
h = output["solution"]["post_prior_kldiv"]
print('log Z =', log_z_mean, '+/-', log_z_sdev)
print('H =', h)


samples = output["solution"]["samples"]
weights = output["solution"]["samples"]["weights"]
samples_coords = np.empty((0, 2))
samples_weights = np.empty(0)
for i, sample in enumerate(samples["coordinates"]):
    samples_coords = np.append(samples_coords, [sample],
                               axis=0)
    samples_weights = np.append(samples_weights, [weights[i]],
                                axis=0)

fig1 = plt.figure()
x = samples_coords[:, 0]
y = samples_coords[:, 1]
plt.scatter(x, y, s=10, c='b')

truth = np.random.multivariate_normal(mean=centre,
                                      cov=spread,
                                      size=1000)
x = truth[:, 0]
y = truth[:, 1]
plt.scatter(x, y, s=10, c='r')


fig2 = plt.figure()
x = np.arange(len(samples_weights))
y = samples_weights
plt.plot(x, y, c='g')


fig3, ax = plt.subplots(1)
x = [item["iteration"] for item in output["performance"]]
y = [item["cpu_secs"]["proposals"]
     for item in output["performance"]]
ax.plot(x, y, 'b-', label='proposals generation')

x = [item["iteration"] for item in output["performance"]]
y = [item["cpu_secs"]["lkhd_evals"]
     for item in output["performance"]]
ax.plot(x, y, 'r-', label='likelihood evaluations')

x = [item["iteration"] for item in output["performance"]]
y = [item["cpu_secs"]["total"] for item in output["performance"]]
ax.plot(x, y, 'g--', label='total')

ax.set_ylabel('CPU seconds')
ax.grid()
ax.legend()

plt.show()
