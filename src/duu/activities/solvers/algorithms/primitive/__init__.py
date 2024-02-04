import abc

from duu.activities.solvers.algorithms import PrimitiveAlgorithm
from duu.activities.solvers.algorithms.primitive.suob import \
    SamplingUniformlyOneBox, SamplingUniformlyOneEllipsoid
from duu.activities.solvers.algorithms.primitive.sumb import \
    SamplingUniformlyMultipleBodies
from duu.activities.solvers.algorithms.primitive.clustering import \
    KMeans, XMeans
