import numpy as np
from toqito.states import bell
from qustop import Ensemble, OptDist, State

# Define the ensemble initially defined in arXiv:1107.3224:
dims = [2, 2, 2, 2]
ensemble = Ensemble(
    [
        State(np.kron(bell(0), bell(0)), dims),
        State(np.kron(bell(2), bell(1)), dims),
        State(np.kron(bell(3), bell(1)), dims),
        State(np.kron(bell(1), bell(1)), dims),
    ]
)

# Determine the optimal probability of distinguishing
# via PPT measurements with minimum error:
ppt_min = OptDist(ensemble, "ppt", "min-error")
ppt_min.solve()
print(ppt_min.value)  # 7/8

# Determine the optimal probability of distinguishing
# via separable measurements with minimum-error:
sep_min = OptDist(ensemble, "sep", "min-error")
sep_min.solve()
print(sep_min.value)  # 3/4

# Determine the optimal probability of distinguishing
# via PPT measurements unambiguously:
ppt_unambig = OptDist(ensemble, "ppt", "unambiguous")
ppt_unambig.solve()
print(ppt_unambig.value)  # 3/4
