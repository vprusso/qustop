"""
Define the following sets of states:

|ϕ_1> = 1/√2|00> + 1/√2|11>, |ϕ_2> = 1/√2|00> - 1/√2|11>,
|ϕ_3> = 1/√2|01> + 1/√2|10>, |ϕ_4> = 1/√2|01> - 1/√2|10>.

Assuming a uniform distribution p_1 = p_2 = p_3 = p_4 = 1/4, the
optimal probability of distinguishing the Bell states via PPT
measurements is at most
"""
import numpy as np

from toqito.states import bell
from qustop.core.ensemble import Ensemble
from qustop.core.state import State
from qustop.opt_dist.opt_dist import OptDist


# Construct the corresponding density matrices of the Bell states.
states = [
    State(bell(0) * bell(0).conj().T, dims=[2, 2]),
    State(bell(1) * bell(1).conj().T, dims=[2, 2]),
    State(bell(2) * bell(2).conj().T, dims=[2, 2]),
    State(bell(3) * bell(3).conj().T, dims=[2, 2])
]
ensemble = Ensemble(states=states, probs=[1/4, 1/4, 1/4, 1/4])
sd = OptDist(ensemble=ensemble, 
                measurement="ppt", 
                error="min-error")
sd.solve()
print(sd.value)