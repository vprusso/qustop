# ![logo](./docs/figures/logo.svg "logo") 

# QUSTOP

[![build status](http://img.shields.io/travis/vprusso/toqito.svg?style=plastic)](https://travis-ci.org/vprusso/qustop)
[![doc status](https://readthedocs.org/projects/toqito/badge/?version=latest&style=plastic)](https://qustop.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/vprusso/toqito/branch/main/graph/badge.svg?style=plastic)](https://codecov.io/gh/vprusso/qustop)

*NOTE*: The `qustop` package is still is under development. 

The `qustop` (QUantum STate OPtimizer) package is a Python toolkit for studying
various quantum state optimization scenarios including calculating optimal
values for quantum state distinguishability, quantum state exclusion, quantum
state cloning, and more.

## Applications

The `qustop` package can be used to:

- Calculate and approximate optimal probabilities of distinguishing quantum
  states over positive, PPT, and separable measurements with either minimum-error
  or unambiguously.

- Calculate and approximate optimal probabilities of excluding quantum states
  with either minimum-error or unambiguously.

## Installation

See the [installation guide](https://qustop.readthedocs.io/en/latest/getting_started.html).

## Usage

See the [documentation](https://qustop.readthedocs.io/en/latest/index.html).

## Examples

For more examples, please consult
[`qustop/examples`](https://github.com/vprusso/qustop/tree/main/examples/)
as well as the `qustop` [introductory
tutorial](https://qustop.readthedocs.io/en/latest/intro_tutorial.html).

### Quantum state distinguishability

Further examples on quantum state distinguishability can be found in the
[`qustop/examples/opt_dist`](https://github.com/vprusso/qustop/tree/main/examples/opt_dist)
directory.

Consider the following Bell states:

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2c}](https://latex.codecogs.com/svg.latex?\Large&space;|\psi_0\rangle=\frac{|00\rangle+|11\rangle}{\sqrt{2}},\quad|\psi_1\rangle=\frac{|01\rangle+|10\rangle}{\sqrt{2}},) 

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2c}](https://latex.codecogs.com/svg.latex?\Large&space;|\psi_2\rangle=\frac{|01\rangle-|10\rangle}{\sqrt{2}},\quad|\psi_3\rangle=\frac{|00\rangle-|11\rangle}{\sqrt{2}}.) 

We will be using these states to consider a number of applications in the realm
of quantum state distinguishability.

#### Distinguishing two orthogonal states

A result of [arXiv:0007098](https://arxiv.org/abs/quant-ph/0007098) states that
any two orthogonal pure states can be distinguished perfectly by LOCC
measurements. As the optimal probability of distinguishing via LOCC
measurements is a lower bound on positive, PPT, separable, etc., we should
expect to also see a value of `1` to indicate perfect probability of
distinguishing.

```python
from toqito.states import bell
from qustop import State, Ensemble, OptDist

dims = [2, 2]
states = [
    State(bell(0) * bell(0).conj().T, dims),
    State(bell(1) * bell(1).conj().T, dims)
]
probs = [1/2, 1/2]
ensemble = Ensemble(states, probs)

sep_res = OptDist(ensemble, "sep", "min-error")
sep_res.solve()

ppt_res = OptDist(ensemble, "ppt", "min-error")
ppt_res.solve()

pos_res = OptDist(ensemble, "pos", "min-error")
pos_res.solve()
```

Checking the respective values of the solved instances, we see that all of the
values are equal to one, which indicate that the two pure states are indeed
perfectly distinguishable under PPT, separable, and positive measurements.

```python
>>> print(pos_res.value)
0.9999999999384911
>>> print(ppt_res.value)
1.0000000047560667
>>> print(sep_res.value)
0.9999999995278338
```

#### Four indistinguishable orthogonal maximally entangled states

It was shown in [arXiv:1205.1031](https://arxiv.org/abs/1205.1031) and later
extended in [arXiv:1307.3232](https://arxiv.org/abs/1307.3232) that for the
following set of states

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2c}](https://latex.codecogs.com/svg.latex?\Large&space;\rho_0=|\psi_0\rangle|\psi_0\rangle\langle\psi_0|\langle\psi_0|,\quad\rho_1=|\psi_1\rangle|\psi_3\rangle\langle\psi_1|\langle\psi_3|,) 

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2c}](https://latex.codecogs.com/svg.latex?\Large&space;\rho_2=|\psi_2\rangle|\psi_3\rangle\langle\psi_2|\langle\psi_3|,\quad\rho_3=|\psi_3\rangle|\psi_3\rangle\langle\psi_3|\langle\psi_3|,) 

that the optimal probability of distinguishing via a PPT measurement should
yield an optimal probability of 7/8.

```python
import numpy as np

from toqito.states import bell
from qustop import State, Ensemble, OptDist

dims = [2, 2, 2, 2]
rho_0 = np.kron(bell(0), bell(0)) * np.kron(bell(0), bell(0)).conj().T
rho_1 = np.kron(bell(2), bell(1)) * np.kron(bell(2), bell(1)).conj().T
rho_2 = np.kron(bell(3), bell(1)) * np.kron(bell(3), bell(1)).conj().T
rho_3 = np.kron(bell(1), bell(1)) * np.kron(bell(1), bell(1)).conj().T

ensemble = Ensemble([
    State(rho_0, dims), State(rho_1, dims),
    State(rho_2, dims), State(rho_3, dims)
])

sd = OptDist(ensemble, "ppt", "min-error")
sd.solve()
```

Indeed the optimal value obtained via `qustop` is equal to 7/8:

```python
# 7/8 \approx 0.875
>>> print(sd.value)
0.8749769201568257
```

It was also shown in [arXiv:1205.1031](https://arxiv.org/abs/1205.1031) that the optimal
probability of distinguishing amongst these same state unambiguously via PPT measurements was
equal to 3/4.

```python
sd = OptDist(ensemble, "ppt", "unambiguous")
sd.solve()

# 3/4 = 0.75
>>> print(sd.value)
0.749999999939434
```

#### Entanglement cost of distinguishing Bell states

One may ask whether the ability to distinguish a state can be improved by
making use of an auxiliary resource state.

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2c}](https://latex.codecogs.com/svg.latex?\Large&space;|\tau_{\epsilon}\rangle=\sqrt{\frac{1+\epsilon}{2}}|00\rangle+\sqrt{\frac{1-\epsilon}{2}}|11\rangle),

for some &epsilon; in [0, 1].

It was shown in [arXiv:1408.6981](https://arxiv.org/abs/1408.6981) that the
probability of distinguishing four Bell states with a resource state via PPT
measurements or separable measurements is given by the closed-form expression

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2c}](https://latex.codecogs.com/svg.latex?\Large&space;\text{opt}_{\text{PPT}}(\eta)=\text{opt}_{\text{SEP}}(\eta)=\frac{1}{2}\left(1+\sqrt{1-\epsilon^2}\right)) 

where the ensemble is defined as

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2c}](https://latex.codecogs.com/svg.latex?\Large&space;\eta=\left\(|\psi_0\rangle\otimes|\tau_{\epsilon}\rangle,\quad|\psi_1\rangle\otimes|\tau_{\epsilon}\rangle,\quad|\psi_2\rangle\otimes|\tau_{\epsilon}\rangle,\quad|\psi_3\rangle\otimes|\tau_{\epsilon}\rangle\right\))


Using `qustop`, we may encode this scenario as follows.

```python
import numpy as np

from toqito.states import basis, bell
from qustop import State, Ensemble, OptDist


e_0, e_1 = basis(2, 0), basis(2, 1)

eps = 0.5
tau = np.sqrt((1 + eps) / 2) * np.kron(e_0, e_0) + np.sqrt((1 - eps) / 2) * np.kron(e_1, e_1)

dims = [2, 2, 2, 2]
states = [
    State(np.kron(bell(0), tau), dims),
    State(np.kron(bell(1), tau), dims),
    State(np.kron(bell(2), tau), dims),
    State(np.kron(bell(3), tau), dims),
]
probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
ensemble = Ensemble(states, probs)

sep_res = OptDist(ensemble, "sep", "min-error")
sep_res.solve()

ppt_res = OptDist(ensemble, "ppt", "min-error")
ppt_res.solve()

eq = 1 / 2 * (1 + np.sqrt(1 - eps ** 2))
```

Note that when we print out the optimal values for both separable and PPT
measurements that the values obtained agree with the closed form expression.

```python
>>> print(eq)
0.9330127018922193
>>> print(ppt_res.value)
0.933010488554166
>>> print(sep_res.value)
0.9330124607534689
```

It was also shown in [arXiv:1408.6981](https://arxiv.org/abs/1408.6981) that
the closed-form probability of distinguishing three Bell states with a resource
state using separable measurements to be given by the closed-form expression:

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2c}](https://latex.codecogs.com/svg.latex?\Large&space;\text{opt}_{\text{SEP}}(\eta)=\frac{1}{3}\left(2+\sqrt{1-\epsilon^2}\right)) 

where the ensemble is defined as

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2c}](https://latex.codecogs.com/svg.latex?\Large&space;\eta=\left\(|\psi_0\rangle\otimes|\tau_{\epsilon}\rangle,\quad|\psi_1\rangle\otimes|\tau_{\epsilon}\rangle,\quad|\psi_2\rangle\otimes|\tau_{\epsilon}\rangle\right\))

Using `qustop`, we may encode this scenario as follows.

```python
import numpy as np

from toqito.states import basis, bell
from qustop import State, Ensemble, OptDist


e_0, e_1 = basis(2, 0), basis(2, 1)

eps = 0.5
tau = np.sqrt((1 + eps) / 2) * np.kron(e_0, e_0) + np.sqrt((1 - eps) / 2) * np.kron(e_1, e_1)

dims = [2, 2, 2, 2]
states = [
    State(np.kron(bell(0), tau), dims),
    State(np.kron(bell(1), tau), dims),
    State(np.kron(bell(2), tau), dims),
]
probs = [1 / 3, 1 / 3, 1 / 3]
ensemble = Ensemble(states, probs)

sep_res = OptDist(ensemble, "sep", "min-error", level=2)
sep_res.solve()

eq = 1 / 3 * (2 + np.sqrt(1 - eps**2))
```

Pritning the values of both the closed-form equation and the value obtained via
the SDP, we obtain:

```python
>>> print(sep_res.value)
0.9583057987150858
>>> print(eq)
0.9553418012614794
```

Note that the value of `sep_res.value` is actually a bit higher than `eq`. This
is because the separable value is calculated by a hierarchy of SDPs. At low
levels of the SDP, the problem can often converge to the optimal value, but
other times it is necessary to compute higher levels of the SDP to eventually
arrive at the optimal value. While this is intractable in general, in practice,
the SDP can often converge or at least get fairly close to the optimal value
for small problem sizes.

#### Werner hiding pairs

In [arXiv:0011042](https://arxiv.org/abs/quant-ph/0011042) and
[arXiv:0103098](https://arxiv.org/abs/quant-ph/0103098) a quantum data hiding
protocol that encodes a classical bit in a Werner hiding pair was provided.

A Werner hiding pair is defined as

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2c}](https://latex.codecogs.com/svg.latex?\Large&space;\sigma_0^n=\frac{\mathbb{I}\otimes\mathbb{I}+W_n}{n(n+1)}\quad\text{and}\quad\sigma_1^n=\frac{\mathbb{I}\otimes\mathbb{I}-W_n}{n(n-1)}) 

where 

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2c}](https://latex.codecogs.com/svg.latex?\Large&space;W_n=\sum_{i,j=0}^{n-1}|i\rangle\langle\text{}j|\otimes|j\rangle\langle\text{}i|\in\text{}U(\mathbb{C}^n\otimes\mathbb{C}^n)) 

is the swap operator defined for some dimension n >= 2.

It was show in [hdl:10012/9572](https://uwspace.uwaterloo.ca/handle/10012/9572) that 

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2c}](https://latex.codecogs.com/svg.latex?\Large&space;\text{opt}_{\text{PPT}}(\eta)=\frac{1}{2}+\frac{1}{n+1}) 

where the ensemble is defined as

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2c}](https://latex.codecogs.com/svg.latex?\Large&space;\eta=\(\sigma_0,\sigma_1\).)

Using `qustop`, we may encode this scenario as follows.

```python
import numpy as np
from toqito.perms import swap_operator
from qustop import Ensemble, State, OptDist


dim = 2
sigma_0 = (np.kron(np.identity(dim), np.identity(dim)) + swap_operator(dim)) / (dim * (dim + 1))
sigma_1 = (np.kron(np.identity(dim), np.identity(dim)) - swap_operator(dim)) / (dim * (dim - 1))

states = [State(sigma_0, [2, 2]), State(sigma_1, [2, 2])]
ensemble = Ensemble(states)

expected_val = 1 / 2 + 1 / (dim + 1)

sd = OptDist(ensemble=ensemble, 
             dist_measurement="ppt",
             dist_method="min-error",
             eps=1e-8)

sd.solve()
```

We can verify that the closed-form expression matches that of the value
returned from `qustop`.

```python
print(sd.value)
0.8333333333668715
print(expected_val)
0.8333333333333333
```

### State exclusion

The primary difference between the quantum state distinguishability
scenario and the quantum state exclusion scenario is that in the former,
Bob want to guess which state he was given, and in the latter, Bob wants to
guess which state he was *not* given.



### State cloning

(Coming soon).

## License

[GNU GPL v.3.0.](https://github.com/vprusso/qustop/blob/master/LICENSE)
