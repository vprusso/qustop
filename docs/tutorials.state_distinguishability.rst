Quantum state distinguishability
=================================

In this tutorial we are going to cover the problem of *quantum state
distinguishability* (sometimes analogously referred to as *quantum state
discrimination*). We are going to briefly describe the problem setting and
then describe how one may use :code:`qustop` to calculate the optimal
probability with which this problem can be solved when given access to
certain measurements.

Further information beyond the scope of this tutorial can be found in the
text [WatrousQI]_ as well as the course [SikoraSDP]_.

The state distinguishability problem
-------------------------------------

The quantum state distinguishability problem is phrased as follows.

1. Alice possesses an ensemble of :math:`n` quantum states:

    .. math::
        \begin{equation}
            \eta = \left( (p_0, \rho_0), \ldots, (p_n, \rho_n)  \right),
        \end{equation}

where :math:`p_i` is the probability with which state :math:`\rho_i` is
selected from the ensemble. Alice picks :math:`\rho_i` with probability
:math:`p_i` from her ensemble and sends :math:`\rho_i` to Bob.

2. Bob receives :math:`\rho_i`. Both Alice and Bob are aware of how the
   ensemble is defined but he does *not* know what index :math:`i`
   corresponding to the state :math:`\rho_i` he receives from Alice is.

3. Bob wants to guess which of the states from the ensemble he was given. In
   order to do so, he may measure :math:`\rho_i` to guess the index :math:`i`
   for which the state in the ensemble corresponds.

This setting is depicted in the following figure.

.. figure:: figures/quantum_state_distinguish.svg
   :alt: quantum state distinguishability
   :align: center

   The quantum state distinguishability setting.

Distinguishability Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Minimum-error discrimination

* Unambiguous discrimination

Distinguishability Measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Depending on the sets of measurements that Alice and Bob are allowed to use,
the optimal probability of distinguishing a given set of states is characterized
by the following image.

.. figure:: figures/measurement_inclusions.svg
   :width: 200
   :alt: measurement hierarchy
   :align: center

   Measurement hierarchy.

That is, the probability that Alice and Bob are able to distinguish using PPT
measurements is a natural upper bound on the optimal probability of
distinguishing via separable measurements and so on.

In general:

* LOCC: These are difficult objects to handle mathematically; difficult to
  design protocols for and difficult to provide bounds on their power.

* `Separable <https://qustop.readthedocs.io/en/latest/tutorials.separable.html>`_:
  Separable measurements have a nicer structure than LOCC.  Unfortunately,
  optimizing over separable measurements in NP-hard.

* `PPT <https://qustop.readthedocs.io/en/latest/tutorials.ppt.html>`_:
  PPT measurements offer a nice structure and there exists efficient techniques
  that allow one to optimize over the set of PPT measurements via semidefinite
  programming.

* `Positive <https://qustop.readthedocs.io/en/latest/tutorials.positive.html>`_:
  These measurements are the most general and constitute the set of all valid
  quantum operations that Alice and Bob can perform. The optimal value of
  distinguishing via positive operations can be phrased as an SDP.


Optimal probability of distinguishing a quantum state
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The optimal probability of distinguishing using positive measurements serves
as an upper bound on the optimal probability of distinguishing using PPT,
separable, and LOCC measurements. Specifically, given an ensemble of quantum
states, :math:`\eta`, it holds that

.. math::
    \text{opt}_{\text{LOCC}}(\eta) \leq
    \text{opt}_{\text{SEP}}(\eta) \leq
    \text{opt}_{\text{PPT}}(\eta) \leq
    \text{opt}_{\text{pos}}(\eta),

where:

- :math:`\text{opt}_{\text{pos}}(\eta)` represents the optimal probability of distinguishing using
  positive measurements,

-   :math:`\text{opt}_{\text{PPT}}(\eta)` represents the probability of distinguishing via PPT
    measurements,

-   :math:`\text{opt}_{\text{SEP}}(\eta)` represents the probability of distinguishing via
    separable measurements,

-   :math:`\text{opt}_{\text{LOCC}}(\eta)` represents the probability of distinguishing via LOCC
    measurements.

References
------------------------------
.. [WatrousQI] Watrous, John
    "The theory of quantum information"
    Section: "A semidefinite program for optimal measurements"
    Cambridge University Press, 2018

.. [SikoraSDP] Sikora, Jamie
    "Semidefinite programming in quantum theory (lecture series)"
    Lecture 2: Semidefinite programs for nice problems and popular functions
    Perimeter Institute for Theoretical Physics, 2019
