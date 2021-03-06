Open problems in quantum state optimization
===========================================

This page consists of certain problems in the domain of quantum state
optimization that may be framed as computational tasks using the :code:`qustop`
module. Most of these examples are provided as brute-force computational
approaches. It may be possible to refine many of these approaches in a more
sophisticated manner, and I welcome any such input and feedback from the
community.

Two-copy problem
-----------------

Problem statement:
^^^^^^^^^^^^^^^^^^

Let :math:`n \geq 1` be an integer, let :math:`\mathcal{X}` be a compelx
Euclidean space, let :math:`\rho_i \in \text{D}(\mathcal{X})` be a pure quantum
state represented as a density operator and let

.. math::
    \eta = \left\{\left(\frac{1}{n}, \rho_1\right), 
                  \ldots, 
                  \left(\frac{1}{n}, \rho_n\right) \right\}
   \subset \mathcal{X}

be an ensemble of pure and mutually orthogonal quantum states. Define
:math:`\eta^{\otimes 2}` as the two-copy ensemble where

.. math::
    \eta^{\otimes 2} = \left\{\left(\frac{1}{n}, \rho_1 \otimes \rho_1\right), 
                             \ldots, 
                             \left(\frac{1}{n}, \rho_n \otimes \rho_n\right) \right\} 
   \subset \mathcal{X} \otimes\mathcal{X}.

**Question**: Does there exist a certain ensemble :math:`\eta^{\otimes 2}` such
that

.. math::
    \text{opt}_{\text{PPT}}\left(\eta^{\otimes 2}\right) < 1 \quad \text{or} \quad
   \text{opt}_{\text{SEP}}\left(\eta^{\otimes 2}\right) < 1?

Computational approach:
^^^^^^^^^^^^^^^^^^^^^^^

The computational approach here is to generate a random ensemble of quantum
states and then construct two copies of that ensemble. We take the two-copy
ensemble and check whether or not we get a value less than :math:`1` for
distinguishing via PPT measurements.


.. literalinclude:: ../research/two_copy_problem.py
   :language: python
   :linenos:
   :start-after: # along with this program.  If not, see <https://www.gnu.org/licenses/>.

One can specify the :code:`num_states` parameter to determine how many states
are in the ensemble and can also modify the :code:`num_trials` parameter to
specify for how many random ensembles should be generated and checked.

For instance, here we can randomly generate a 4-state two-copy ensemble 5
different times:

.. code-block:: python

   >>> run(num_states=4, num_trials=5)
   PPT 2-copy: 0.9999999997183334
   PPT 2-copy: 1.0000000011197347
   PPT 2-copy: 1.000000005640175
   PPT 2-copy: 1.0000000003272558
   PPT 2-copy: 0.9999999981951567


Entanglement cost of distinguishing four arbitrary two-qubit ensemble with resource state
-----------------------------------------------------------------------------------------

Problem statement:
^^^^^^^^^^^^^^^^^^

Let :math:`\mathbb{C}^2 = \mathcal{X}_1 = \mathcal{X}_2 = \mathcal{Y}_1 =
\mathcal{Y}_2` be complex Euclidean spaces and let :math:`\mathcal{X} =
\mathcal{X}_1 \otimes \mathcal{X}_2` and :math:`\mathcal{Y} = \mathcal{Y}_1
\otimes \mathcal{Y}_2`. Define the following orthogonal two-qubit basis

.. math::
    \begin{equation}
        \begin{aligned}
            | \psi_0 \rangle = \alpha | 00 \rangle + \beta | 11 \rangle, \quad
            | \psi_1 \rangle = \beta | 00 \rangle - \alpha | 11 \rangle, \\
            | \psi_2 \rangle = \alpha | 01 \rangle + \beta | 10 \rangle, \quad
            | \psi_3 \rangle = \beta | 01 \rangle - \alpha | 10 \rangle,
        \end{aligned}
    \end{equation}

where :math:`| \psi_i \rangle \in \mathcal{X}_1 \otimes \mathcal{Y}_1` for all
:math:`i \in [0,1,2,3]` and where

.. math::
    \alpha = \sqrt{\frac{1+n}{2}}
    \quad \text{and} \quad
    \beta = \sqrt{\frac{1-n}{2}}.

Define the resource state

.. math::
    | \tau_{\epsilon} \rangle = \sqrt{\frac{1+\epsilon}{2}} | 00 \rangle +
                                \sqrt{\frac{1-\epsilon}{2}} | 11 \rangle
    \in \mathcal{X}_2 \otimes \mathcal{Y}_2

for some choice of :math:`\epsilon \in [0,1]`.

Consider the ensemble

.. math::
    \eta = \left\{
    | \psi_0 \rangle \otimes | \tau_{\epsilon} \rangle,
    | \psi_1 \rangle \otimes | \tau_{\epsilon} \rangle,
    | \psi_2 \rangle \otimes | \tau_{\epsilon} \rangle,
    | \psi_3 \rangle \otimes | \tau_{\epsilon} \rangle
    \right\}
    \subset \mathcal{X} \otimes \mathcal{Y}.

**Question**: Assuming a uniform distribution :math:`p_1 = p_2 = p_3 = p_4 =
1/4`, with any state from :math:`\eta` being selected, what is the closed-form
entanglement cost of distinguishing :math:`\eta` via PPT measurements for any
choice of :math:`\alpha` and :math:`\beta`?

.. note::

    When :math:`\alpha = \beta = \frac{1}{\sqrt{2}}`, the ensemble :math:`\eta`
    consists of the Bell states. In this case, it is known that the closed-form
    entanglement cost is

    .. math::
        \frac{1}{2} \left(1 + \sqrt{1-\epsilon^2}\right).

**Follow-up Question**: Assuming a uniform distribution :math:`p_1 = p_2 = p_3 = p_4 =
1/4`, with any state from :math:`\eta` being selected, what is the closed-form
entanglement cost of distinguishing :math:`\eta` via *separable* measurements for any
choice of :math:`\alpha` and :math:`\beta`?

This question will most likely be *more* difficult than the PPT case, but it's
still worth asking.

Computational approach:
^^^^^^^^^^^^^^^^^^^^^^^

Consider the following snippet of code that considers this problem:

.. literalinclude:: ../research/entanglement_cost_four_state_two_qubit.py
   :language: python
   :linenos:
   :start-after: # along with this program.  If not, see <https://www.gnu.org/licenses/>.

Is it possible to determine a closed form of the optimal value of distinguishing
via PPT measurements for any choice of :code:`n` and :code:`eps`? It may be
illuminating to alter these values and see if any obvious closed-form
expressions emerge.

It is also possible to alter the :code:`ppt` argument to :code:`sep` to get a
sense on the follow-up question. However, keep in mind that the computational
technique used to compute separable values is an upper bound--a hierarchy of
SDPs that will eventually converge, but may take substantial computational
resources and time to do so.

Entanglement cost of distinguishing three arbitrary two-qubit ensemble with resource state
------------------------------------------------------------------------------------------

Problem statement:
^^^^^^^^^^^^^^^^^^

Let :math:`\mathbb{C}^2 = \mathcal{X}_1 = \mathcal{X}_2 = \mathcal{Y}_1 =
\mathcal{Y}_2` be complex Euclidean spaces and let :math:`\mathcal{X} =
\mathcal{X}_1 \otimes \mathcal{X}_2` and :math:`\mathcal{Y} = \mathcal{Y}_1
\otimes \mathcal{Y}_2`. Define the following orthogonal two-qubit basis

.. math::
    \begin{equation}
        \begin{aligned}
            | \psi_0 \rangle = \alpha | 00 \rangle + \beta | 11 \rangle, \quad
            | \psi_1 \rangle = \beta | 00 \rangle - \alpha | 11 \rangle, \\
            | \psi_2 \rangle = \alpha | 01 \rangle + \beta | 10 \rangle, \quad
        \end{aligned}
    \end{equation}

where :math:`| \psi_i \rangle \in \mathcal{X}_1 \otimes \mathcal{Y}_1` for all
:math:`i \in [0,1,2]` and where

.. math::
    \alpha = \sqrt{\frac{1+n}{2}}
    \quad \text{and} \quad
    \beta = \sqrt{\frac{1-n}{2}}.

Define the resource state

.. math::
    | \tau_{\epsilon} \rangle = \sqrt{\frac{1+\epsilon}{2}} | 00 \rangle +
                                \sqrt{\frac{1-\epsilon}{2}} | 11 \rangle
    \in \mathcal{X}_2 \otimes \mathcal{Y}_2

for some choice of :math:`\epsilon \in [0,1]`.

Consider the ensemble

.. math::
    \eta = \left\{
    | \psi_0 \rangle \otimes | \tau_{\epsilon} \rangle, \quad
    | \psi_1 \rangle \otimes | \tau_{\epsilon} \rangle, \\
    | \psi_2 \rangle \otimes | \tau_{\epsilon} \rangle,
    \right\}
    \subset \mathcal{X} \otimes \mathcal{Y}.

**Question**: Assuming a uniform distribution :math:`p_1 = p_2 = p_3 =
1/3`, with any state from :math:`\eta` being selected, what is the closed-form
entanglement cost of distinguishing :math:`\eta` via separable measurements for any
choice of :math:`\alpha` and :math:`\beta`?

.. note::

    When :math:`\alpha = \beta = \frac{1}{\sqrt{2}}`, the ensemble :math:`\eta`
    consists of three out of the four the Bell states. In this case, it is known
    that the closed-form entanglement cost for separable measurements is

    .. math::
        \frac{1}{3} \left(2 + \sqrt{1-\epsilon^2}\right).

A simpler version of this question is to determine
:math:`\text{opt}_{\text{PPT}}(\eta)` for :math:`n=0` for any :math:`\epsilon`.

Since :math:`\text{opt}_{\text{SEP}}(\eta) < \text{opt}_{\text{PPT}}(\eta)`, we
have a lower bound on :math:`\text{opt}_{\text{PPT}}(\eta)` when :math:`n=0`. It
also holds that :math:`\text{opt}_{\text{POS}}(\eta) = 1` for all :math:`n` and
:math:`\epsilon`, so we have an obvious upper bound.

Computational approach:
^^^^^^^^^^^^^^^^^^^^^^^

Consider the following snippet of code that considers this problem:

.. literalinclude:: ../research/entanglement_cost_three_state_two_qubit.py
   :language: python
   :linenos:
   :start-after: # along with this program.  If not, see <https://www.gnu.org/licenses/>.

Separation between LOCC and separable value
-------------------------------------------

Problem statement:
^^^^^^^^^^^^^^^^^^

Does there exist an ensemble, :math:`\eta`, fully composed of orthogonal
maximally entangled states for which the following strict inequality holds:

.. math::
    \text{opt}_{\text{LOCC}}(\eta) < \text{opt}_{\text{SEP}}(\eta)?


The mathematical structure of LOCC is difficult to generalize, but it is
possible to computationally analyze the separable case via a hierarchy of
semidefinite programs.

The antidistinguishability conjecture
---------------------------------------

A set of :math:`n` pure quantum states :math:`| \rho_1 \rangle, \ldots, |
\rho_n \rangle` are *antidistinguishable* if there exists an :math:`n`-outcome
measurement that never outputs the result :math:`i` on :math:`| \rho_i
\rangle`.

Problem statement:
^^^^^^^^^^^^^^^^^^

Let :math:`| \rho_1 \rangle, \ldots, | \rho_d \rangle` be :math:`d` pure quantum
states each of dimension :math:`d`. If 

.. math:: 
   | \langle \rho_i | \rho_j\rangle| \leq (d-2)/(d-1) 
   
for all :math:`i \not= j`, then the states are antidistinguishable.

**Question**: Does this conjecture hold for all integers :math:`d`?

.. note::
    This conjecture is known to hold when :math:`d=2` and :math:`d=3`.

Computational approach:
^^^^^^^^^^^^^^^^^^^^^^^

We can randomly generate an ensemble of :math:`d` pure states of dimension
:math:`d` using the following function.

X

Next, we check whether the states are antidistinguishable. This is equivalent to
checking whether the optimal value of the semidefinite program for the problem
for the problem of unambiguous quantum state exclusion from `arXiv:1306.4683
<https://arxiv.org/pdf/1306.4683.pdf>`_ yields a non-zero result.

If the states *are* antidistinguishable, check the bound in the conjecture:

    - If the bound is satisfied it implies a satisfaction of the conjecture, so
      we go back to the beginning and try to generate another random ensemble.

    - If the bound is *not* satisfied, this implies a violation of the
      conjecture.
