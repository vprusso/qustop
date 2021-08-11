Open problems in quantum state optimization
===========================================

This page consists of certain problems in the domain of quantum state optimization that may be framed as
computational tasks using the :code:`qustop` module. Most of these examples are provided as brute-force computational
approaches. It may be possible to refine many of these approaches in a more sophisticated manner, and I welcome any
such input and feedback from the community.

Two-copy problem
-----------------

Problem statement:
^^^^^^^^^^^^^^^^^^

Let :math:`n \geq 1` be an integer, let :math:`\mathcal{X}` be a compelx Euclidean space, let :math:`\rho_i \in
\text{D}(\mathcal{X})` be a pure quantum state represented as a density operator and let

.. math::
    \eta = \left\{\rho_1, \ldots, \rho_n \right\} \subset \mathcal{X}

be an ensemble of pure and mutually orthogonal quantum states. Define :math:`\eta^{\otimes 2}` as the two-copy ensemble
where

.. math::
    \eta^{\otimes 2} = \left\{\rho_1 \otimes \rho_1, \ldots, \rho_n \otimes \rho_n \right\} \subset \mathcal{X} \otimes\mathcal{X}

**Question**: Does there exist a certain ensemble :math:`\eta^{\otimes 2}` such that

.. math::
    \text{opt}_{\text{PPT}}(\eta^{\otimes 2}) < 1 \quad \text{or} \quad
    \text{opt}_{\text{SEP}}(\eta^{\otimes 2}) < 1?

Computational approach:
^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../research/two_copy_problem.py
   :language: python
   :linenos:
   :start-after: # along with this program.  If not, see <https://www.gnu.org/licenses/>.

Entanglement cost of distinguishing four arbitrary two-qubit ensemble
---------------------------------------------------------------------

Problem statement:
^^^^^^^^^^^^^^^^^^

Let :math:`\mathbb{C}^2 = \mathcal{X}_1 = \mathcal{X}_2 = \mathcal{Y}_1 = \mathcal{Y}_2` be complex Euclidean spaces
and let :math:`\mathcal{X} = \mathcal{X}_1 \otimes \mathcal{X}_2` and
:math:`\mathcal{Y} = \mathcal{Y}_1 \otimes \mathcal{Y}_2`. Define the following orthogonal two-qubit basis

.. math::
    \begin{equation}
        \begin{aligned}
            | \psi_0 \rangle = \alpha | 00 \rangle + \beta | 11 \rangle, \quad
            | \psi_1 \rangle = \beta | 00 \rangle - \alpha | 11 \rangle, \\
            | \psi_2 \rangle = \alpha | 01 \rangle + \beta | 10 \rangle, \quad
            | \psi_3 \rangle = \beta | 01 \rangle - \alpha | 10 \rangle,
        \end{aligned}
    \end{equation}

where :math:`| \psi_i \rangle \in \mathcal{X}_1 \otimes \mathcal{Y}_1` for all :math:`i \in [0,1,2,3]` and where

.. math::
    \alpha = \sqrt{\frac{1+n}{2}}
    \quad \text{and} \quad
    \beta = \sqrt{\frac{1-n}{2}}.

Define the state

.. math::
    | \tau_{\epsilon} \rangle = \sqrt{\frac{1+\epsilon}{2}} | 00 \rangle +
                                \sqrt{\frac{1-\epsilon}{2}} | 11 \rangle
    \in \mathcal{X_2} \otimes \mathcal{Y_2}

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

**Question**: Assuming a uniform distribution :math:`p_1 = p_2 = p_3 = p_4 = 1/4`, any state from :math:`\eta` being
selected, what is the closed-form entanglement cost of distinguishing :math:`\eta` via PPT measurements for any
choice of :math:`\alpha` and :math:`\beta`?

.. note::

    When :math:`\alpha = \beta = \frac{1}{\sqrt{2}}`, the ensemble :math:`\eta` is consists of the Bell states. In this
    case, it is known that the closed-form entanglement cost is

    .. math::
        \frac{1}{2} \left(1 + \sqrt{1-\epsilon^2}\right).



The anti-distinguishability conjecture
---------------------------------------

