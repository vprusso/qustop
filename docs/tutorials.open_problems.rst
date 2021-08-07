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

The anti-distinguishability conjecture
---------------------------------------
