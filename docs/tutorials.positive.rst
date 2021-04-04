Distinguishing Quantum States via Positive Measurements
=======================================================

In this tutorial, we are going to show how to make use of :code:`qustop` to calculate the optimal
probability of distinguishing a state from an ensemble of quantum states when Alice and Bob are
allowed to use global (positive) measurements on their system.

Minimum-error distinguishability via positive measurements
----------------------------------------------------------

The optimal probability with which Bob can distinguish the state he is given
may be obtained by solving the following semidefinite program (SDP).

.. math::
    \begin{align*}
        \textbf{Primal:} \quad & \\
        \text{maximize:} \quad & \sum_{i=0}^n p_i \langle M_i,
        \rho_i \rangle \\
        \text{subject to:} \quad & \sum_{i=0}^n M_i = \mathbb{I}_{\mathcal{A} \otimes
                                   \mathcal{B}},\\
                                 & M_1, \ldots, M_n \in \text{Pos}(\mathcal{A} \otimes \mathcal{B}).
    \end{align*}

.. math::
    \begin{equation}
        \begin{aligned}
            \textbf{Dual:} \quad & \\
            \text{minimize:} \quad & \text{Tr}(Y) \\
            \text{subject to:} \quad & Y - \rho_i \in \text{Pos}(\mathcal{A} \otimes \mathcal{B}),
                                        \quad \forall i = 1, \ldots, n, \\
                                     & Y \in \text{Herm}(\mathcal{A} \otimes
                                        \mathcal{B}).
        \end{aligned}
    \end{equation}

The :code:`qustop` package solves either of these two optimization problems depending on whether the
optimal measurements are required.

For the special case of distinguishing between two states, the probability of optimally
distinguishing is exactly

.. math::
    \text{opt}_{\text{pos}}(\eta) = \frac{1}{2} + \frac{1}{4} \left\lVert \eta(0) - \eta(1) \right\rVert_1

where :math:`\left\lVert \cdot \right\rVert_1`.

A result of [tWalgate00]_ shows that any two orthogonal pure states can be distinguished perfectly.
This result actually applies to LOCC measurements and is a stronger claim than just for positive
measurements, but since :math:`\text{opt}_{\text{LOCC}} \leq \text{opt}_{\text{pos}}` it also
holds for positive measurements.

For example, consider the two orthogonal pure states

.. math::
    | \psi_0 \rangle = \sqrt{\frac{3}{4}} | + \rangle + \sqrt{\frac{1}{4}} | - \rangle,
    \quad \text{and} \quad
    | \psi_1 \rangle = \sqrt{\frac{1}{4}} | + \rangle + \sqrt{\frac{3}{4}} | - \rangle.

Since :math:`| \psi_0 \rangle` and :math:`| \psi_1 \rangle` are pure and orthogonal with each
other, they are able to be perfectly distinguished.

.. literalinclude:: ../examples/opt_dist/positive/min_error/two_pure_states.py
   :language: python
   :linenos:
   :start-after: # along with this program.  If not, see <https://www.gnu.org/licenses/>.

Consider now the following two mixed states

.. math::
    | \phi_1 \rangle = \frac{3}{4} |+ \rangle \langle + | + \frac{1}{4} |- \rangle \langle - |
    \quad \text{and} \quad
    | \phi_2 \rangle = \frac{1}{4} |+ \rangle \langle + | - \frac{3}{4} |- \rangle \langle - |

The following code sample shows that the closed-form equation matches the result obtained from
:code:`qustop`, however, since they are mixed states and not pure, we are not able to perfectly
distinguish them.

.. literalinclude:: ../examples/opt_dist/positive/min_error/two_mixed_states.py
   :language: python
   :linenos:
   :start-after: # along with this program.  If not, see <https://www.gnu.org/licenses/>.

On the note of orthogonality, if the ensemble of states provided consist of all mutually
orthogonal states, then it is possible to distinguish with perfect probability in this special case.

As a prototypical example, consider the four Bell states

.. math::
    \begin{equation}
        \begin{aligned}
            | \psi_0 \rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}, &\quad
            | \psi_1 \rangle = \frac{|01\rangle + |10\rangle}{\sqrt{2}}, \\
            | \psi_2 \rangle = \frac{|01\rangle - |10\rangle}{\sqrt{2}}, &\quad
            | \psi_3 \rangle = \frac{|00\rangle - |11\rangle}{\sqrt{2}}.
        \end{aligned}
    \end{equation}

.. literalinclude:: ../examples/opt_dist/positive/min_error/bell_states.py
   :language: python
   :linenos:
   :start-after: # along with this program.  If not, see <https://www.gnu.org/licenses/>.

If there are more than two states and those states are not mutually orthogonal, no closed-form
equation is known to exist, so we resort to solving the SDP.

EXAMPLE

Unambiguous distinguishability via positive measurements
---------------------------------------------------------

The optimal probability with which Bob can distinguish the state he is given unambiguously may be
obtained by solving the following semidefinite program (SDP).

.. math::
    \begin{align*}
        \textbf{Primal:} \quad & \\
        \text{maximize:} \quad & \sum_{i=0}^n p_i \langle M_i,
        \rho_i \rangle \\
        \text{subject to:} \quad & \sum_{i=0}^{n+1} M_i = \mathbb{I}_{\mathcal{A} \otimes
                                   \mathcal{B}},\\
                                 & \langle M_i \rho_j \rangle = 0 \quad
                                    \forall i \not= j = 1, \ldots, n \\
                                 & M_1, \ldots, M_n \in \text{Pos}(\mathcal{A} \otimes \mathcal{B}).
    \end{align*}

References
------------------------------
.. [tWalgate00] Walgate, J., Short, A. J., Hardy, L., & Vedral, V
    "Local distinguishability of multipartite orthogonal quantum states."
    Physical Review Letters 85.23 (2000): 4972.
