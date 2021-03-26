Distinguishing Quantum States via Positive Measurements
=======================================================

In this tutorial, we are going to show how ...

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

The `qustop` package solves either of these two optimization problems depending on whether the
optimal measurements are required.

For the special case of distinguishing between two states, the probability of optimally
distinguishing is exactly

.. math::
    \text{opt}_{\text{pos}}(\eta) = \frac{1}{2} + \frac{1}{4} \left\lVert \eta(0) - \eta(1) \right\rVert_1

where :math:`\left\lVert \cdot \right\rVert_1`.

For example, consider the following two states:

.. math::
    v_1 = \frac{\sqrt{3}}{2} | 00 \rangle + \frac{1}{2} | 11 \rangle
    \quad \text{and} \quad
    v_2 = \frac{1}{2} |00 \rangle + \frac{\sqrt{3}}{2} | 11 \rangle.

The following code sample shows that the closed-form equation matches the result obtained from
:code:`qustop`.

.. literalinclude:: ../examples/opt_dist/positive/min_error_two_states.py
   :language: python
   :linenos:
   :start-after: # along with this program.  If not, see <https://www.gnu.org/licenses/>.

If the ensemble of states provided consist of all mutually orthogonal states, then it is possible
to distinguish with perfect probability in this case.

EXAMPLE

If there are more than two states and those states are not mutually orthogonal, no closed-form
equation is known to exist, so we resort to solving the SDP.

EXAMPLE

Unambiguous distinguishability via positive measurements
---------------------------------------------------------

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
