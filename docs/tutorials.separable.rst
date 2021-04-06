Distinguishing quantum states via separable measurements
========================================================

As previously mentioned, optimizing over the set of separable measurements is
NP-hard. However, there does exist a hierarchy of semidefinite programs which
eventually does converge to the separable value. This hierarchy is based off
the notion of symmetric extensions. More information about this hierarchy of
SDPs can be found here [Nav08]_.

Minimum-error distinguishability via separable measurements
-----------------------------------------------------------

.. math::

    \begin{equation}
        \begin{aligned}
            \textbf{Primal:} \quad & \\
            \text{maximize:} \quad & \sum_{k=1}^N p_k \langle \rho_k, \mu(k) \rangle, \\
            \text{subject to:} \quad & \sum_{k=1}^N \mu(k) =
                                       \mathbb{I}_{\mathcal{X} \otimes \mathcal{Y}}, \\
                                    & \text{Tr}_{\mathcal{Y}_2 \otimes \ldots \otimes
                                      \mathcal{Y}_s}(X_k) = \mu(k), \\
                                    & \left( \mathbb{I}_{\mathcal{X}} \otimes
                                      \Pi_{\mathcal{Y} \ovee \mathcal{Y}_2 \ovee \ldots \ovee
                                      \mathcal{Y}_s} \right) X_k
                                      \left(\mathbb{I}_{\mathcal{X}} \otimes
                                      \Pi_{\mathcal{Y} \ovee \mathcal{Y}_2 \ovee \ldots \ovee
                                      \mathcal{Y}_s} \right)
                                      = X_k \\
                                    & \text{T}_{\mathcal{X}}(X_k) \in \text{Pos}\left(
                                        \mathcal{X} \otimes \mathcal{Y} \otimes \mathcal{Y}_2
                                        \otimes \ldots \otimes \mathcal{Y}_s \right), \\
                                    & \text{T}_{\mathcal{Y}_2 \otimes \ldots \otimes
                                        \mathcal{Y}_s}(X_k) \in \text{Pos}\left(
                                        \mathcal{X} \otimes \mathcal{Y} \otimes \mathcal{Y}_2
                                        \otimes \ldots \otimes \mathcal{Y}_s \right), \\
                                    & X_1, \ldots, X_N \in
                                      \text{Pos}\left(\mathcal{X} \otimes \mathcal{Y} \otimes
                                      \mathcal{Y}_2 \otimes \ldots \otimes \mathcal{Y}_s
                                      \right).
        \end{aligned}
    \end{equation}

Entanglement cost of distinguishing Bell states
-----------------------------------------------

One may ask whether the ability to distinguish a state can be improved by
making use of an auxiliary resource state.

.. math::
    \begin{equation}
        | \tau_{\epsilon} \rangle = \sqrt{\frac{1+\epsilon}{2}} | 00 \rangle +
        \sqrt{\frac{1-\epsilon}{2}} | 11 \rangle,
    \end{equation}

for some :math:`\epsilon \in [0,1]`.


Distinguishing four Bell states
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It was shown in [BCJRWY15]_ that the probability of distinguishing four Bell
states with a resource state via separable measurements is given by the
closed-form expression:

.. math::
    \begin{equation}
        \text{opt}_{\text{PPT}}(\eta) = 
        \text{opt}_{\text{SEP}}(\eta) =
        \frac{1}{2} \left(1 + \sqrt{1 - \epsilon^2} \right)
    \end{equation}

where the ensemble is defined as 

.. math::
    \begin{equation}
        \eta = \left\{ 
            | \psi_0 \rangle \otimes | \tau_{\epsilon} \rangle, 
            | \psi_1 \rangle \otimes | \tau_{\epsilon} \rangle, 
            | \psi_2 \rangle \otimes | \tau_{\epsilon} \rangle, 
            | \psi_3 \rangle \otimes | \tau_{\epsilon} \rangle
         \right\}.
    \end{equation}

Using :code:`qustop`, we may encode this scenario as follows.

.. literalinclude:: ../examples/opt_dist/ppt/min_error/entanglement_cost_four_bell_states.py
   :language: python
   :linenos:
   :start-after: # along with this program.  If not, see <https://www.gnu.org/licenses/>.

Note that [BCJRWY15]_ also proved the same closed-form expression for when
Alice and Bob make use of PPT measurements (which is an upper bound for
separable measurements). More on that in the
`tutorial on distinguishing via PPT measurements <https://qustop.readthedocs.io/en/latest/tutorials.ppt.html>`_.

Distinguishing three Bell states
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(TODO)

References
-----------
.. [Nav08] Navascués, Miguel.
    "Pure state estimation and the characterization of entanglement."
    Physical review letters 100.7 (2008): 070503.
    https://arxiv.org/abs/0707.4398
