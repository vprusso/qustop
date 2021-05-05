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
                                    & \text{T}_{\mathcal{Y}_2} (X_k) \in \text{Pos}\left(
                                        \mathcal{X} \otimes \mathcal{Y} \otimes \mathcal{Y}_2
                                        \otimes \ldots \otimes \mathcal{Y}_s \right), \\
                                    & \vdots \\
                                    & \text{T}_{\mathcal{Y}_s} (X_k) \in \text{Pos}\left(
                                        \mathcal{X} \otimes \mathcal{Y} \otimes \mathcal{Y}_2
                                        \otimes \ldots \otimes \mathcal{Y}_s \right), \\
                                    & X_1, \ldots, X_N \in
                                      \text{Pos}\left(\mathcal{X} \otimes \mathcal{Y} \otimes
                                      \mathcal{Y}_2 \otimes \ldots \otimes \mathcal{Y}_s
                                      \right).
        \end{aligned}
    \end{equation}

.. math::

    \begin{equation}
        \begin{aligned}
            \textbf{Dual:} \quad & \\
            \text{minimize:} \quad & \text{Tr}(H), \\
            \text{subject to:} \quad & H - Q_k \geq p_k \rho_k, \\
                                    & Q_k \otimes \mathbb{I}_{\mathcal{Y}_2
                                    \otimes \ldots \otimes \mathcal{Y}_s} +
                                    \left(\mathbb{I}_{\mathcal{X}} \otimes
                                    \Pi_{\mathcal{Y} \ovee \mathcal{Y}_2 \ldots
                                    \ovee \mathcal{Y_s}} \right) R_k
                                    \left(\mathbb{I}_{\mathcal{X}} \otimes
                                    \Pi_{\mathcal{Y} \ovee \mathcal{Y}_2 \ldots
                                    \ovee \mathcal{Y_s}} \right) \\
                                    & \quad - R_k - \text{T}_{\mathcal{X}}(S_k)
                                    - \text{T}_{\mathcal{Y}}(Z_k) \in
                                      \text{Pos}(\mathcal{X} \otimes
                                      \mathcal{Y} \otimes \mathcal{Y}_2 \otimes
                                      \ldots \otimes \mathcal{Y}_s) \\
                                    & H, Q_1, \ldots, Q_N \in
                                    \text{Herm}(\mathcal{X} \otimes
                                    \mathcal{Y}), \\
                                    & R_1, \ldots R_N \in
                                    \text{Herm}(\mathcal{X} \otimes \mathcal{Y}
                                    \otimes \mathcal{Y}_2 \otimes \ldots
                                    \otimes \mathcal{Y}_s), \\ & S_1, \ldots,
                                    S_N, Z_1, \ldots, Z_N \in
                                    \text{Pos}(\mathcal{X} \otimes \mathcal{Y}
                                    \otimes \mathcal{Y}_2 \otimes \ldots
                                    \otimes \mathcal{Y}_s).
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
It was also shown in [BCJRWY15]_ that the closed-form probability of distinguishing three Bell states with a resource
state using separable measurements to be given by the closed-form expression:

.. math::
    \begin{equation}
        \text{opt}_{\text{SEP}}(\eta) = \frac{1}{3} \left( 2 + \sqrt{1 - \epsilon^2} \right)
    \end{equation}

where the ensemble is defined as

.. math::
    \begin{equation}
        \eta = \left( | \psi_0 \rangle \otimes | \tau_{\epsilon} \rangle,
                      | \psi_1 \rangle \otimes | \tau_{\epsilon} \rangle,
                      | \psi_2 \rangle \otimes | \tau_{\epsilon} \rangle
        \right).
    \end{equation}

Using :code:`qustop`, we may encode this scenario as follows.

.. literalinclude:: ../examples/opt_dist/sep/min_error/entanglement_cost_three_bell_states.py
   :language: python
   :linenos:
   :start-after: # along with this program.  If not, see <https://www.gnu.org/licenses/>.

Note that the value of :code:`sep_res.value` is actually a bit higher than :code:`eq`. This is because the separable
value is calculated by a hierarchy of SDPs. At low levels of the SDP, the problem can often converge to the optimal
value, but other times it is necessary to compute higher levels of the SDP to eventually arrive at the optimal value.
While this is intractable in general, in practice, the SDP can often converge, or at least get fairly close to the
optimal value for small problem sizes.

Unextendible product bases and separable measurements
-----------------------------------------------------

For complex Euclidean spaces :math:`\mathcal{X}` and :math:`\mathcal{Y}`, an *unextendable product basis* is defined as
an orthonormal collection of vectors

.. math::
    \begin{equation}
        \mathcal{U} = \left\{u_1 \otimes v_1, \ldots, u_m \otimes v_m \right\}
        \subset \mathcal{X} \otimes \mathcal{Y},
    \end{equation}

for unit vectors :math:`u_1, \ldots, u_m \in \mathcal{X}` and :math:`v_1, \ldots, v_m \in \mathcal{Y}` where:

1. :math:`m < \text{dim}(\mathcal{X} \otimes \mathcal{Y})`.

2. For every :math:`x \in \mathcal{X}` and :math:`y \in \mathcal{Y}` satisfying :math:`x \otimes y \perp \mathcal{U}`
it holds that :math:`x \otimes y = 0`.

All UPBs are known to be indistinguishable by LOCC measurements and all UPBs are distinguishable by PPT measurements.
As separable measurements lie in between LOCC and PPT measurements, it is of interest to know which UPBs are
distinguishable by separable measurements.

For instance, in [DMSST99]_, it was shown that all UPBs in :math:`\mathbb{C}^3 \otimes \mathbb{C}^3` are
perfectly distinguishable via separable measurements.

Consider the "Tiles" UPB

.. math::
    \begin{equation}
      \begin{array}{llll}
        | \phi_0 \rangle = | 0 \rangle \left(\frac{| 0 \rangle - | 1 \rangle}{\sqrt{2}}\right),
        &| \phi_1 \rangle = | 2 \rangle\left(\frac{| 1 \rangle - | 2 \rangle }{\sqrt{2}}\right), \\
        | \phi_2 \rangle = \left(\frac{| 0 \rangle - | 1 \rangle}{\sqrt{2}}\right)| 2 \rangle,
        &| \phi_3 \rangle = \left(\frac{| 1 \rangle - | 2 \rangle}{\sqrt{2}}\right)| 0 \rangle,\\
        | \phi_4 \rangle = \frac{1}{3}\left(| 0 \rangle + | 1 \rangle + | 2 \rangle\right),
                  \left(| 0 \rangle + | 1 \rangle + | 2 \rangle \right).
      \end{array}
    \end{equation}

Note that the "Tiles" states are contained in :math:`\mathbb{C}^3 \otimes \mathbb{C}^3`. We can use :code:`qustop` to
indeed verify that these states are perfectly distinguishable via separable measurements.

.. literalinclude:: ../examples/opt_dist/sep/min_error/tile_upb.py
   :language: python
   :linenos:
   :start-after: # along with this program.  If not, see <https://www.gnu.org/licenses/>.

In [BCJRWY15]_, it was shown that the 8-state UPB contained in :math:`\mathbb{C}^4 \otimes \mathbb{C}^4` introduced in
[Feng06]_ defined as

.. math::
    \begin{equation}
      \begin{array}{ll}
        | \phi_1 \rangle = | 0 \rangle | 0 \rangle, &
        | \phi_5 \rangle = \left(| 1 \rangle + | 2 \rangle + | 3 \rangle \right)\left(| 0 \rangle
        - | 1 \rangle + | 2 \rangle \right)/3, \\
        | \phi_2 \rangle = | 1 \rangle \left(| 0 \rangle - | 2 \rangle + | 3 \rangle \right)/\sqrt{3},
        \quad &
        | \phi_6 \rangle = \left(| 0 \rangle - | 2 \rangle + | 3 \rangle \right)| 2 \rangle/\sqrt{3}, \\
        | \phi_3 \rangle = | 2 \rangle \left(| 0 \rangle + | 1 \rangle - | 3 \rangle \right)/\sqrt{3}, &
        | \phi_7 \rangle = \left(| 0 \rangle + | 1 \rangle - | 3 \rangle \right)| 1 \rangle/\sqrt{3}, \\
        | \phi_4 \rangle = | 3 \rangle | 3 \rangle, &
        | \phi_8 \rangle = \left(| 0 \rangle - | 1 \rangle + | 2 \rangle \right)\left(| 1 \rangle
        + | 2 \rangle + | 3 \rangle \right)/3.
      \end{array}
    \end{equation}

is not perfectly distinguishable via separable measurements. This can be observed using :code:`qustop` as follows.

.. literalinclude:: ../examples/opt_dist/sep/min_error/feng_upb.py
   :language: python
   :linenos:
   :start-after: # along with this program.  If not, see <https://www.gnu.org/licenses/>.

Impossibility to distinguish a UPB plus one extra pure state
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It was shown in [BCJRWY15]_ that if we consider an ensemble of states consisting of a UPB along with a pure state
that is orthogonal to all states in said ensemble, then it is impossible to perfectly distinguish this ensemble.

.. math::
    \begin{equation}
      \begin{array}{llll}
        | \phi_0 \rangle = | 0 \rangle \left(\frac{| 0 \rangle - | 1 \rangle}{\sqrt{2}}\right),
        &| \phi_1 \rangle = | 2 \rangle\left(\frac{| 1 \rangle - | 2 \rangle }{\sqrt{2}}\right), \\
        | \phi_2 \rangle = \left(\frac{| 0 \rangle - | 1 \rangle}{\sqrt{2}}\right)| 2 \rangle,
        &| \phi_3 \rangle = \left(\frac{| 1 \rangle - | 2 \rangle}{\sqrt{2}}\right)| 0 \rangle,\\
        | \phi_4 \rangle = \frac{1}{3}\left(| 0 \rangle + | 1 \rangle + | 2 \rangle\right)
                  \left(| 0 \rangle + | 1 \rangle + | 2 \rangle \right),
        & | \phi_5 \rangle = \frac{1}{2} \left( | 0 \rangle | 0 \rangle + | 0 \rangle | 1 \rangle - 
                             | 0 \rangle | 2 \rangle - | 1 \rangle | 2 \rangle \right)
      \end{array}
    \end{equation}

.. literalinclude:: ../examples/opt_dist/sep/min_error/tile_upb_plus_pure.py
   :language: python
   :linenos:
   :start-after: # along with this program.  If not, see <https://www.gnu.org/licenses/>.

References
-----------
.. [Nav08] Navascués, Miguel.
    "Pure state estimation and the characterization of entanglement."
    Physical review letters 100.7 (2008): 070503.
    https://arxiv.org/abs/0707.4398

.. [DMSST99] DiVincenzo, David P., et al.
     "Unextendible product bases, uncompletable product bases and bound entanglement."
     Communications in Mathematical Physics 238.3 (2003): 379-410.

.. [Feng06] Feng, Keqin.
            "Unextendible product bases and 1-factorization of complete graphs."
            Discrete applied mathematics 154.6 (2006): 942-949.
