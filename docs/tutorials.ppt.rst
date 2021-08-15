Distinguishing quantum states via PPT measurements
==================================================

In this section we will be investigation how to make use of the
:code:`qustop` package to optimally distinguish quantum states via PPT
measurements.

Minimum-error
-------------

In [Cosentino13]_, an semidefinite program formulation whose optimal value
corresponds to the optimal probability of distinguishing a quantum state from
an ensemble using PPT measurements with minimum error was provided. The
primal and dual problems of this SDP are defined as follows.

.. math::
    \begin{equation}
        \begin{aligned}
            \textbf{Primal:} \quad & \\
            \text{maximize:} \quad & \sum_{j=1}^k \langle P_j, \rho_j \rangle \\
            \text{subject to:} \quad & P_1 + \cdots + P_k = \mathbb{I}_{\mathcal{A}}
                                        \otimes \mathbb{I}_{\mathcal{B}}, \\
                                     & P_1, \ldots, P_k \in \text{PPT}(\mathcal{A} : \mathcal{B}).
        \end{aligned}
    \end{equation}

.. math::
    \begin{equation}
        \begin{aligned}
            \textbf{Dual:} \quad & \\
            \text{minimize:} \quad & \frac{1}{k} \text{Tr}(Y) \\
            \text{subject to:} \quad & Y - \rho_j \geq \text{T}_{\mathcal{A}} (Q_j),
                                        \quad j = 1, \ldots, k, \\
                                     & Y \in \text{Herm}(\mathcal{A} \otimes
                                        \mathcal{B}), \\
                                     & Q_1, \ldots, Q_k \in
                                        \text{Pos}(\mathcal{A} \otimes \mathcal{B}).
        \end{aligned}
    \end{equation}

Unambiguous
-----------

In [Cosentino13]_, an semidefinite program formulation whose optimal value
corresponds to the optimal probability of distinguishing a quantum state from
an ensemble using PPT measurements unambiguously was provided. The primal and dual problems
of this SDP are defined as follows.

.. math::
    \begin{equation}
        \begin{aligned}
            \textbf{Primal:} \quad & \\
            \text{maximize:} \quad & \sum_{j=1}^k \langle P_j, \rho_j \rangle \\
            \text{subject to:} \quad & P_1 + \cdots + P_k = \mathbb{I}_{\mathcal{A}}
                                        \otimes \mathbb{I}_{\mathcal{B}}, \\
                                     & P_1, \ldots, P_k
                                      \in \text{PPT}(\mathcal{A} : \mathcal{B}), \\
                                     & \langle P_i, \rho_j \rangle = 0,
                                       \quad 1 \leq i, j \leq k, \quad i \not= j.
        \end{aligned}
    \end{equation}

.. math::
    \begin{equation}
        \begin{aligned}
            \textbf{Dual:} \quad & \\
            \text{minimize:} \quad & \frac{1}{k} \text{Tr}(Y) \\
            \text{subject to:} \quad & Y - \rho_j \geq \text{T}_{\mathcal{A}} (Q_j),
                                        \quad j = 1, \ldots, k, \\
                                     & Y \in \text{Herm}(\mathcal{A} \otimes
                                        \mathcal{B}), \\
                                     & Q_1, \ldots, Q_k \in
                                        \text{Pos}(\mathcal{A} \otimes \mathcal{B}).
        \end{aligned}
    \end{equation}

Distinguishing four Bell states
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider the following Bell states:

.. math::
    \begin{equation}
        \begin{aligned}
            | \psi_0 \rangle = \frac{| 00 \rangle + | 11 \rangle}{\sqrt{2}}, \quad 
            | \psi_1 \rangle = \frac{| 01 \rangle + | 10 \rangle}{\sqrt{2}}, \\
            | \psi_2 \rangle = \frac{| 01 \rangle - | 10 \rangle}{\sqrt{2}}, \quad 
            | \psi_3 \rangle = \frac{| 00 \rangle - | 11 \rangle}{\sqrt{2}}.
        \end{aligned}
    \end{equation}

Assuming a uniform probability of selecting from any one of these states, that is, assuming we define an ensemble of
Bell states defined as

.. math::
    \begin{equation}
        \mathbb{B} = \left\{
                        \left(| \psi_0 \rangle, \frac{1}{4} \right),
                        \left(| \psi_1 \rangle, \frac{1}{4} \right),
                        \left(| \psi_2 \rangle, \frac{1}{4} \right),
                        \left(| \psi_3 \rangle, \frac{1}{4} \right)
                     \right\}
    \end{equation}

it holds that

.. math::
    \begin{equation}
        \text{opt}_{\text{PPT}}(\mathbb{B}) = \frac{1}{2}.
    \end{equation}

We can observe this using :code:`qustop` as follows.

.. literalinclude:: ../examples/opt_dist/ppt/min_error/four_bell_states.py
   :language: python
   :linenos:
   :start-after: # along with this program.  If not, see <https://www.gnu.org/licenses/>.

Indeed, a stronger statement is known to hold for :math:`\mathbb{B}`, that is

.. math::
    \begin{equation}
        \text{opt}_{\text{LOCC}}(\mathbb{B}) = \frac{1}{2}.
    \end{equation}

Recall that for any ensemble :math:`\eta`, it holds that :math:`\text{opt}_{\text{LOCC}}(\eta) <
\text{opt}_{\text{PPT}}(\eta)`.

Four indistinguishable orthogonal maximally entangled states
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In [YDY12]_ the following ensemble of states was shown not to be perfectly distinguishable by PPT
measurements, and therefore also indistinguishable via LOCC measurements.

.. math::
    \rho_0 = |\psi_0\rangle |\psi_0 \rangle \langle \psi_0 | \langle \psi_0 |, \quad
    \rho_1 = |\psi_1 \rangle |\psi_3 \rangle \langle \psi_1 | \langle \psi_3 |, \\
    \rho_2 = |\psi_3\rangle |\psi_1 \rangle \langle \psi_3 | \langle \psi_1 |, \quad
    \rho_3 = |\psi_1 \rangle |\psi_1 \rangle \langle \psi_1 | \langle \psi_1 |, \\

While it was known that perfect distinguishability could not be achieved, the actual value and
bound of optimal distinguishability was not known. It was shown in [Cosentino13]_ and later extended
in [CR13]_ that the optimal probability of distinguishing the above ensemble via a PPT
measurement should yield an optimal probability of 7/8.

.. literalinclude:: ../examples/opt_dist/ppt/min_error/indstinguishable_mes.py
   :language: python
   :linenos:
   :start-after: # along with this program.  If not, see <https://www.gnu.org/licenses/>.

In was also shown in [Cosentino13]_ that the optimal probability of distinguishing this ensemble
unambiguously when making use of PPT measurements was equal to 3/4.

.. literalinclude:: ../examples/opt_dist/ppt/unambiguous/indstinguishable_mes.py
   :language: python
   :linenos:
   :start-after: # along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
states with a resource state via PPT measurements is given by the closed-form
expression:

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

Note that [BCJRWY15]_ also proved the same closed-form expression for when Alice and
Bob make use of separable measurements. More on that in the 
`tutorial on distinguishing via separable measurements <https://qustop.readthedocs.io/en/latest/tutorials.separable.html>`_.

Werner hiding pairs
-------------------

In [TDL01]_ and [DLT02]_, a quantum data hiding protocol that encodes a
classical bit in a Werner hiding pair was provided.

A Werner hiding pair is defined by

.. math::
    \begin{equation}
        \sigma_0^{(n)} = \frac{\mathbb{I} \otimes \mathbb{I} + W_n}{n(n+1)}
        \quad \text{and} \quad
        \sigma_1^{(n)} = \frac{\mathbb{I} \otimes \mathbb{I} - W_n}{n(n-1)}
    \end{equation}

where 

.. math::
    W_n = \sum_{i,j=0}^{n-1} | i \rangle \langle j |
    \otimes | j \rangle \langle i | 
    \in \text{U}\left(\mathbb{C}^n \otimes \mathbb{C}^n\right)

is the swap operator defined for some dimension :math:`n \geq 2`.

It was shown in [Cosentino15]_ that

.. math::
    \begin{equation}
        \text{opt}_{\text{PPT}}(\eta) = \frac{1}{2} + \frac{1}{n+1},
    \end{equation}

where :math:`\eta = \left\{\sigma_0, \sigma_1\right\}`. Using :code:`qustop`,
we may encode this scenario as follows.

.. literalinclude:: ../examples/opt_dist/ppt/min_error/werner_hiding_pair.py
   :language: python
   :linenos:
   :start-after: # along with this program.  If not, see <https://www.gnu.org/licenses/>.

References
------------------------------
.. [TDL01] Terhal, Barbara M., David P. DiVincenzo, and Debbie W. Leung.
    "Hiding bits in Bell states." 
    Physical review letters 86.25 (2001): 5807.

.. [DLT02] DiVincenzo, David P., Debbie W. Leung, and Barbara M. Terhal. 
    "Quantum data hiding." 
    IEEE Transactions on Information Theory 48.3 (2002): 580-598.

.. [Cosentino15] Cosentino, Alessandro
    "Quantum state local distinguishability via convex optimization".
    University of Waterloo, Thesis
    https://uwspace.uwaterloo.ca/handle/10012/9572

.. [Cosentino13] Cosentino, Alessandro,
    "Positive-partial-transpose-indistinguishable states via semidefinite programming",
    Physical Review A 87.1 (2013): 012321.
    https://arxiv.org/abs/1205.1031

.. [CR13] Cosentino, Alessandro and Russo, Vincent
    "Small sets of locally indistinguishable orthogonal maximally entangled states",
    Quantum Information & Computation, Volume 14,
    https://arxiv.org/abs/1307.3232

.. [YDY12] Yu, Nengkun, Runyao Duan, and Mingsheng Ying.
    "Four locally indistinguishable ququad-ququad orthogonal
    maximally entangled states."
    Physical review letters 109.2 (2012): 020506.
    https://arxiv.org/abs/1107.3224

.. [BCJRWY15] Bandyopadhyay, Somshubhro, Cosentino, Alessandro, Johnston, Nathaniel, Russo, Vincent, Watrous, John, & Yu, Nengkun.
    "Limitations on separable measurements by convex optimization".
    IEEE Transactions on Information Theory 61.6 (2015): 3593-3604.
