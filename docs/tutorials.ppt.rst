Distinguishing Quantum States via PPT Measurements
==================================================

Minimum-Error
-------------

It was shown in [arXiv:1205.1031](https://arxiv.org/abs/1205.1031)[Cosentino13]_ and later
extended in [arXiv:1307.3232](https://arxiv.org/abs/1307.3232)[CR13]_ that for the
following set of states

.. math::
    \rho_0 = |\psi_0\rangle |\psi_0 \rangle \langle \psi_0 | \langle \psi_0 |

\rho_0=|\psi_0\rangle|\psi_0\rangle\langle\psi_0|\langle\psi_0|,\quad\rho_1=|\psi_1\rangle|\psi_3\rangle\langle\psi_1|\langle\psi_3|,)

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2c}](https://latex.codecogs.com/svg.latex?\Large&space;\rho_2=|\psi_2\rangle|\psi_3\rangle\langle\psi_2|\langle\psi_3|,\quad\rho_3=|\psi_3\rangle|\psi_3\rangle\langle\psi_3|\langle\psi_3|,)

that the optimal probability of distinguishing via a PPT measurement should
yield an optimal probability of 7/8.

TODO Cite [Cosentino13]_ [Cosentino15]_

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
            \textbf{Primal:} \quad & \\
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

[Cosentino13]_

Entanglement cost of distinguishing Bell states
-----------------------------------------------

[BCJRWY15]_

Distinguishing three Bell states
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Distinguishing four Bell states
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Distinguishing Yu-Duan-Ying states
----------------------------------

[CR13]_ [YDY12]_

Werner hiding pairs
-------------------

TODO Cite [TDL01]_, [DLT02]_

A Werner hiding pair is defined by

.. math::
    \begin{equation}
        \sigma_0^{(n)} = \frac{\mathbb{I} \otimes \mathbb{I} + W_n}{n(n+1)}
        \quad \text{and} \quad
        \sigma_1^{(n)} = \frac{\mathbb{I} \otimes \mathbb{I} - W_n}{n(n-1)}
    \end{equation}

.. literalinclude:: ../examples/opt_dist/ppt/werner_hiding_pair.py
   :language: python
   :linenos:

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
