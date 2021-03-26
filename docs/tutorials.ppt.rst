Distinguishing Quantum States via PPT Measurements
==================================================

Minimum-Error
-------------

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


Entanglement cost of distinguishing Bell states
-----------------------------------------------

Distinguishing three Bell states
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Distinguishing four Bell states
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Distinguishing Yu-Duan-Ying states
----------------------------------

Werner hiding pairs
-------------------
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