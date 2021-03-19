Distinguishing Quantum States via PPT Measurements
==================================================

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