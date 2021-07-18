Introductory tutorial
======================

This tutorial will illustrate the basics of how to use :code:`qustop`.

This is a user guide for :code:`qustop` and is not meant to serve as an
introduction to quantum information. For introductory material on quantum
information, please consult "Quantum Information and Quantum Computation" by
Nielsen and Chuang or the freely available lecture notes `"Introduction to
Quantum Computing"
<https://cs.uwaterloo.ca/~watrous/LectureNotes/CPSC519.Winter2006/all.pdf)>`_
by John Watrous.

More advanced tutorials can be found on the `tutorials page
<https://qustop.readthedocs.io/en/latest/tutorials.html>`_.

This tutorial assumes you have :code:`qustop` installed on your machine. If you
do not, please consult the `installation instructions
<https://qustop.readthedocs.io/en/latest/install.html>`_.

States, ensembles, and measurements
-----------------------------------



Measurements
^^^^^^^^^^^^

A *measurement* is defined as a function

.. math::
    \mu : \Sigma \rightarrow \text{Pos}(\mathcal{X})

for some finite and nonempty set :math:`\Sigma` and some complex Euclidean
space :math:`\mathcal{X}` satisfying the constraint that

.. math::
    \sum_{a \in \Sigma} \mu(a) = \mathbb{I}_{\mathcal{X}}.

There are many different classes of measurements.

LOCC measurements
^^^^^^^^^^^^^^^^^

Separable measurements
^^^^^^^^^^^^^^^^^^^^^^^

PPT measurements
^^^^^^^^^^^^^^^^

Positive (global) measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Quantum state distinguishability
---------------------------------

Given an ensemble of quantum states, we can consider the setting of *quantum
state distinguishability*. This setting can be considered as an interaction between
two parties--typically denoted as *Alice* and *Bob*.

A more in-depth description and tutorial on this setting in :code:`qustop` can
be found in:

- `Tutorial : Quantum State Distinguishability <https://qustop.readthedocs.io/en/latest/tutorials.quantum_state_distinguishabiliy.html>`_.

More in-depth descriptions pertaining to quantum state distinguishability under PPT, separable, and positive measurements can be found in:

- `Tutorial: Quantum State Distinguishability using PPT Measurements
  <https://qustop.readthedocs.io/en/latest/tutorials.ppt.html>`_.

- `Tutorial: Quantum State Distinguishability using Separable Measurements
  <https://qustop.readthedocs.io/en/latest/tutorials.separable.html>`_.

- `Tutorial: Quantum State Distinguishability using Positive Measurements
  <https://qustop.readthedocs.io/en/latest/tutorials.positive.html>`_.

Quantum state exclusion
-----------------------
(Coming soon).