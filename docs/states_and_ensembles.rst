States and Ensembles
====================

Quantum States
---------------

A *quantum state* is a density operator

.. math::
    \rho \in \text{D}(\mathcal{X})

where :math:`\mathcal{X}` is a complex Euclidean space and where
:math:`\text{D}(\cdot)` represents the set of density matrices.

.. toctree::

.. autosummary::
    :toctree: _autosummary

    qustop.State

Ensembles
----------

An *ensemble* is a collection of quantum states defined by a function

.. math::
    \eta : \Gamma \rightarrow \text{Pos}(\mathcal{X})

that satisfies

.. math::
    \text{Tr}\left( \sum_{a \in \Gamma} \eta(a) \right) = 1.

.. toctree::

.. autosummary::
    :toctree: _autosummary

    qustop.Ensemble
