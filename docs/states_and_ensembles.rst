States and ensembles
====================

Quantum states, and collections of those quantum states that form ensembles,
are the core building blocks of :code:`qustop`.

Quantum states
---------------

.. toctree::

.. autosummary::
    :toctree: _autosummary

    qustop.State

A *quantum state* is a density operator

.. math::
    \rho \in \text{D}(\mathbb{C}^d)

where :math:`\mathbb{C}^d` is a complex Euclidean space of dimension :math:`d`
and where :math:`\text{D}(\cdot)` represents the set of density matrices, that
is, the set of matrices that are positive semidefinite with trace equal to
:math:`1`. We will typically represent complex Euclidean spaces using the
scripted capital letters :math:`\mathcal{A}, \mathcal{B}, \mathcal{X},
\mathcal{Y}`, etc.

Consider the density matrix corresponding to one of the four Bell states

.. math::
   \rho_0 = | \psi_0 \rangle \langle \psi_0 | = \frac{1}{2}
   \begin{pmatrix}
    1 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 \\
    1 & 0 & 0 & 1
   \end{pmatrix} \in \text{D}(\mathcal{A} \otimes \mathcal{B})

where

.. math::
    | \psi_0 \rangle = \frac{1}{\sqrt{2}}
   \left( | 00 \rangle + | 11 \rangle \right) \in
   \mathcal{A} \otimes \mathcal{B}

such that :math:`\mathcal{A} = \mathbb{C}^2` and :math:`\mathcal{B} =
\mathbb{C}^2`. We can use :code:`qustop` to encode this state as follows.

.. code-block:: python

    import numpy as np
    from qustop import State

    # Define the |0> and |1> ket vectors.
    q_0 = np.array([[1, 0]]).T
    q_1 = np.array([[0, 1]]).T

    # Define the respective dimensions of each complex Euclidean space.
    dims = [2, 2]

    # Define the Bell state vector.
    psi_0 = 1/np.sqrt(2) * np.kron(q_0, q_0) + 1/np.sqrt(2) * np.kron(q_1, q_1)

    # Define a `State` object corresponding to the Bell vector.
    rho_0 = State(psi_0, dims)


Printing the :code:`rho_0` variable gives some further information about the
state.

.. code-block:: python

    >>> print(rho_0)
    State:
     dimensions = [2, 2],
     spaces = ℂ^2 ⊗ ℂ^2,
     labels = A_1 ⊗ B_2,
     shape = (4, 4),

For instance, we see the :code:`shape` attribute gives information about the
size of the density matrix of the state. There is also information about the
subsystems along with which party the subsystems belong to (either Alice or
Bob), etc.

We can use the :code:`value` property of any :code:`State` object to obtain the
:code:`numpy` matrix representation of the quantum state

.. code-block:: python

    >>> print(rho_0.value)
    [[0.5 0.  0.  0.5]
     [0.  0.  0.  0. ]
     [0.  0.  0.  0. ]
     [0.5 0.  0.  0.5]]



Ensembles
---------

.. toctree::

.. autosummary::
    :toctree: _autosummary

    qustop.Ensemble


An *ensemble* is a collection of :math:`N` quantum states defined over some
complex Euclidean space :math:`\mathcal{X}` as

.. math::
    \eta = \left\{(p_1, \rho_1), \ldots, (p_N, \rho_N) \right\},

where :math:`(p_1, \ldots, p_N)` is a vector of probability values and where
:math:`\rho_1, \ldots, \rho_N \in \text{D}(\mathcal{X})` are quantum states.

Recall the four two-qubit Bell states

.. math::
    \begin{equation}
        \begin{aligned}
            | \psi_0 \rangle = \frac{| 00 \rangle + | 11 \rangle}{\sqrt{2}}, &\quad
            | \psi_1 \rangle = \frac{| 01 \rangle + | 10 \rangle}{\sqrt{2}}, \\
            | \psi_2 \rangle = \frac{| 01 \rangle - | 10 \rangle}{\sqrt{2}}, &\quad
            | \psi_3 \rangle = \frac{| 00 \rangle - | 11 \rangle}{\sqrt{2}}.
        \end{aligned}
    \end{equation}

We can define the following ensemble consisting of the Bell states where the
probability of selecting any one state from the ensemble is equal to
:math:`1/4`:

.. math::
    \begin{equation}
        \eta = \left\{
                \left(| \psi_0 \rangle, \frac{1}{4} \right),
                \left(| \psi_1 \rangle, \frac{1}{4} \right),
                \left(| \psi_2 \rangle, \frac{1}{4} \right),
                \left(| \psi_3 \rangle, \frac{1}{4} \right)
               \right\}.
    \end{equation}

In :code:`qustop`, we would define this ensemble like so

.. code-block:: python

    from toqito.states import bell
    from qustop import State, Ensemble

    # Construct the corresponding density matrices of the Bell states.
    dims = [2, 2]
    states = [
        State(bell(0) * bell(0).conj().T, dims),
        State(bell(1) * bell(1).conj().T, dims),
        State(bell(2) * bell(2).conj().T, dims),
        State(bell(3) * bell(3).conj().T, dims)
    ]
    ensemble = Ensemble(states=states, probs=[1/4, 1/4, 1/4, 1/4])
