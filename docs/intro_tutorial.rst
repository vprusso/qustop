Introductory tutorial
======================

This tutorial will illustrate the basics of how to use :code:`qustop`.

This is a user guide for :code:`qustop` and is not meant to serve as an
introduction to quantum information. For introductory material on quantum
information, please consult "Quantum Information and Quantum Computation" by
Nielsen and Chuang or the freely available lecture notes `"Introduction to
Quantum Computing"
<https://cs.uwaterloo.ca/~watrous/LectureNotes/CPSC519.Winter2006/all.pdf)>`_ by
John Watrous.

More advanced tutorials can be found on the `main documentation directory
<https://qustop.readthedocs.io/en/latest/index.html>`_.

This tutorial assumes you have :code:`qustop` installed on your machine. If you
do not, please consult the `getting started instructions
<https://qustop.readthedocs.io/en/latest/getting_started.html>`_.

States, ensembles, and measurements
-----------------------------------

Quantum states, and collections of those quantum states that form ensembles, are
the core building blocks of :code:`qustop`.

Quantum states
^^^^^^^^^^^^^^

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
     pure = True,
     shape = (4, 4),


For instance, we see the :code:`shape` attribute gives information about the
size of the density matrix of the state. There is also information about the
subsystems along with which party the subsystems belong to (either Alice or
Bob), whether the state is pure, etc.

We can use the :code:`value` property of any :code:`State` object to obtain the
:code:`numpy` matrix representation of the quantum state

.. code-block:: python

    >>> print(rho_0.value)
    [[0.5 0.  0.  0.5]
     [0.  0.  0.  0. ]
     [0.  0.  0.  0. ]
     [0.5 0.  0.  0.5]]

We can also do things like take tensor products of :code:`State` objects.

.. code-block:: python

    >>> sigma_0 = rho_0.kron(rho_0)
    >>> print(sigma_0)
    State:
     dimensions = [2, 2, 2, 2],
     spaces = ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2,
     labels = A_1 ⊗ B_2 ⊗ A_3 ⊗ B_4,
     pure = True,
     shape = (16, 16),

It is sometimes convenient to swap the subsystems of a given state. For
instance, this example shows how we can swap the second and third subsystems of
the :code:`sigma_0` state.

.. code-block:: python

    >>> sigma_0.swap([2, 3])
    >>> print(sigma_0)
    State:
     dimensions = [2, 2, 2, 2],
     spaces = ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2,
     labels = A_1 ⊗ A_3 ⊗ B_2 ⊗ B_4,
     pure = True,
     shape = (16, 16),

Notice how the :code:`A_3` and :code:`B_2` subsystems are swapped.

Ensembles
^^^^^^^^^

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

The corresponding density operators may be defined as

.. math::
    \begin{equation}
        \begin{aligned}
            \rho_0 = | \psi_0 \rangle \langle \psi_0 |, &\quad
            \rho_1 = | \psi_1 \rangle \langle \psi_1 |, \\
            \rho_2 = | \psi_2 \rangle \langle \psi_2 |, &\quad
            \rho_3 = | \psi_3 \rangle \langle \psi_3 |.
        \end{aligned}
    \end{equation}

We can define the following ensemble consisting of the Bell states where the
probability of selecting any one state from the ensemble is equal to
:math:`1/4`:

.. math::
    \begin{equation}
        \eta = \left\{
                \left(\frac{1}{4}, \rho_0 \right),
                \left(\frac{1}{4}, \rho_1 \right),
                \left(\frac{1}{4}, \rho_2 \right),
                \left(\frac{1}{4}, \rho_3 \right)
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

Printing out any :code:`Ensemble` object gives us some information about the
contents:

.. code-block:: python

    >>> print(ensemble)
    Ensemble:
     num_states = 4,
     states = ρ_0 ⊗ ρ_1 ⊗ ρ_2 ⊗ ρ_3,
     is_mutually_orthogonal = True,
     is_linearly_independent = True,

We can see certain pieces of information including how many states are contained
in the ensemble, whether the states in the ensemble are all mutually orthogonal,
linearly independent, etc.

We can access any of the states from the :code:`Ensemble` object using standard
array indexing notation. For instance, here is how we can access the first state
in the ensemble.

.. code-block:: python

    >>> print(ensemble[0])
    State:
     dimensions = [2, 2],
     spaces = ℂ^2 ⊗ ℂ^2,
     labels = A_1 ⊗ B_2,
     pure = True,
     shape = (4, 4),

We may also wish to apply some of the functions that we saw before for
:code:`State` objects onto the entire ensemble. For instance, here is an example
of how we can swap the first and second subsystems of each state in the
ensemble.

.. code-block:: python

    >>> ensemble.swap([1, 2])
    >>> print(ensemble[0])
    State:
     dimensions = [2, 2],
     spaces = ℂ^2 ⊗ ℂ^2,
     labels = B_2 ⊗ A_1,
     pure = True,
     shape = (4, 4),


Measurements
^^^^^^^^^^^^

A *measurement* is defined as a function

.. math::
    \mu : \Sigma \rightarrow \text{Pos}(\mathcal{X})

for some finite and nonempty set :math:`\Sigma` and some complex Euclidean space
:math:`\mathcal{X}` satisfying the constraint that

.. math::
    \sum_{a \in \Sigma} \mu(a) = \mathbb{I}_{\mathcal{X}}.

There are many different classes of measurements.

Quantum state distinguishability
---------------------------------

Given an ensemble of quantum states, we can consider the setting of *quantum
state distinguishability*. This setting can be considered as an interaction
between two parties--typically denoted as *Alice* and *Bob*.

A more in-depth description and tutorial on this setting in :code:`qustop` can
be found in:

- `Tutorial : Quantum state distinguishability <https://qustop.readthedocs.io/en/latest/tutorials.state_distinguishability.html>`_.

More in-depth descriptions pertaining to quantum state distinguishability under
positive, PPT, and separable measurements can be found in:

- `Tutorial: Quantum state distinguishability using positive measurements
  <https://qustop.readthedocs.io/en/latest/tutorials.positive.html>`_.

- `Tutorial: Quantum state distinguishability using PPT measurements
  <https://qustop.readthedocs.io/en/latest/tutorials.ppt.html>`_.

- `Tutorial: Quantum state distinguishability using separable measurements
  <https://qustop.readthedocs.io/en/latest/tutorials.separable.html>`_.

Quantum state exclusion
-----------------------

- `Tutorial : Quantum state exclusion <https://qustop.readthedocs.io/en/latest/tutorials.state_exclusion.html>`_.

Quantum state cloning
----------------------

- `Tutorial : Quantum state cloning <https://qustop.readthedocs.io/en/latest/tutorials.state_cloning.html>`_.
