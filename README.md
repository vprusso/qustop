# ![logo](./docs/figures/logo.svg "logo") 

# qustop

qustop (QUantum STate OPtimizer) is a Python toolkit for studying various
quantum state optimization scenarios including calculating optimal values for
quantum state distinguishability, quantum state exclusion, quantum state
cloning, and more.

## Installation

TODO: Mention Python 3.9 (min)

## The setting

The `qustop` package is focused on problems pertaining to the following setting:

# ![setting](./docs/figures/quantum_state_distinguish.svg "setting") 

## Applications

### State distinguishability

1. Alice possesses an ensemble of `n` quantum states:

    &eta; = ((p<sub>0</sub> &rho;<sub>0</sub>), ... , (p<sub>n</sub> &rho;<sub>n</sub>))

    where p<sub>i</sub> is the probability with which &rho;<sub>i</sub> is selected from the
    ensemble. Alice picks &rho;<sub>i</sub> with probability p<sub>i</sub> from her ensemble
    and sends &rho;<sub>i</sub> to Bob.

2. Bob receives &rho;<sub>i</sub> from Alice. Bob Alice and Bob are aware of how the ensemble is
   defined but Bob does **not** know what the value of index `i` corresponding to the state with
   &rho;<sub>i</sub> he receives from Alice is.

3. Bob wants to guess which of the states from the ensemble he was given. Inorder to do so, he
   may measure &rho;<sub>i</sub> to guess the index `i` for which the state in the ensemble
   corresponds.
   
Task: Compute the optimal value of

#### Examples

##### Distinguishing two orthogonal states

##### Four indistinguishable orthogonal maximally entangled states

##### Entanglement cost of discriminating Bell states

### State exclusion

### State cloning

## License

[GNU GPL v.3.0.](https://github.com/vprusso/qustop/blob/master/LICENSE)