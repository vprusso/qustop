from toqito.states import basis

# Define |0>, |1>, |2>:
dim = 3
e_0 = basis(dim, 0)
e_1 = basis(dim, 1)
e_2 = basis(dim, 2)

# Define the mutually unbiased bases:
z = np.exp((2 * np.pi * 1j) / dim)

mub_0 = [e_0, e_1, e_2]
mub_1 = [
    (e_0 + e_1 + e_2) / np.sqrt(3),
    (e_0 + z ** 2 * e_1 + z * e_2) / np.sqrt(3),
    (e_0 + z * e_1 + z ** 2 * e_2) / np.sqrt(3),
]
mub_2 = [
    (e_0 + e_1 + z * e_2) / np.sqrt(3),
    (e_0 + z ** 2 * e_1 + z ** 2 * e_2) / np.sqrt(3),
    (e_0 + z * e_1 + e_2) / np.sqrt(3),
]
mub_3 = [
    (e_0 + e_1 + z ** 2 * e_2) / np.sqrt(3),
    (e_0 + z ** 2 * e_1 + e_2) / np.sqrt(3),
    (e_0 + z * e_1 + z * e_2) / np.sqrt(3),
]
