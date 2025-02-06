from pennylane.templates import SimplifiedTwoDesign

# for i in range(8, 1, -1):
#    print(i)

n_layers = 3
n_qubits = 4
print(SimplifiedTwoDesign.shape(n_layers=3, n_wires=n_qubits))

