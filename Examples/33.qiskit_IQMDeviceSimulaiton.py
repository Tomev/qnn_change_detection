import os

from iqm.qiskit_iqm import IQMProvider
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def main() -> None:
    # get backend
    iqm_server_url = "https://cocos.resonance.meetiqm.com/garnet"
    token = os.environ["IQM_TOKEN"]
    provider = IQMProvider(iqm_server_url, token=token)
    backend = provider.get_backend()

    # create a bell circuit
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    circuit.draw()

    # transpile to backend
    circ = transpile(circuit, backend=backend)

    print(circ.num_qubits)

    # create a simulator for backend
    sim = AerSimulator().from_backend(backend)

    # simulate and extract results
    simulator_result = sim.run(circ).result()
    simulator_counts = simulator_result.get_counts()
    print(simulator_counts)


if __name__ == "__main__":
    print("Experiment start.")
    main()
    print("Experiment finish.")
