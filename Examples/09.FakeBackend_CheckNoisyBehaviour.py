from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeManilaV2


def main() -> None:
    # Get a fake backend from the fake provider
    backend = FakeManilaV2()

    # Create a simple circuit
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.measure_all()
    circuit.draw("mpl", style="iqp")

    # Transpile the ideal circuit to a circuit that can be directly executed by the backend
    transpiled_circuit = transpile(circuit, backend)
    transpiled_circuit.draw("mpl", style="iqp")

    # Run the transpiled circuit using the simulated fake backend
    sampler = SamplerV2(backend)
    job = sampler.run([transpiled_circuit])
    pub_result = job.result()[0]
    counts = pub_result.data.meas.get_counts()
    # print(counts)
    print(f"Is device noisy: {len(counts) > 2}")


if __name__ == "__main__":
    print("Experiment start!")
    main()
    print("Experiment stop!")
