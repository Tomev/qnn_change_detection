# Our goal is to use https://pennylane.ai/qml/demos/tutorial_noisy_circuits
# to simulate the training on a selected noisy qiskit device.

# We first have to be able to understand and programatically access the errors.

# Let's start with understanding what the errors are. To do so, we probably need to
# understand how they model error in qiskit. This is described in
# https://quantumcomputing.stackexchange.com/questions/14035/what-do-the-gate-errors-rates-mean-physically-for-ibms-quantum-computers
# and backed up by
# https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html#qiskit_aer.noise.NoiseModel.from_backend

# By analyzing qiskit.aer code, we can follow how the noise model is obtained from
# from the backend. We start with NoiseModel.from_backend method
# https://github.com/Qiskit/qiskit-aer/blob/269c26fd0552836db54f27d511e07167f58d7990/qiskit_aer/noise/noise_model.py#L243
# and see that when backend is specified it calls basic_device_gate_errors to add
# the gate errors to the NoiseModel.

# Analysis of the basic_device_gate_errors
# https://github.com/Qiskit/qiskit-aer/blob/main/qiskit_aer/noise/device/models.py#L82
# shows that when target (meaning backend) is specified, then
# _basic_device_target_gate_errors is called and returned.

# We can find _basic_device_target_gate_errors in the same file.
# https://github.com/Qiskit/qiskit-aer/blob/main/qiskit_aer/noise/device/models.py#L238
# There we can see what parameters are used to instantiate depolarizing and
# thermal relaxation errors.

# Ideally, we might even be able to use the very same function to add noise to our
# pennylane model.

########################################################################################
# In the following, we will try to get all error gates from a selected qiskit backend.
# Contrary to qiskit.aer, we want to get separate thermal_relaxation and depolarizing
# gate errors. We will get them by modyfing _basic_device_target_gate_errors method
# of qiskit.aer.

from os import environ

from qiskit_aer.noise.device.models import (Gate, Measure,
                                            _device_depolarizing_error,
                                            _device_thermal_relaxation_error)
from qiskit_ibm_runtime import QiskitRuntimeService


def custom_basic_device_target_gate_errors(
    backend_target, gate_error=True, thermal_relaxation=True, temperature=0
):
    """Return QuantumErrors derived from a devices Target.
    Note that, in the resulting error list, non-Gate instructions (e.g. Reset) will have
    no gate errors while they may have thermal relaxation errors. Exceptionally,
    Measure instruction will have no errors, neither gate errors nor relaxation errors.

    Note: Units in use: Time [s], Frequency [Hz], Temperature [mK]
    """
    # Notice that the "NoiseModel.from_backend" method is called with temperature=0 by
    # default, so it's all good.

    errors = []
    for op_name, inst_prop_dic in backend_target.items():
        operation = backend_target.operation_from_name(op_name)
        if isinstance(operation, Measure):
            continue
        if inst_prop_dic is None:  # ideal simulator
            continue
        for qubits, inst_prop in inst_prop_dic.items():
            if inst_prop is None:
                continue
            depol_error = None
            relax_error = None
            # Get relaxation error
            if thermal_relaxation and inst_prop.duration:
                relax_params = {
                    q: (
                        backend_target.qubit_properties[q].t1,
                        backend_target.qubit_properties[q].t2,
                        backend_target.qubit_properties[q].frequency,
                    )
                    for q in qubits
                }
                relax_error = _device_thermal_relaxation_error(
                    qubits=qubits,
                    gate_time=inst_prop.duration,
                    relax_params=relax_params,
                    temperature=temperature,
                )
            # Get depolarizing error
            if gate_error and inst_prop.error and isinstance(operation, Gate):
                depol_error = _device_depolarizing_error(
                    qubits=qubits,
                    error_param=inst_prop.error,
                    relax_error=relax_error,
                )
            errors.append((op_name, qubits, (relax_error, depol_error)))

    return errors


def main() -> None:
    # Get selected qiskit backend.

    # The code below assumes that the account is already saved.
    # If not then run:
    # token = "xxx"
    # service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    # service.save_account(channel="ibm_quantum", token=token)
    # This has to run only once.

    print("\tGet backend!")
    service = QiskitRuntimeService()
    backend = service.backend("ibm_brisbane")

    print("\tGet gate errors!")
    # Get gate errors
    gate_errors = custom_basic_device_target_gate_errors(backend_target=backend.target)

    print("\tPrint gate errors!")
    # Print the gates
    for ge in gate_errors:
        print(f"\t{ge}")


if __name__ == "__main__":
    print("Experiment start.")
    main()
    print("Experiment finish.")
