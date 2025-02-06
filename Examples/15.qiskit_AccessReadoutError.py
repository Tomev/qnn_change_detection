# In 14, we were able to prepare all the gate errors that we need. What we lack are the
# readout errors. With them, we will have full qiskit.aer.NoiseModel recreated on pennylane
# device.
#
# For a single qubit, the readout error is basically a BitFlip operation with some
# specified probability. We have to get that probability from the qiskit.Backend.
# We will do so analogously to 13. and 14., by modifying adequate qiskit.aer methods.

from typing import Dict

from numpy import mean
from qiskit.circuit.library import PauliTwoDesign
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.noise.device.models import (Gate, Measure, _excited_population,
                                            _truncate_t2_value, inf, qi,
                                            thermal_relaxation_error)
from qiskit_ibm_runtime import QiskitRuntimeService


def layout_to_map(layout, quantum_register) -> Dict[int, int]:
    # Get simple qubit map from circuit layout.
    # The map is a dict in form {virtual -> physical}.
    # We want to act on the physical qubits.
    initial_layout = layout.initial_layout.get_virtual_bits()
    # print(test)

    # print(initial_layout)

    qubit_map = {}

    for k in quantum_register:
        qubit_map[k._index] = initial_layout[k]

    return qubit_map


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
            depol_error_param = None
            relax_error = None
            relax_error_params = None

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
                relax_error, relax_error_params = device_thermal_relaxation_error(
                    qubits=qubits,
                    gate_time=inst_prop.duration,
                    relax_params=relax_params,
                    temperature=temperature,
                )
            # Get depolarizing error
            if gate_error and inst_prop.error and isinstance(operation, Gate):
                depol_error_param = device_depolarizing_error_param(
                    qubits=qubits,
                    error_param=inst_prop.error,
                    relax_error=relax_error,
                )

            errors.append((op_name, qubits, relax_error_params, depol_error_param))

    return errors


# For depolarization it's easy, because I only need the p parameter to initialize
# pennylane depolarizing channel. We will compute it as in qiskit.
def device_depolarizing_error_param(qubits, error_param, relax_error=None):
    """Construct a depolarizing_error for device.
    If un-physical parameters are supplied, they are truncated to the theoretical bound values.
    """

    # We now deduce the depolarizing channel error parameter in the
    # presence of T1/T2 thermal relaxation. We assume the gate error
    # parameter is given by e = 1 - F where F is the average gate fidelity,
    # and that this average gate fidelity is for the composition
    # of a T1/T2 thermal relaxation channel and a depolarizing channel.

    # For the n-qubit depolarizing channel E_dep = (1-p) * I + p * D, where
    # I is the identity channel and D is the completely depolarizing
    # channel. To compose the errors we solve for the equation
    # F = F(E_dep * E_relax)
    #   = (1 - p) * F(I * E_relax) + p * F(D * E_relax)
    #   = (1 - p) * F(E_relax) + p * F(D)
    #   = F(E_relax) - p * (dim * F(E_relax) - 1) / dim

    # Hence we have that the depolarizing error probability
    # for the composed depolarization channel is
    # p = dim * (F(E_relax) - F) / (dim * F(E_relax) - 1)
    if relax_error is not None:
        relax_fid = qi.average_gate_fidelity(relax_error)
        relax_infid = 1 - relax_fid
    else:
        relax_fid = 1
        relax_infid = 0
    if error_param is not None and error_param > relax_infid:
        num_qubits = len(qubits)
        dim = 2**num_qubits
        error_max = dim / (dim + 1)
        # Check if reported error param is un-physical
        # The minimum average gate fidelity is F_min = 1 / (dim + 1)
        # So the maximum gate error is 1 - F_min = dim / (dim + 1)
        error_param = min(error_param, error_max)
        # Model gate error entirely as depolarizing error
        num_qubits = len(qubits)
        dim = 2**num_qubits
        depol_param = dim * (error_param - relax_infid) / (dim * relax_fid - 1)
        max_param = 4**num_qubits / (4**num_qubits - 1)
        if depol_param > max_param:
            depol_param = min(depol_param, max_param)
        return {"depol_param": depol_param, "num_qubits": num_qubits}
    return None


#
def device_thermal_relaxation_error(qubits, gate_time, relax_params, temperature):
    """Construct a thermal_relaxation_error for device.

    Expected units: frequency in relax_params [Hz], temperature [mK].
    Note that gate_time and T1/T2 in relax_params must be in the same time unit.
    """
    # Check trivial case
    if gate_time is None or gate_time == 0:
        return None

    # Construct a tensor product of single qubit relaxation errors
    # for any multi qubit gates
    first = True
    error = None
    params = {}  # I'll use Pennylane name for params.
    for qubit in qubits:
        t1, t2, freq = relax_params[qubit]
        t2 = _truncate_t2_value(t1, t2)
        if t1 is None:
            t1 = inf
        if t2 is None:
            t2 = inf
        population = _excited_population(freq, temperature)
        if first:
            error = thermal_relaxation_error(t1, t2, gate_time, population)
            params[qubit] = {"t1": t1, "t2": t2, "pe": population, "tg": gate_time}
            first = False
        else:
            single = thermal_relaxation_error(t1, t2, gate_time, population)
            params[qubit] = {"t1": t1, "t2": t2, "pe": population, "tg": gate_time}
            error = error.expand(single)
    return error, params


def filter_gate_error_params(gate_errors, qubit_map):
    filtered_params = []
    physical_qubits_used = qubit_map.values()

    for ge in gate_errors:
        gate_wires = ge[1]
        if all(gate_wire in physical_qubits_used for gate_wire in gate_wires):
            filtered_params.append(ge)

    return filtered_params


def permute_qubits_in_error_params(gate_errors, qubit_map):
    permuted_params = []
    inv_qubit_map = {v: k for k, v in qubit_map.items()}

    for ge in gate_errors:
        permuted_params.append(
            (
                ge[0],
                tuple(inv_qubit_map[q] for q in ge[1]),
                {inv_qubit_map[k]: v for k, v in ge[2].items()} if ge[2] else None,
                ge[3],
            )
        )

    return permuted_params


def basic_device_readout_errors(target):
    """
    Return readout error parameters from either of device Target or BackendProperties.

    If ``target`` is supplied, ``properties`` will be ignored.

    Args:
        properties (BackendProperties): device backend properties
        target (Target): device backend target

    Returns:
        list: A list of pairs ``(qubits, ReadoutError)`` for qubits with
        non-zero readout error values.

    Raises:
        NoiseError: if neither properties nor target is supplied.
    """
    errors = []

    # create from Target
    for q in range(target.num_qubits):
        meas_props = target.get("measure", None)
        if meas_props is None:
            continue
        prop = meas_props.get((q,), None)
        if prop is None:
            continue
        if hasattr(prop, "prob_meas1_prep0") and hasattr(prop, "prob_meas0_prep1"):
            p0m1, p1m0 = prop.prob_meas1_prep0, prop.prob_meas0_prep1
        else:
            p0m1, p1m0 = prop.error, prop.error
        # Return mean probability of error
        errors.append((q, mean((p0m1, p1m0))))

    return errors


def filter_readout_errors(readout_errors, qubit_map):
    filtered_errors = []
    physical_qubits_used = qubit_map.values()

    for re in readout_errors:
        gate_wire = re[0]
        if gate_wire in physical_qubits_used:
            filtered_errors.append(re)

    return filtered_errors


def permute_readout_error_qubits(readout_errors, qubit_map):
    permuted_params = []
    inv_qubit_map = {v: k for k, v in qubit_map.items()}

    for re in readout_errors:
        permuted_params.append((inv_qubit_map[re[0]], re[1]))

    return permuted_params


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

    # Now get the mapping for our circuit
    print("\tGet the mapping!")
    # Config
    n_qubits = 4
    n_wires = 2 * n_qubits

    ansatz = PauliTwoDesign(n_wires, reps=1)

    # Ansatz transpilation

    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
    transpiled_ansatz = pm.run(ansatz)

    # print(transpiled_ansatz)

    # return

    qubit_map = layout_to_map(transpiled_ansatz.layout, ansatz.qubits)
    print(qubit_map)

    print("\t\nFilter the results!\n")
    gate_errors = filter_gate_error_params(gate_errors, qubit_map)

    for ge in gate_errors:
        print(f"\t{ge}")

    print("\tPermute the qubits!")
    gate_errors = permute_qubits_in_error_params(gate_errors, qubit_map)

    for ge in gate_errors:
        print(f"\t{ge}")

    print("\tGet readout errors!\n")
    readout_errors = basic_device_readout_errors(backend.target)

    for re in readout_errors:
        print(f"\t\t{re}")

    print("\n\tFilter readout errors!\n")
    readout_errors = filter_readout_errors(readout_errors, qubit_map)

    for re in readout_errors:
        print(f"\t\t{re}")

    print("\n\tPermute readout error qubits!\n")
    readout_errors = permute_readout_error_qubits(readout_errors, qubit_map)

    for re in readout_errors:
        print(f"\t\t{re}")


if __name__ == "__main__":
    print("Experiment start.")
    main()
    print("Experiment finish.")
