# In the last example, we could get the access to the errors, but it was already too
# much compiled to be usable. What we actually want are the parameters for particular
# gate errors. We will obtain them, again, by modifying some default qiskit.aer functions.

from qiskit_aer.noise.device.models import (Gate, Measure, _excited_population,
                                            _truncate_t2_value, inf, qi,
                                            thermal_relaxation_error)
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

            errors.append((op_name, qubits, (relax_error_params, depol_error_param)))

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
