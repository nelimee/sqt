import typing as ty

from qiskit import QuantumCircuit
from qiskit.providers.backend import BackendV2 as Backend
from qiskit_ibm_runtime import RuntimeJobV2 as RuntimeJob


def submit(
    circuits: list[QuantumCircuit],
    backend: Backend,
    tags: list[str] | None = None,
    **kwargs,
) -> RuntimeJob:
    """Submit the given circuits on the backend and returns.

    This function sumits the given circuits to the backend, using a job manager if
    needed to split the jobs in batches of appropriate sizes, and returns without
    waiting for the jobs to finish.

    :param circuits: quantum circuit instances to execute on the given backend.
    :param backend: backend used to execute the given circuits.
    :param tags: tags for each of the submitted jobs.
    :param kwargs: forwarded to the backend.run method.
        Configuration of the runtime environment. Some
        examples of these configuration parameters include:
        "qobj_id", "qobj_header", "shots", "memory",
        "seed_simulator", "qubit_lo_freq", "meas_lo_freq",
        "qubit_lo_range", "meas_lo_range", "schedule_los",
        "meas_level", "meas_return", "meas_map",
        "memory_slot_size", "rep_time", and "parameter_binds".

        Refer to the documentation on :func:`qiskit.compiler.assemble`
        for details on these arguments.
    """
    return backend.run(circuits, dynamic=False, job_tags=tags, **kwargs)  # type: ignore


def execute(
    circuits: ty.List[QuantumCircuit],
    backend: Backend,
    tags: ty.Optional[ty.List[str]] = None,
    **kwargs,
):
    """Execute the given circuits on the backend and returns the result.

    This function sumits the given circuits to the backend, using a job manager if
    needed to split the jobs in batches of appropriate sizes, and wait for all the
    jobs to finish in order to retrieve the results.

    :param circuits: quantum circuit instances to execute on the given backend.
    :param backend: backend used to execute the given circuits.
    :param job_name: prefix used for the job name. IBMQJobManager will add
        a suffix for each job.
    :param tags: tags for each of the submitted jobs.
    :param kwargs: forwarded to run_config.
        Configuration of the runtime environment. Some
        examples of these configuration parameters include:
        ``qobj_id``, ``qobj_header``, ``shots``, ``memory``,
        ``seed_simulator``, ``qubit_lo_freq``, ``meas_lo_freq``,
        ``qubit_lo_range``, ``meas_lo_range``, ``schedule_los``,
        ``meas_level``, ``meas_return``, ``meas_map``,
        ``memory_slot_size``, ``rep_time``, and ``parameter_binds``.

        Refer to the documentation on :func:`qiskit.compiler.assemble`
        for details on these arguments.
    """
    job = submit(circuits, backend, tags, **kwargs)
    return job.result()
