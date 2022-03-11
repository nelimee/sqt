import typing as ty

from qiskit import QuantumCircuit
from qiskit.result import Result
from qiskit.providers.aer import AerSimulator, AerJob, AerJobSet
from qiskit.providers.ibmq import IBMQBackend
from qiskit.providers.ibmq.managed import IBMQJobManager, ManagedJobSet, ManagedJob


def submit(
    circuits: ty.List[QuantumCircuit],
    backend: ty.Union[IBMQBackend, AerSimulator],
    job_name: ty.Optional[str] = None,
    tags: ty.Optional[ty.List[str]] = None,
    **kwargs,
) -> ty.Union[ManagedJobSet, AerJob, AerJobSet]:
    """Submit the given circuits on the backend and returns.

    This function sumits the given circuits to the backend, using a job manager if
    needed to split the jobs in batches of appropriate sizes, and returns without
    waiting for the jobs to finish.

    :param circuits: quantum circuit instances to execute on the given backend.
    :param backend: backend used to execute the given circuits.
    :param job_name: prefix used for the job name. IBMQJobManager will add
        a suffix for each job.
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
    if isinstance(backend, IBMQBackend):
        manager = IBMQJobManager()
        managed_job: ManagedJobSet = manager.run(
            circuits, backend, name=job_name, job_tags=tags, **kwargs
        )
        return managed_job
    else:
        job: ty.Union[AerJob, AerJobSet] = backend.run(
            circuits, name=job_name, job_tags=tags, **kwargs
        )
        return job


def execute(
    circuits: ty.List[QuantumCircuit],
    backend: ty.Union[IBMQBackend, AerSimulator],
    job_name: ty.Optional[str] = None,
    tags: ty.Optional[ty.List[str]] = None,
    **kwargs,
) -> Result:
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
    job = submit(circuits, backend, job_name, tags, **kwargs)
    if isinstance(job, ManagedJobSet):
        return job.results().combine_results()
    else:
        return job.result()
