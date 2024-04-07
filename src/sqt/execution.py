from qiskit import QuantumCircuit
from qiskit.providers.backend import BackendV2 as Backend

from sqt.job import BaseJob


def submit(
    circuits: list[QuantumCircuit],
    backend: Backend,
    hub: str,
    group: str,
    project: str,
    tags: list[str] | None = None,
    **kwargs,
) -> BaseJob:
    """Submit the given circuits on the backend and returns.

    This function sumits the given circuits to the backend, using a job manager if
    needed to split the jobs in batches of appropriate sizes, and returns without
    waiting for the jobs to finish.

    Args:
        circuits: quantum circuit instances to execute on the given
            backend.
        backend: backend used to execute the given circuits.
        tags: tags for each of the submitted jobs.
        **kwargs: forwarded to the backend.run method. Configuration of
            the runtime environment. Some examples of these
            configuration parameters include: "qobj_id", "qobj_header",
            "shots", "memory", "seed_simulator", "qubit_lo_freq",
            "meas_lo_freq", "qubit_lo_range", "meas_lo_range",
            "schedule_los", "meas_level", "meas_return", "meas_map",
            "memory_slot_size", "rep_time", and "parameter_binds".

            Refer to the documentation on :func:`qiskit.compiler.assemble`
            for details on these arguments.
    """
    job = backend.run(circuits, dynamic=False, job_tags=tags, **kwargs)
    return BaseJob.from_job(job, hub, group, project)  # type: ignore


def execute(
    circuits: list[QuantumCircuit],
    backend: Backend,
    hub: str,
    group: str,
    project: str,
    tags: list[str] | None = None,
    **kwargs,
):
    """Execute the given circuits on the backend and returns the result.

    This function sumits the given circuits to the backend, using a job manager if
    needed to split the jobs in batches of appropriate sizes, and wait for all the
    jobs to finish in order to retrieve the results.

    Args:
        circuits: quantum circuit instances to execute on the given
            backend.
        backend: backend used to execute the given circuits.
        job_name: prefix used for the job name. IBMQJobManager will add
            a suffix for each job.
        tags: tags for each of the submitted jobs.
        **kwargs: forwarded to run_config. Configuration of the runtime
            environment. Some examples of these configuration parameters
            include: ``qobj_id``, ``qobj_header``, ``shots``,
            ``memory``, ``seed_simulator``, ``qubit_lo_freq``,
            ``meas_lo_freq``, ``qubit_lo_range``, ``meas_lo_range``,
            ``schedule_los``, ``meas_level``, ``meas_return``,
            ``meas_map``, ``memory_slot_size``, ``rep_time``, and
            ``parameter_binds``.

            Refer to the documentation on :func:`qiskit.compiler.assemble`
            for details on these arguments.
    """
    job = submit(circuits, backend, hub, group, project, tags, **kwargs)
    return job.result()
