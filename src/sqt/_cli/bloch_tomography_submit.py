import typing as ty
import argparse
import itertools as it
import pickle
from pathlib import Path

from qiskit import IBMQ, QuantumCircuit
from qiskit.providers.aer.jobs.aerjob import AerJob
from qiskit.providers.aer.jobs.aerjobset import AerJobSet
from qiskit.providers.ibmq import AccountProvider
from qiskit.providers.ibmq.managed.managedjobset import ManagedJobSet

from sqt.basis.base import BaseMeasurementBasis

from sqt.basis.equidistant import get_approximately_equidistant_circuits
from sqt.basis.pauli import PauliMeasurementBasis
from sqt.basis.tetrahedral import TetrahedralMeasurementBasis
from sqt.circuits import one_qubit_tomography_circuits
from sqt.execution import submit
from sqt.passes import compile_circuits

_BASIS = {
    "pauli": PauliMeasurementBasis(),
    "tetrahedral": TetrahedralMeasurementBasis(),
}

_DEFAULT_SAVE = Path.cwd()


def get_backup_filename(
    backup_dir: Path,
    raw_circuits: ty.List[QuantumCircuit],
    backend,
    basis: BaseMeasurementBasis,
    job: ManagedJobSet,
) -> Path:
    filename: str = (
        f"backend-{backend.name()}_basis-{basis.name}_points-{len(raw_circuits)}"
        f"_jobid-{job.job_set_id()}.pkl"
    )
    path: Path = backup_dir / filename
    if path.is_file():
        path_tmp: Path = backup_dir / (filename + ".tmp")
        print(
            f"Warning: the file '{path}' already exists on disk! I will save to "
            f"'{path_tmp}' instead to avoid overwriting the file. "
            "Make sure to secure this file as it might be overwriten in a next run!"
        )
        return path_tmp
    return path


def backup(
    backup_dir: Path,
    raw_circuits: ty.List[QuantumCircuit],
    backend,
    basis: BaseMeasurementBasis,
    hub: str,
    group: str,
    project: str,
    job: ManagedJobSet,
) -> None:
    backup_filename: Path = get_backup_filename(
        backup_dir, raw_circuits, backend, basis, job
    )
    print(f"Backing up in '{backup_filename}'.")
    with open(backup_filename, "wb") as f:
        pickle.dump(
            {
                "job_set_id": job.job_set_id(),
                "raw_circuits": raw_circuits,
                "basis": basis,
                "backend_name": backend.name(),
                "provider": {"hub": hub, "group": group, "project": project},
            },
            f,
        )


def get_basis(name: str) -> BaseMeasurementBasis:
    return _BASIS[name]


def get_backend(hub: str, group: str, project: str, backend_name: str):
    if not IBMQ.active_account():
        print("Loading IBMQ account, this might take some time...")
        IBMQ.load_account()
    provider: AccountProvider = IBMQ.get_provider(hub=hub, group=group, project=project)
    potential_backends: ty.List = provider.backends(name=backend_name)
    if len(potential_backends) == 0:
        raise RuntimeError(
            f"No backend found with name '{backend_name}'. "
            "Check that 1) you did not mispelled the backend name "
            "and 2) the provider you used has access to this backend."
        )
    if len(potential_backends) > 1:
        raise RuntimeError(
            f"More than one backend found with name '{backend_name}': "
            + "["
            + ", ".join([b.name() for b in potential_backends])
            + "]"
        )

    return potential_backends[0]


def wait_for_job(managed_jobs: ty.Union[AerJob, AerJobSet, ManagedJobSet]) -> None:
    if isinstance(managed_jobs, ManagedJobSet):
        print(f"Waiting for IBMQ job with ID '{managed_jobs.job_set_id()}'...")
        print(
            "Note: you can safely terminate this program (via Ctrl-C or any other mean) "
            "and wait for the job to finish. You will then be able to recover the tomography "
            "results with the backup file printed above."
        )
        try:
            for i, job in enumerate(managed_jobs.jobs()):
                if job is None:
                    print("One job failed to be submitted...")
                    continue
                print(f"Waiting for the completion of job {i}")
                job.wait_for_final_state()
        except KeyboardInterrupt:
            print(
                "Interupting this program, the quantum circuit execution is still in "
                "progress on IBM backend."
            )
    else:
        raise NotImplementedError(
            "This should not happen: only ManagedJobSet should be here."
        )


def submit_circuits(
    circuits: ty.List[QuantumCircuit], backend, rep_delay: ty.Optional[float]
) -> ManagedJobSet:
    print(f"Compiling the {len(circuits)} circuits that will be submitted.")
    compiled_circuits: ty.List[QuantumCircuit] = compile_circuits(circuits)
    print(f"Submitting '{len(compiled_circuits)}' circuits.")
    tags = ["tomography", "bloch"]
    if rep_delay is not None:
        tags.append(f"rep_delay={rep_delay}")
    job = submit(compiled_circuits, backend, tags=tags, rep_delay=rep_delay)
    if isinstance(job, (AerJob, AerJobSet)):
        raise NotImplementedError(
            "This should not happen: job should be a ManagedJobSet."
        )
    return job


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Execute the circuits to perform state tomography for "
            "approximately uniformly distributed quantum states over "
            "the Bloch sphere."
        )
    )
    parser.add_argument("backend", type=str, help="Backend to perform tomography on.")
    parser.add_argument(
        "approximate_point_number",
        type=int,
        help=(
            "Approximate number of points used to cover the Bloch sphere. "
            "The actual number of points used might vary a little bit from this value. "
            "Each point will be tomographied independently."
        ),
    )
    parser.add_argument(
        "basis",
        type=str,
        choices=set(_BASIS.keys()),
        help="Name of the tomography basis used to perform quantum state tomography.",
    )
    parser.add_argument(
        "--hub",
        type=str,
        default="ibm-q",
        help="Hub of your IBMQ provider. Defaults to 'ibm-q' available to all users.",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="open",
        help="Group of your IBMQ provider. Defaults to 'open' available to all users.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="main",
        help="Project of your IBMQ provider. Defaults to 'main' available to all users.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=_DEFAULT_SAVE,
        help="Directory used to save the data needed to post-process job results.",
    )
    parser.add_argument(
        "--rep-delay",
        type=float,
        default=None,
        help="Delay between each shot. Default to the backend default value.",
    )

    args = parser.parse_args()

    backend = get_backend(args.hub, args.group, args.project, args.backend)
    qubit_number: int = backend.configuration().num_qubits
    basis: BaseMeasurementBasis = get_basis(args.basis)
    print(f"Using backend '{backend.name()}'.")
    print(f"Using basis   '{basis.name}'.")

    circuits: ty.List[QuantumCircuit] = get_approximately_equidistant_circuits(
        args.approximate_point_number
    )
    tomography_circuits: ty.List[QuantumCircuit] = list(
        it.chain(
            *[
                one_qubit_tomography_circuits(c, basis=basis, qubit_number=qubit_number)
                for c in circuits
            ]
        )
    )
    job = submit_circuits(tomography_circuits, backend, args.rep_delay)
    backup(
        args.backup_dir,
        circuits,
        backend,
        basis,
        args.hub,
        args.group,
        args.project,
        job,
    )
    wait_for_job(job)
