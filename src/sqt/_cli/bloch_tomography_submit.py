import argparse
import itertools as it
import pickle
from datetime import datetime
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.providers.backend import BackendV2 as Backend
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import RuntimeJobV2 as RuntimeJob
from rich import print

from sqt.basis.base import BaseMeasurementBasis
from sqt.basis.equidistant import (
    EquidistantMeasurementBasis,
    get_approximately_equidistant_circuits,
)
from sqt.basis.pauli import PauliMeasurementBasis
from sqt.basis.tetrahedral import TetrahedralMeasurementBasis
from sqt.circuits import one_qubit_tomography_circuits
from sqt.execution import submit
from sqt.passes import compile_circuits

_BASIS = {
    "pauli": PauliMeasurementBasis(),
    "tetrahedral": TetrahedralMeasurementBasis(),
    "equidistant": EquidistantMeasurementBasis,
}

_DEFAULT_SAVE = Path.cwd()


def get_backup_filename(
    backup_dir: Path,
    raw_circuits: list[QuantumCircuit],
    backend,
    basis: BaseMeasurementBasis,
    job: RuntimeJob,
    delay_dt,
    shots: int,
) -> Path:
    now = datetime.now()
    now_str: str = now.isoformat()
    jobid: str = job.job_id()
    backend_name: str = backend.name.strip("' ").replace("(", "-").replace(")", "-")
    filename: str = (
        f"backend-{backend_name}_basis-{basis.name}_points-{len(raw_circuits)}"
        f"_delay-{delay_dt}_shots-{shots}_{now_str}_jobid-{jobid}.pkl"
    )
    path: Path = backup_dir / filename
    if path.is_file():
        path_tmp: Path = backup_dir / (filename + ".tmp")
        print(
            f":warning: the file '{path}' already exists on disk! I will save to "
            f"'{path_tmp}' instead to avoid overwriting the file. "
            "Make sure to secure this file as it might be overwriten in a next run!",
        )
        return path_tmp
    return path


def backup(
    backup_dir: Path,
    raw_circuits: list[QuantumCircuit],
    backend,
    basis: BaseMeasurementBasis,
    hub: str,
    group: str,
    project: str,
    job: RuntimeJob,
    delay_dt: int,
    qubit_number: int,
    shots: int,
) -> None:
    backup_filename: Path = get_backup_filename(
        backup_dir, raw_circuits, backend, basis, job, delay_dt, shots
    )
    jobid: str = job.job_id()
    result = job.result()
    print(f"Backing up in '{backup_filename}'.")
    with open(backup_filename, "wb") as f:
        pickle.dump(
            {
                "job_id": jobid,
                "raw_circuits": raw_circuits,
                "basis": basis,
                "backend_name": backend.name,
                "provider": {"hub": hub, "group": group, "project": project},
                "delay_dt": delay_dt,
                "qubit_number": qubit_number,
                "result": result,
                "shots": shots,
            },
            f,
        )


def get_basis(name: str, equidistant_points: int | None) -> BaseMeasurementBasis:
    if name == "equidistant":
        if equidistant_points is None:
            raise RuntimeError(
                "The '--equidistant-points' option should be set with 'basis=equidistant'."
            )
        else:
            return _BASIS[name](equidistant_points)
    return _BASIS[name]


def _get_ibmq_backend(
    hub: str,
    group: str,
    project: str,
    backend_name: str,
) -> Backend:
    service = QiskitRuntimeService(
        channel="ibm_quantum", instance=f"{hub}/{group}/{project}"
    )
    if not service.active_account():
        raise RuntimeError(
            f"Could not load account with '{hub}' '{group}' '{project}'."
        )
    potential_backends: list = service.backends(name=backend_name)
    if len(potential_backends) == 0:
        print(f"[bold red]No backend found with name '{backend_name}'.[/bold red]")
        print("[bold orange]Check that:[/bold orange]")
        print("\t1. You did not mispelled the backend name.")
        print("\t2. The provider you used has access to this backend.")
        print(
            "[bold green]Possible backends:[/bold green]\n\t- ",
            "\n\t- ".join(b.name for b in service.backends()),
            sep="",
        )
        exit()
    if len(potential_backends) > 1:
        print(
            f"[bold orange]More than one backend found with name '{backend_name}':[/bold orange]"
        )
        print("\t- " + "\n\t- ".join(b.name for b in potential_backends))
        exit()

    return potential_backends[0]


def get_backend(
    hub: str,
    group: str,
    project: str,
    backend_name: str,
    local_backend: bool,
    noisy_simulator: bool,
) -> AerSimulator | Backend:
    if local_backend and not noisy_simulator:
        return AerSimulator(method="matrix_product_state")
    else:
        ibmq_backend = _get_ibmq_backend(hub, group, project, backend_name)
        if noisy_simulator:
            return AerSimulator.from_backend(ibmq_backend)
        else:
            return ibmq_backend


def wait_for_job(job: RuntimeJob) -> None:
    job.wait_for_final_state()
    print("[green]Job finished![/green]")


def submit_circuits(
    circuits: list[QuantumCircuit],
    backend: Backend | AerSimulator,
    rep_delay: float | None,
    shots: int,
    delay_dt: int,
) -> RuntimeJob:
    print(f"Compiling the {len(circuits)} circuits that will be submitted.")
    compiled_circuits: list[QuantumCircuit] = compile_circuits(circuits)
    print(f"Submitting {len(compiled_circuits)} circuits.")
    tags = ["tomography", "bloch", f"shots={shots}", f"delay={delay_dt}"]
    if rep_delay is not None:
        tags.append(f"rep_delay={rep_delay}")
    job = submit(
        compiled_circuits, backend, tags=tags, rep_delay=rep_delay, shots=shots
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
        choices=_BASIS.keys(),
        help="Name of the tomography basis used to perform quantum state tomography.",
    )
    parser.add_argument(
        "--equidistant-points",
        type=int,
        default=None,
        help="If basis is 'equidistant', the number of approximately equidistant projectors that will be used. Else, this option is ignored.",
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
    parser.add_argument(
        "--shots",
        type=int,
        default=20000,
        help="Number of shots performed for each circuit.",
    )
    parser.add_argument(
        "--delay-dt",
        type=int,
        default=0,
        help="Duration (in dt) of the delay to insert before state tomography.",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=None,
        help="Maximum number of qubits that should be used.",
    )
    parser.add_argument(
        "--local-backend",
        action="store_true",
        help="If present, the backend used is a local one.",
    )
    parser.add_argument(
        "--noisy-simulator",
        action="store_true",
        help="If present, the given IBMQ backend will be used to initialise a noisy simulator. Implies '--local-backend'.",
    )
    args = parser.parse_args()

    backend = get_backend(
        args.hub,
        args.group,
        args.project,
        args.backend,
        args.local_backend,
        args.noisy_simulator,
    )
    qubit_number: int = backend.num_qubits
    if args.max_qubits is not None:
        qubit_number = min(qubit_number, args.max_qubits)
    basis: BaseMeasurementBasis = get_basis(args.basis, args.equidistant_points)
    print(
        f"Using {qubit_number} qubit{'s' if qubit_number > 1 else ''} "
        f"from backend '{backend.name}'."
    )
    print(f"Using basis '{basis.name}'.")

    circuits: list[QuantumCircuit] = get_approximately_equidistant_circuits(
        args.approximate_point_number
    )
    if args.delay_dt > 0:
        if args.delay_dt % 32 != 0:
            print(
                f"[red]:warning:[/red] Delay duration {args.delay_dt} is not a multiple of 32. [orange]Expect issues.[/orange]"
            )
        for circuit in circuits:
            circuit.delay(args.delay_dt, 0, unit="dt")
    tomography_circuits: list[QuantumCircuit] = list(
        it.chain(
            *[
                one_qubit_tomography_circuits(c, basis=basis, qubit_number=qubit_number)
                for c in circuits
            ]
        )
    )
    shots: int = args.shots
    job = submit_circuits(
        tomography_circuits, backend, args.rep_delay, shots, args.delay_dt
    )
    backup(
        args.backup_dir,
        circuits,
        backend,
        basis,
        args.hub,
        args.group,
        args.project,
        job,
        args.delay_dt,
        qubit_number,
        shots,
    )
    if args.local_backend:
        wait_for_job(job)
