import typing as ty
import argparse
import pickle
from pathlib import Path

import numpy

from qiskit import QuantumCircuit, IBMQ
from qiskit.providers.aer.jobs.aerjob import AerJob
from qiskit.providers.aer.jobs.aerjobset import AerJobSet
from qiskit.providers.ibmq.managed.managedresults import ManagedResults
from qiskit.result import Result
from qiskit.providers.ibmq.managed import IBMQJobManager, ManagedJobSet

from sqt.basis.base import BaseMeasurementBasis

from sqt.fit.grad import post_process_tomography_results_grad
from sqt.fit.lssr import post_process_tomography_results_lssr
from sqt.fit.mle import post_process_tomography_results_mle
from sqt.fit.pauli import post_process_tomography_results_pauli

_POST_PROCESSING = {
    "grad": post_process_tomography_results_grad,
    "lssr": post_process_tomography_results_lssr,
    "mle": post_process_tomography_results_mle,
    "pauli": post_process_tomography_results_pauli,
}


def unpack_data(
    backup_filename: Path,
) -> ty.Tuple[
    str,
    ty.List[QuantumCircuit],
    BaseMeasurementBasis,
    str,
    ty.Dict[str, str],
    int,
    Result,
    int,
]:
    print(f"Recovering data from '{backup_filename}'.")
    with open(backup_filename, "rb") as f:
        data = pickle.load(f)
    return (
        data["job_set_id"],
        data["raw_circuits"],
        data["basis"],
        data["backend_name"],
        data["provider"],
        data["qubit_number"],
        data["result"],
        data["shots"],
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Post-process the job result and compute the reconstructed density matrices."
        )
    )
    parser.add_argument(
        "backup_filepath",
        type=Path,
        help="Backup file path that has been saved during the job submission.",
    )
    parser.add_argument(
        "post_processing_method",
        type=str,
        choices=set(_POST_PROCESSING.keys()),
        help="Post-processing method used to reconstruct the density matrices.",
        nargs="+",
    )
    # parser.add_argument(
    #     "-p",
    #     "--processors",
    #     type=int,
    #     default=None,
    #     help="Number of process in parallel to use. Default to the number of processors available.",
    # )
    args = parser.parse_args()

    (
        job_set_id,
        raw_circuits,
        basis,
        backend_name,
        provider_data,
        qubit_number,
        result,
        shots,
    ) = unpack_data(args.backup_filepath)

    results: Result = result
    if result is None:
        if not IBMQ.active_account():
            print("Loading IBMQ account, this might take some time...")
            IBMQ.load_account()
        print("Recovering provider and backend data...")
        provider = IBMQ.get_provider(**provider_data)
        print(f"Recovering results from job '{job_set_id}'...")
        job_manager = IBMQJobManager()
        job_set: ManagedJobSet = job_manager.retrieve_job_set(job_set_id, provider)
        managed_result: ManagedResults = job_set.results()
        results = managed_result.combine_results()

    for post_processing_method_name in args.post_processing_method:
        print(f"Starting post-processing with '{post_processing_method_name}' method!")
        post_processing_method: ty.Callable[
            [Result, QuantumCircuit, BaseMeasurementBasis, int], ty.List[numpy.ndarray]
        ] = _POST_PROCESSING[post_processing_method_name]
        # data[qubit_index][i] = (point, density_matrix)
        data: ty.List[ty.List[ty.Tuple[numpy.ndarray, numpy.ndarray]]] = [
            list() for _ in range(qubit_number)
        ]
        # Might be parallelised.
        for circuit in raw_circuits:
            point: numpy.ndarray = numpy.array(eval(circuit.name))  # Quite bad...
            density_matrices: ty.List[numpy.ndarray] = post_processing_method(
                results, circuit, basis, qubit_number
            )
            for i in range(qubit_number):
                data[i].append((point, density_matrices[i]))

        backup_filename: str = args.backup_filepath.name
        backup_dirpath: Path = args.backup_filepath.parent
        post_process_backup_filename: str = backup_filename.replace(
            ".pkl", f".post_process_{post_processing_method_name}.pkl"
        )
        post_process_filepath: Path = backup_dirpath / post_process_backup_filename
        print(f"Saving post-processed data to '{post_process_filepath}'")
        with open(post_process_filepath, "wb") as f:
            pickle.dump(
                {
                    "qubit_number": qubit_number,
                    "density_matrices": data,
                    "backend_name": backend_name,
                    "basis_name": basis.name,
                    "provider": provider_data,
                    "post_processing_method": post_processing_method_name,
                    "shots": shots,
                },
                f,
            )
