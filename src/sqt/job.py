from __future__ import annotations

import typing as ty
from abc import ABC, abstractmethod

from qiskit.result import Result
from qiskit_aer.jobs import AerJob, AerJobSet
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeJob


class BaseJob(ABC):
    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def result(self) -> Result:
        pass

    @abstractmethod
    def to_dict(self) -> ty.Mapping[str, ty.Any]:
        pass

    @abstractmethod
    def wait_for_completion(self) -> None:
        pass

    @staticmethod
    def from_dict(d: ty.Mapping[str, ty.Any]) -> "BaseJob":
        if "id" in d:
            return RemoteJob.from_dict(d)
        else:
            return LocalJob.from_dict(d)

    @staticmethod
    def from_job(
        job: RuntimeJob | AerJob | AerJobSet, hub: str, group: str, project: str
    ) -> "BaseJob":
        if isinstance(job, RuntimeJob):
            return BaseJob._from_ibmq_runtime_job(job, hub, group, project)
        elif isinstance(job, (AerJob, AerJobSet)):
            return BaseJob._from_aer_job(job)
        else:
            raise RuntimeError(f"Unsupported job type: {type(job)}")

    @staticmethod
    def _from_ibmq_runtime_job(
        job: RuntimeJob, hub: str, group: str, project: str
    ) -> "RemoteJob":
        return RemoteJob(job.job_id(), hub, group, project)

    @staticmethod
    def _from_aer_job(job: AerJob | AerJobSet) -> "LocalJob":
        return LocalJob(job.result())


class LocalJob(BaseJob):
    def __init__(self, result: Result) -> None:
        super().__init__()

        self._result = result

    def id(self) -> str:
        return self._result.job_id

    def result(self):
        return self._result

    def to_dict(self) -> ty.Mapping[str, ty.Any]:
        return {"result": self._result.to_dict()}

    @staticmethod
    def from_dict(d: ty.Mapping[str, ty.Any]) -> "LocalJob":
        return LocalJob(Result.from_dict(d["result"]))

    def wait_for_completion(self) -> None:
        return


class RemoteJob(BaseJob):
    def __init__(self, id: str, hub: str, group: str, project: str) -> None:
        super().__init__()
        self._id = id
        self._hub = hub
        self._group = group
        self._project = project
        self._service = None

    def id(self) -> str:
        return self._id

    def result(self) -> Result:
        return self._get_service().job(self.id()).result()

    def _get_service(self) -> QiskitRuntimeService:
        if self._service is None:
            self._service = QiskitRuntimeService(
                channel="ibm_quantum",
                instance=f"{self._hub}/{self._group}/{self._project}",
            )
            if not self._service.active_account():
                raise RuntimeError(
                    f"Could not load account with '{self._hub}' '{self._group}' '{self._project}'."
                )
        return self._service

    def to_dict(self) -> ty.Mapping[str, ty.Any]:
        return {
            "id": self._id,
            "hub": self._hub,
            "group": self._group,
            "project": self._project,
        }

    @staticmethod
    def from_dict(d: ty.Mapping[str, ty.Any]) -> "RemoteJob":
        return RemoteJob(d["id"], d["hub"], d["group"], d["project"])

    def wait_for_completion(self) -> None:
        self._get_service().job(self.id()).wait_for_final_state()
