# mocking.py: stand-ins for external services, to test and develop without them

from pathlib import Path
from typing import Any


def _cost(request: dict[str, Any]) -> float:
    """Toy cost of a request: years x months x days x hours x variables

    Grows with the size of the request.
    """
    return float(len(request["year"]) * len(request["month"]) * len(request["day"])
                 * len(request["time"]) * len(request["variable"]))


class _FakeDatastores:
    """Stands in for ``cdsapi.Client().client``"""

    def __init__(self, limit: float) -> None:
        self.limit = limit

    def estimate_costs(self, collection_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """Same form of answer as the CDS costing endpoint, with the toy cost"""
        return {"id": "size", "cost": _cost(request), "limit": self.limit}


class FakeResult:
    """Stands in for the result of ``cdsapi.Client().retrieve``"""

    def download(self, target: str | Path) -> str:
        """Write an empty file at ``target`` and return its path"""
        Path(target).touch()
        return str(target)


class FakeCDSClient:
    """Stands in for ``cdsapi.Client``, without network

    Covers what the CDS source uses: ``client.estimate_costs`` and ``retrieve``. The cost of a
    request is fake, years x months x days x hours x variables. The files downloaded are
    empty.

    Parameters
    ----------
    limit: float, default 1000
        Cost limit.

    Attributes
    ----------
    sent: list of dict
        Requests received by ``retrieve``, refused ones included.
    """

    def __init__(self, limit: float = 1000.0) -> None:
        self.client = _FakeDatastores(limit)
        self.sent: list[dict[str, Any]] = []

    def retrieve(self, name: str, request: dict[str, Any]) -> FakeResult:
        """Same as ``cdsapi.Client.retrieve``

        Raises
        ------
        requests.HTTPError
            403, with the message of CDS, if the toy cost of the request exceeds the limit.
        """
        self.sent.append(request)
        if _cost(request) > self.client.limit:
            import requests
            response = requests.Response()
            response.status_code = 403
            raise requests.HTTPError(
                "403 Client Error: Forbidden\n"
                "cost limits exceeded\n"
                "Your request is too large, please reduce your selection.",
                response=response,
            )
        return FakeResult()
