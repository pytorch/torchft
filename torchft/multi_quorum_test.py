"""
Validate that one Lighthouse server can host isolated quorums
for multiple logical rooms (job IDs) via `room-id` metadata header.
"""

from __future__ import annotations

import datetime as _dt

import pytest

import torchft._torchft as ext

_TIMEOUT = _dt.timedelta(seconds=3)  # connect + RPC timeout


def _client(addr: str, room: str) -> ext.LighthouseClient:
    """Utility: create a client with a logical room-id."""
    return ext.LighthouseClient(addr, _TIMEOUT, room)


@pytest.mark.asyncio
async def test_multi_room_quorums() -> None:
    # 1) one server, any free port
    server = ext.LighthouseServer("[::]:0", 1)
    addr = server.address()

    # 2) two clients in two separate rooms
    a = _client(addr, "jobA")
    b = _client(addr, "jobB")

    # 3) explicit heartbeats (exercises RPC path)
    a.heartbeat("a0")
    b.heartbeat("b0")

    # 4) ask for a quorum from each room
    qa = a.quorum("a0", _TIMEOUT)
    qb = b.quorum("b0", _TIMEOUT)

    # 5) verify the rooms are independent
    assert qa.quorum_id == qb.quorum_id == 1
    assert len(qa.participants) == 1 and qa.participants[0].replica_id == "a0"
    assert len(qb.participants) == 1 and qb.participants[0].replica_id == "b0"

    # 6) shutdown
    server.shutdown()
