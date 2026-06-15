# SPDX-License-Identifier: Apache-2.0
"""myelon SHM-ring carrier for the inter-stage control plane.

This module is the ``OMNI_CONTROL_TRANSPORT=myelon`` backend for the four
control-plane socket classes in :mod:`sglang_omni.pipeline.control_plane`.
It swaps **only the carrier** under ``send`` / ``recv`` / ``publish`` /
``poll``: the msgpack codec (``serialize_message`` / ``deserialize_message``)
is reused verbatim and produces the byte-identical payload that rides in
``frame[2]`` of the multipart envelope.

Topology mapping (linear/DAG pipelines):

* **requests** (ZMQ PUSH->PULL bind, fan-in N->1): one 1p1c SHM ring per
  ``(producer -> consumer-endpoint)`` edge. The producer *creates* the ring
  (myelon's ``SendSocket`` is the segment owner); the consumer attaches to
  every inbound ring and round-robin drains. Late-joining producers are
  discovered via a per-run filesystem rendezvous directory.
* **responses** (ZMQ PUSH->PULL bind, fan-in N->1): same shape, one ring per
  ``(stage -> coordinator)`` edge.
* **abort** (ZMQ PUB->SUB fan-out 1->N): one 1p1c ring per subscriber; the
  publisher loops ``send_multipart`` to each.

The data plane (``relay_io.py`` / ``relay/``) is **not** touched: ``relay_info``
stays an opaque msgpack field inside ``frame[2]``.

SHM sizing note: each ring is ``slots * FRAME_BYTES_SHM`` (2 MiB/slot). On the
csm container ``/dev/shm`` is 64 MiB, so the default ring depth is small
(``OMNI_MYELON_RING_SLOTS``, default 2 -> 4 MiB/ring). Control messages are
small and low-rate, so shallow rings are correct here. We stay on the **SHM**
backend (not the 16 MiB MMAP, which has a stack-overflow trap and is the wrong
latency regime for sub-KB control traffic).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .control_plane import deserialize_message, serialize_message

if TYPE_CHECKING:
    from .control_plane import AbortMessage, ControlMessage

logger = logging.getLogger(__name__)

# Imported lazily so the default OMNI_CONTROL_TRANSPORT=zmq path does not
# require the myelon_zmq_py extension to be installed.
_myelon = None


def _myelon_mod():
    global _myelon
    if _myelon is None:
        import myelon_zmq_py as _m

        _myelon = _m
    return _myelon

# --- envelope -------------------------------------------------------------

WIRE_VERSION = b"\x01"

# 1-byte type tags mirroring proto.messages.parse_message's d["type"] switch.
_TYPE_TAGS: dict[str, bytes] = {
    "submit": b"\x01",
    "data_ready": b"\x02",
    "admin": b"\x03",
    "shutdown": b"\x04",
    "profiler_start": b"\x05",
    "profiler_stop": b"\x06",
    "complete": b"\x10",
    "stream": b"\x11",
    "admin_result": b"\x12",
    "abort": b"\x20",
}


def _type_tag(msg: "ControlMessage") -> bytes:
    # to_dict()["type"] is the single source of truth for the wire tag; falling
    # back to the class name keeps the tag cheap without unpacking msgpack on
    # the recv side.
    t = msg.to_dict().get("type", "")
    return _TYPE_TAGS.get(t, b"\x00")


def encode_envelope(msg: "ControlMessage") -> list[bytes]:
    """3-frame envelope: [version, type-tag, msgpack(to_dict())].

    frame[2] is BYTE-IDENTICAL to the ZMQ path's serialize_message(msg).
    """
    payload = serialize_message(msg)
    return [WIRE_VERSION, _type_tag(msg), payload]


def decode_envelope(frames: list[bytes]) -> "ControlMessage":
    """Inverse of encode_envelope. Only frame[2] feeds the unchanged codec."""
    if not frames:
        raise ValueError("empty myelon multipart frame")
    if len(frames) == 1:
        # Defensive: a future N=1 fast-path send would carry only the payload.
        return deserialize_message(frames[0])
    return deserialize_message(frames[-1])


# --- tuning ---------------------------------------------------------------


def control_transport() -> str:
    """Read the carrier flag. Default ``zmq``; fail-loud on unknown values."""
    v = os.environ.get("OMNI_CONTROL_TRANSPORT", "zmq").strip().lower()
    if v not in ("zmq", "myelon"):
        raise ValueError(
            f"OMNI_CONTROL_TRANSPORT must be 'zmq' or 'myelon', got {v!r}"
        )
    return v


def use_myelon() -> bool:
    return control_transport() == "myelon"


def _ring_slots() -> int:
    return max(2, int(os.environ.get("OMNI_MYELON_RING_SLOTS", "2")))


def _drain_slice_us() -> int:
    # A small slice of the 80 ms frame clock; the drain thread parks
    # (GIL released) and re-checks. Far below the frame budget.
    return max(100, int(os.environ.get("OMNI_MYELON_DRAIN_SLICE_US", "2000")))


def _drain_batch() -> int:
    return max(1, int(os.environ.get("OMNI_MYELON_DRAIN_BATCH", "32")))


# --- segment naming -------------------------------------------------------
#
# The disruptor-mp SHM segment-name budget is 14 chars (PSHMNAMLEN-portable),
# enforced even on Linux. Both producer and consumer must independently derive
# the SAME segment name for an edge. We hash the consumer endpoint (which both
# sides know) + a role byte + a producer token; the per-run IPC tempdir nonce
# is already inside the endpoint string, so names are per-run unique.

_SHM_NAME_MAX = 14
_RENDEZVOUS_PREFIX = "_myelon_edges"


def mint_segment_name(consumer_endpoint: str, role: str, producer_token: str) -> str:
    """Deterministic <=14-char SHM segment name for one ring edge.

    ``role`` is one of ``r`` (requests), ``s`` (responses), ``a`` (abort).
    Layout: ``o`` + role + 11 base36 hash chars  (<= 14 total).
    """
    if role not in ("r", "s", "a"):
        raise ValueError(f"bad ring role {role!r}")
    h = hashlib.blake2b(
        f"{role}|{consumer_endpoint}|{producer_token}".encode(), digest_size=8
    ).digest()
    n = int.from_bytes(h, "big")
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = []
    for _ in range(11):
        n, r = divmod(n, 36)
        out.append(alphabet[r])
    name = "o" + role + "".join(out)
    if len(name) > _SHM_NAME_MAX:
        raise ValueError(
            f"myelon segment name {name!r} exceeds {_SHM_NAME_MAX}-char SHM budget"
        )
    return name


def _consumer_id(consumer_endpoint: str, role: str) -> str:
    h = hashlib.blake2b(
        f"{role}|{consumer_endpoint}".encode(), digest_size=4
    ).hexdigest()
    return f"c{h}"


def _rendezvous_dir(endpoint: str) -> Path:
    """Per-run rendezvous dir derived from the ipc:// endpoint's base dir.

    Producers drop a marker file naming the ring they created toward a
    consumer endpoint; the consumer scans for markers to discover all
    inbound rings (handles fan-in + late joiners) without pre-knowing
    producer identities.
    """
    # endpoint looks like ipc:///tmp/.../<name>.sock
    path = endpoint[len("ipc://") :] if endpoint.startswith("ipc://") else endpoint
    base = Path(path).parent
    d = base / _RENDEZVOUS_PREFIX
    d.mkdir(parents=True, exist_ok=True)
    return d


def _edge_token() -> str:
    """Process-unique producer token (pid + monotonic counter)."""
    with _token_lock:
        global _token_counter
        _token_counter += 1
        return f"{os.getpid()}-{_token_counter}"


_token_lock = threading.Lock()
_token_counter = 0


# --- shared cleanup -------------------------------------------------------
#
# myelon SHM segments are POSIX shm_open objects that are not unlinked on drop.
# Track every segment this process creates and unlink on close so a crashed /
# restarted run does not leak /dev/shm (which is only 64 MiB here).
_owned_segments: set[str] = set()
_owned_lock = threading.Lock()


def _track_segment(name: str) -> None:
    with _owned_lock:
        _owned_segments.add(name)


def _unlink_segment(name: str) -> None:
    # disruptor-mp creates the main segment plus coordination cursors named
    # with assorted suffixes (<name>_ci, <name>_cr, <name>_producer_seq, and
    # <name>_<consumer_id>_seq). Glob the prefix so none leak in /dev/shm.
    shm = Path("/dev/shm")
    try:
        for p in shm.glob(f"{name}*"):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:  # pragma: no cover - best effort
                logger.debug("could not unlink shm %s: %s", p, exc)
    except OSError:
        pass


def cleanup_owned_segments() -> None:
    with _owned_lock:
        names = list(_owned_segments)
        _owned_segments.clear()
    for n in names:
        _unlink_segment(n)


# --- send side (PUSH / one abort subscriber) ------------------------------


class _RingProducer:
    """One myelon SendSocket toward a consumer endpoint, plus rendezvous."""

    def __init__(self, consumer_endpoint: str, role: str):
        self.consumer_endpoint = consumer_endpoint
        self.role = role
        self.token = _edge_token()
        self.segment = mint_segment_name(consumer_endpoint, role, self.token)
        self.consumer_id = _consumer_id(consumer_endpoint, role)
        slots = _ring_slots()
        self._sock = _myelon_mod().SendSocket(self.segment, slots, self.consumer_id)
        _track_segment(self.segment)
        # Advertise this ring so the consumer can discover + attach it. The
        # marker name carries the target consumer_id so a consumer only
        # attaches rings minted for *it* (matters on the abort fan-out, where
        # many per-subscriber rings share one rendezvous dir).
        self._marker = (
            _rendezvous_dir(consumer_endpoint)
            / f"{role}-{self.consumer_id}-{self.segment}"
        )
        try:
            self._marker.write_text(self.consumer_id)
        except OSError as exc:  # pragma: no cover
            logger.warning("rendezvous marker write failed: %s", exc)
        # Give the consumer a moment to attach, then register it with the
        # producer's backpressure barrier (discovery-race close).
        self._sock.warm_discovery(int(os.environ.get("OMNI_MYELON_WARM_MS", "3000")))
        logger.debug(
            "myelon producer ring %s (role=%s -> %s) ready",
            self.segment,
            role,
            consumer_endpoint,
        )

    def send_frames(self, frames: list[bytes]) -> None:
        self._sock.send_multipart(frames)

    def close(self) -> None:
        try:
            self._marker.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        _unlink_segment(self.segment)


# --- recv side (PULL / SUB) -----------------------------------------------


class _RingConsumer:
    """Drain all inbound rings for one consumer endpoint into an asyncio.Queue.

    A single dedicated thread round-robins across every attached ring, draining
    each in one ``recv_multipart_drain_spin`` call (GIL-released) and posting
    whole batches via ``loop.call_soon_threadsafe`` -> one cross-thread hop per
    batch. New producer rings are discovered by re-scanning the rendezvous dir.
    """

    def __init__(self, consumer_endpoint: str, role: str):
        self.consumer_endpoint = consumer_endpoint
        self.role = role
        self.consumer_id = _consumer_id(consumer_endpoint, role)
        self.slots = _ring_slots()
        self._rendezvous = _rendezvous_dir(consumer_endpoint)
        self._queue: asyncio.Queue[list[bytes]] = asyncio.Queue()
        self._loop = asyncio.get_running_loop()
        self._stop = threading.Event()
        self._recvs: dict[str, object] = {}  # segment -> RecvSocket
        self._thread = threading.Thread(
            target=self._drain_loop,
            name=f"myelon-drain-{role}-{self.consumer_id}",
            daemon=True,
        )
        self._thread.start()

    # -- discovery --
    def _scan_and_attach(self) -> None:
        try:
            markers = list(self._rendezvous.iterdir())
        except OSError:
            return
        # Markers are named "<role>-<consumer_id>-<segment>"; only attach the
        # rings minted for this consumer_id.
        prefix = f"{self.role}-{self.consumer_id}-"
        for m in markers:
            if not m.name.startswith(prefix):
                continue
            segment = m.name[len(prefix) :]
            if segment in self._recvs:
                continue
            try:
                recv = _myelon_mod().RecvSocket(segment, self.slots, self.consumer_id)
            except Exception as exc:  # segment not yet created by producer
                logger.debug("attach %s pending: %s", segment, exc)
                continue
            self._recvs[segment] = recv
            logger.debug(
                "myelon consumer attached ring %s (role=%s)", segment, self.role
            )

    # -- drain --
    def _drain_loop(self) -> None:
        slice_us = _drain_slice_us()
        batch = _drain_batch()
        last_scan = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if not self._recvs or now - last_scan > 0.05:
                self._scan_and_attach()
                last_scan = now
            if not self._recvs:
                time.sleep(0.002)
                continue
            got_any = False
            for recv in list(self._recvs.values()):
                try:
                    msgs = recv.recv_multipart_drain_spin(slice_us, batch)
                except Exception as exc:  # pragma: no cover
                    logger.error("myelon drain error: %s", exc)
                    continue
                if msgs:
                    got_any = True
                    self._loop.call_soon_threadsafe(self._post_batch, msgs)
            if not got_any:
                # All rings empty this pass; the per-ring spin already waited
                # one slice, so just loop (cheap try-recvs) without busy-burning.
                pass

    def _post_batch(self, msgs: list[list[bytes]]) -> None:
        for frames in msgs:
            self._queue.put_nowait(frames)

    # -- async surface --
    async def recv_msg(self) -> "ControlMessage":
        frames = await self._queue.get()
        return decode_envelope(frames)

    def recv_msg_nowait(self) -> "ControlMessage | None":
        try:
            frames = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        return decode_envelope(frames)

    def has_pending(self) -> bool:
        return not self._queue.empty()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._recvs.clear()


# --- socket-class backends (mirror the pyzmq surface) ---------------------


class MyelonPushSocket:
    """myelon backend for PushSocket (one ring toward `endpoint`)."""

    def __init__(self, endpoint: str, role: str = "r"):
        self.endpoint = endpoint
        self.role = role
        self._producer: _RingProducer | None = None

    async def connect(self) -> None:
        self._producer = _RingProducer(self.endpoint, self.role)

    async def send(self, msg: "ControlMessage") -> None:
        if self._producer is None:
            raise RuntimeError("Socket not connected")
        self._producer.send_frames(encode_envelope(msg))

    def close(self) -> None:
        if self._producer is not None:
            self._producer.close()
            self._producer = None


class MyelonPullSocket:
    """myelon backend for PullSocket (drain all inbound rings on `endpoint`)."""

    def __init__(self, endpoint: str, role: str = "r"):
        self.endpoint = endpoint
        self.role = role
        self._consumer: _RingConsumer | None = None

    async def start(self) -> None:
        self._consumer = _RingConsumer(self.endpoint, self.role)

    async def recv(self) -> "ControlMessage":
        if self._consumer is None:
            raise RuntimeError("Socket not started")
        return await self._consumer.recv_msg()

    async def recv_nowait(self) -> "ControlMessage | None":
        if self._consumer is None:
            raise RuntimeError("Socket not started")
        return self._consumer.recv_msg_nowait()

    def close(self) -> None:
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None


class MyelonPubSocket:
    """myelon backend for PubSocket: one producer ring per subscriber.

    Subscriber rings are discovered lazily via the rendezvous dir under the
    abort endpoint: each SubSocket creates a *reverse* marker the publisher
    scans, and the publisher creates one ring per discovered subscriber.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._producers: dict[str, _RingProducer] = {}  # sub_id -> producer
        self._rendezvous: Path | None = None

    async def bind(self) -> None:
        self._rendezvous = _rendezvous_dir(self.endpoint)
        # Give subscribers a beat to register (parity with the ZMQ slow-joiner
        # sleep in PubSocket.bind).
        await asyncio.sleep(0.1)

    def _refresh_subscribers(self) -> None:
        if self._rendezvous is None:
            return
        try:
            markers = list(self._rendezvous.iterdir())
        except OSError:
            return
        for m in markers:
            if not m.name.startswith("sub-"):
                continue
            sub_id = m.name[len("sub-") :]
            if sub_id in self._producers:
                continue
            # One abort ring per subscriber, keyed by the subscriber id so each
            # subscriber's RecvSocket attaches to its own segment.
            self._producers[sub_id] = _RingProducer(
                f"{self.endpoint}#{sub_id}", "a"
            )

    async def publish(self, msg: "AbortMessage") -> None:
        self._refresh_subscribers()
        frames = encode_envelope(msg)
        for prod in self._producers.values():
            prod.send_frames(frames)

    def close(self) -> None:
        for prod in self._producers.values():
            prod.close()
        self._producers.clear()


class MyelonSubSocket:
    """myelon backend for SubSocket: one inbound abort ring for this stage."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.sub_id = _consumer_id(endpoint, "a") + f"-{os.getpid()}"
        self._consumer: _RingConsumer | None = None
        self._marker: Path | None = None

    async def connect(self) -> None:
        # Register this subscriber so the publisher mints a ring toward us.
        d = _rendezvous_dir(self.endpoint)
        self._marker = d / f"sub-{self.sub_id}"
        try:
            self._marker.write_text("1")
        except OSError as exc:  # pragma: no cover
            logger.warning("sub marker write failed: %s", exc)
        # Our inbound ring lives under the per-subscriber endpoint the
        # publisher uses to mint the producer side.
        self._consumer = _RingConsumer(f"{self.endpoint}#{self.sub_id}", "a")

    async def recv(self) -> "AbortMessage":
        if self._consumer is None:
            raise RuntimeError("Socket not connected")
        return await self._consumer.recv_msg()  # type: ignore[return-value]

    def poll(self, timeout_ms: int = 0) -> bool:
        if self._consumer is None:
            raise RuntimeError("Socket not connected")
        if self._consumer.has_pending():
            return True
        if timeout_ms > 0:
            deadline = time.monotonic() + timeout_ms / 1000.0
            while time.monotonic() < deadline:
                if self._consumer.has_pending():
                    return True
                time.sleep(0.001)
        return False

    def close(self) -> None:
        if self._marker is not None:
            try:
                self._marker.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None
