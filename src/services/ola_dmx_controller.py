import array
import time
from ola.ClientWrapper import ClientWrapper

# Default universe
UNIVERSE = 1
wrapper = None

def DmxSent(state):
    if not state.Succeeded():
        print("Failed to send DMX")
        wrapper.Stop()

def _coalesce_identical_frames(frame_delays_ms, dmx_frames):
    """
    Merge consecutive identical frames by summing their delays.
    This can drastically reduce SendDmx call count when many frames don't change.
    """
    if not dmx_frames:
        return frame_delays_ms, dmx_frames

    new_delays = []
    new_frames = []

    prev = dmx_frames[0]
    acc_delay = int(frame_delays_ms[0]) if frame_delays_ms else 0

    for i in range(1, len(dmx_frames)):
        cur = dmx_frames[i]
        d = int(frame_delays_ms[i]) if i < len(frame_delays_ms) else 0

        if cur == prev:
            acc_delay += d
        else:
            new_frames.append(prev)
            new_delays.append(acc_delay)
            prev = cur
            acc_delay = d

    new_frames.append(prev)
    new_delays.append(acc_delay)
    return new_delays, new_frames

def play_dmx_sequence(frame_delays_ms, dmx_frames, universe=UNIVERSE):
    """
    Play a DMX frame sequence using per-frame delays.

    Uses streaming scheduling (only schedules the next event) and compensates drift
    against an absolute start time to prevent cumulative lag and reduce event-queue load.
    """
    global wrapper
    print(f"----Playing DMX sequence with OLA on universe {universe}")

    if not dmx_frames:
        return

    # Reduce load
    frame_delays_ms, dmx_frames = _coalesce_identical_frames(frame_delays_ms, dmx_frames)

    wrapper = ClientWrapper()

    i = 0
    cumulative_ms = 0
    start_t = time.monotonic()

    def send_next():
        nonlocal i, cumulative_ms

        if i >= len(dmx_frames):
            wrapper.Stop()
            return

        wrapper.Client().SendDmx(universe, dmx_frames[i], DmxSent)

        delay_ms = frame_delays_ms[i] if i < len(frame_delays_ms) else 33
        cumulative_ms += int(delay_ms)
        i += 1

        next_due = start_t + (cumulative_ms / 1000.0)
        wait_ms = int(max(0.0, (next_due - time.monotonic()) * 1000.0))
        wrapper.AddEvent(wait_ms, send_next)

    wrapper.AddEvent(0, send_next)
    wrapper.Run()