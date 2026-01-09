import array
from ola.ClientWrapper import ClientWrapper

# Default universe
UNIVERSE = 1
# Global wrapper instance used during playback; allows external stop.
wrapper = None

# Compat wrapper for old ola-python expecting .tostring()
class _OLAArrayCompat:
    def __init__(self, buf):
        self._buf = buf
    def tostring(self):
        try:
            return self._buf.tobytes()
        except AttributeError:
            return bytes(self._buf)

def DmxSent(state):
    if not state.Succeeded():
        print("Failed to send DMX")
        wrapper.Stop()

def stop_current_playback():
    """
    Stop the currently running DMX playback, if any.
    Safe to call even if nothing is playing.
    """
    global wrapper
    try:
        if wrapper is not None:
            try:
                wrapper.Stop()
                print("--DMX playback stopped")
            except Exception as e:
                print(f"--Error stopping DMX: {e}")
    except Exception:
        # Defensive: ignore any unexpected issues
        pass

def play_dmx_sequence(frame_delays_ms, dmx_frames, universe=UNIVERSE):
    """
    Play a DMX frame sequence using per-frame delays.
    Uses absolute timing to avoid cumulative lag.
    """
    global wrapper
    print(f"----Playing----")

    wrapper = ClientWrapper()
    total_delay = 0

    def make_send_func(frame):
        def send():
            wrapper.Client().SendDmx(universe, _OLAArrayCompat(frame), DmxSent)
        return send

    for i, frame in enumerate(dmx_frames):
        delay = frame_delays_ms[i] if i < len(frame_delays_ms) else 33
        wrapper.AddEvent(total_delay, make_send_func(frame))
        total_delay += delay

    wrapper.AddEvent(total_delay, wrapper.Stop)
    wrapper.Run()