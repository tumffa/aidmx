import array
from ola.ClientWrapper import ClientWrapper

# Default universe
UNIVERSE = 1
wrapper = None

def DmxSent(state):
    if not state.Succeeded():
        print("Failed to send DMX")
        wrapper.Stop()

def play_dmx_sequence(frame_delays_ms, dmx_frames, universe=UNIVERSE):
    """
    Play a DMX frame sequence using per-frame delays.
    Uses absolute timing to avoid cumulative lag.
    """
    global wrapper
    print(f"----Playing DMX sequence with OLA on universe {universe}")

    wrapper = ClientWrapper()
    total_delay = 0

    def make_send_func(frame):
        def send():
            wrapper.Client().SendDmx(universe, frame, DmxSent)
        return send

    # Schedule all frames at their absolute time offsets
    for i, frame in enumerate(dmx_frames):
        delay = frame_delays_ms[i] if i < len(frame_delays_ms) else 33
        wrapper.AddEvent(total_delay, make_send_func(frame))
        total_delay += delay

    # Stop wrapper after last frame
    wrapper.AddEvent(total_delay, wrapper.Stop)
    wrapper.Run()

# Example usage (uncomment and replace with real data):
if __name__ == '__main__':
    delays = [1000, 1000, 1000]
    dmx_frames = [
        array.array('B', [0, 0, 0, 0, 0, 0, 255, 0, 0, 255]),
        array.array('B', [0, 0, 0, 0, 0, 0, 0, 255, 0, 255]),
        array.array('B', [0, 0, 0, 0, 0, 0, 0, 0, 255, 255]),
    ]
    play_dmx_sequence(delays, dmx_frames, universe=1)