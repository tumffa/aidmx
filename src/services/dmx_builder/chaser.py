from typing import Any, Dict, List, Optional, Tuple, Union

from src.services.dmx_builder.fixture_pattern import FixturePattern


class Chaser:
    """
    Combines a sequence of waits (fractions) and Pattern compilations.

    Usage:
        ch = Chaser(name="MyChaser", default_interval=500)
        ch.add_wait(0.25)  # fraction uses default_interval
        ch.add_pattern(fp1, fixtureA, interval=500, scale_dimmer="both")
        ch.add_wait(0.5)
        ch.add_pattern(fp2, fixtureB, interval=250)
        compiled = ch.compile()
    """

    def __init__(self, name: str, default_interval: int = 500):
        self.name = name
        self.default_interval = int(default_interval)
        # Sequence entries:
        # - ('wait', fraction, optional_interval)
        # - ('pattern', pattern, fixture, interval, scale_dimmer)
        self.sequence: List[
            Union[
                Tuple[str, float, Optional[int]],
                Tuple[str, FixturePattern, Dict[str, Any], int, Optional[Any]],
            ]
        ] = []

    def add_wait(self, fraction: float, interval: Optional[int] = None) -> None:
        self.sequence.append(("wait", float(fraction), interval))

    def add_pattern(
        self,
        pattern: FixturePattern,
        fixture: Dict[str, Any],
        interval: int,
        scale_dimmer: Optional[Any] = None,
    ) -> None:
        self.sequence.append(("pattern", pattern, fixture, int(interval), scale_dimmer))

    def _combine_sequences(self, sequences: List[List[Any]]) -> List[Any]:
        """
        Combine multiple precompiled sequences by:
        1) Emitting leading command tuples from each sequence until a wait appears.
        2) Repeatedly selecting the shortest head wait, appending it, subtracting it
           from the others, then emitting all subsequent commands from the sequences
           whose wait reached zero (until their next wait).
        The input sequences are lists like [wait, cmd, wait, cmd, ...] or may start
        with command tuples.
        """
        # Clone sequences and prepare pointers
        seqs: List[List[Any]] = [s[:] for s in sequences if s]
        positions: List[int] = [0 for _ in seqs]
        combined: List[Any] = []

        # Helper: emit consecutive commands from a sequence starting at current position
        def emit_commands(idx: int) -> None:
            nonlocal combined
            while positions[idx] < len(seqs[idx]) and not isinstance(seqs[idx][positions[idx]], int):
                combined.append(seqs[idx][positions[idx]])
                positions[idx] += 1

        # Step 1: emit leading commands from each sequence
        for i in range(len(seqs)):
            emit_commands(i)

        # Initialize head waits
        waits: List[Optional[int]] = []
        for i in range(len(seqs)):
            if positions[i] < len(seqs[i]) and isinstance(seqs[i][positions[i]], int):
                waits.append(int(seqs[i][positions[i]]))
            else:
                waits.append(None)

        # Step 2: time-slice by shortest waits
        def all_done() -> bool:
            for i in range(len(seqs)):
                if positions[i] < len(seqs[i]):
                    return False
            return True

        while not all_done():
            # Collect active waits
            active = [(i, w) for i, w in enumerate(waits) if w is not None]
            if not active:
                break
            _, min_wait = min(active, key=lambda x: x[1])
            combined.append(int(min_wait))
            # Subtract min_wait from all
            for i, w in enumerate(waits):
                if w is not None:
                    waits[i] = max(0, w - int(min_wait))

            # For every sequence that reached zero, advance past wait and emit commands
            zero_idxs = [i for i, w in enumerate(waits) if w == 0]
            for i in zero_idxs:
                # consume the wait int
                if positions[i] < len(seqs[i]) and isinstance(seqs[i][positions[i]], int):
                    positions[i] += 1
                # emit all commands until next wait
                emit_commands(i)
                # load next wait if any
                if positions[i] < len(seqs[i]) and isinstance(seqs[i][positions[i]], int):
                    waits[i] = int(seqs[i][positions[i]])
                else:
                    waits[i] = None

        # Drop trailing waits without following commands
        while combined and isinstance(combined[-1], int):
            combined.pop()
        return combined

    def compile(self, length: Optional[int] = None) -> List[Any]:
        """
        Convert the assigned sequence into a combined list of wait_times and tuples
        produced by pattern compilers, merging consecutive pattern segments by
        looking at their wait times.
        If length (ms) is provided, cap the total sum of waits to exactly length,
        truncating the final wait as needed and including any commands at that final tick.
        """
        output: List[Any] = []
        pending: List[List[Any]] = []

        def flush_pending():
            nonlocal output, pending
            if not pending:
                return
            if len(pending) == 1:
                # Append the entire precompiled pattern as-is
                output.extend(pending[0])
            else:
                # Combine consecutive patterns and append the combined list
                combined = self._combine_sequences(pending)
                output.extend(combined)
            pending = []

        for entry in self.sequence:
            if not entry:
                continue
            if entry[0] == "wait":
                # Flush any pending pattern sequences before adding a standalone wait
                flush_pending()
                _, fraction, interval = entry  # type: ignore
                eff_interval = int(interval) if interval is not None else self.default_interval
                wait_ms = max(1, int(round(float(fraction) * float(eff_interval))))
                output.append(wait_ms)
            elif entry[0] == "pattern":
                _, pattern, fixture, interval, scale_dimmer = entry  # type: ignore
                seq = pattern.compile_pattern(fixture, interval, scale_dimmer)
                if seq:
                    pending.append(seq)

        # Flush any remaining pattern sequences
        if pending:
            flush_pending()

        # If a length is requested, repeat the compiled cycle to fill the length,
        # and truncate the final wait as needed (including commands at that tick).
        if length is not None:
          cap_ms = max(0, int(length))
          cycle = output[:]  # one cycle of the sequence
          # Sum of waits in the cycle
          cycle_wait_sum = sum(x for x in cycle if isinstance(x, int))
          # If cycle has no waits, return it once (can't advance time)
          if cycle_wait_sum <= 0:
            return cycle

          repeated: List[Any] = []
          used = 0

          # Append whole cycles while we have room
          while used + cycle_wait_sum <= cap_ms:
            repeated.extend(cycle)
            used += cycle_wait_sum

          # Append partial cycle if there is remaining time
          remaining = cap_ms - used
          if remaining > 0:
            i = 0
            n = len(cycle)
            # Include any leading commands before the first wait
            while i < n and not isinstance(cycle[i], int):
              repeated.append(cycle[i])
              i += 1

            while i < n:
              entry = cycle[i]
              if isinstance(entry, int):
                if remaining <= 0:
                  break
                wait_val = int(entry)
                if wait_val <= remaining:
                  repeated.append(wait_val)
                  remaining -= wait_val
                  i += 1
                  # append commands following this wait
                  while i < n and not isinstance(cycle[i], int):
                    repeated.append(cycle[i])
                    i += 1
                  continue
                else:
                  # Truncate final wait to remaining and include subsequent commands at this tick
                  repeated.append(remaining)
                  remaining = 0
                  # Append commands at this tick
                  j = i + 1
                  while j < n and not isinstance(cycle[j], int):
                    repeated.append(cycle[j])
                    j += 1
                  break
              else:
                # commands encountered before any wait
                repeated.append(entry)
                i += 1

          return repeated

        return output

def color_pulse(universe, interval, length=None):
    fixtures = universe["abovewash"]

    fp = FixturePattern(name="ColorPulse")
    fp.define_pattern([
        ("dimmer", 255),
        ("shutter", "open"),
        ("rgb", "red", ("green", 2)),
        ("rgb", "green", ("red", 2))
    ])

    ch = Chaser(name="ColorPulseChaser", default_interval=interval)
    ch.add_pattern(fp, fixtures["1"], interval=interval, scale_dimmer="both")
    ch.add_pattern(fp, fixtures["2"], interval=interval, scale_dimmer="both")
    ch.add_pattern(fp, fixtures["3"], interval=interval, scale_dimmer="both")
    ch.add_pattern(fp, fixtures["4"], interval=interval, scale_dimmer="both")

    return ch.compile(length=length)

def strobe(universe, interval=500, length=None):
    fixtures = universe["abovewash"]

    ch = Chaser(name="StrobeChaser", default_interval=interval)

    initialise = FixturePattern(name="StrobeInitialise")
    initialise.define_pattern([("dimmer", 0), ("shutter", "closed"), ("rgb", "white")])
    for i in range(1, 5):
        fixture = fixtures[str(i)]
        ch.add_pattern(initialise, fixture, interval=interval, scale_dimmer=None)
    ch.add_wait(0.0)

    fp = FixturePattern(name="StrobePattern")
    fp.define_pattern([("strobe", "nicestrobe"), ("rgb", "white"), ("dimmer", 255), 0.1, ("shutter", "closed"), ("dimmer", 0)])
    steps = [0, 2, 1, 3, 2, 0, 3, 1]
    for step in steps:
        fixture = fixtures[str(step + 1)]
        ch.add_pattern(fp, fixture, interval=interval, scale_dimmer=None)
        ch.add_wait(0)
    
    stop = FixturePattern(name="StrobeStopPattern")
    stop.define_pattern([("shutter", "open")])
    for i in range(1, 5):
        fixture = fixtures[str(i)]
        ch.add_pattern(stop, fixture, interval=interval, scale_dimmer=None)

    return ch.compile(length=length)
    

if __name__ == "__main__":
    universe = {
    "size": 32,
    "abovewash": {
      "1": {
        "address": 7,
        "colorchannels": {
          "blue": 2,
          "green": 1,
          "red": 0
        },
        "colortype": "seperate",
        "dimmer": 3,
        "id": 1,
        "nicestrobe": 250,
        "shutter": 4,
        "shutters": {
          "open": 0
        },
        "strobe": 4,
        "stroberange": [
          20,
          255
        ]
      },
      "2": {
        "address": 12,
        "colorchannels": {
          "blue": 2,
          "green": 1,
          "red": 0
        },
        "colortype": "seperate",
        "dimmer": 3,
        "id": 2,
        "nicestrobe": 250,
        "shutter": 4,
        "shutters": {
          "open": 0
        },
        "strobe": 4,
        "stroberange": [
          20,
          255
        ]
      },
      "3": {
        "address": 17,
        "colorchannels": {
          "blue": 2,
          "green": 1,
          "red": 0
        },
        "colortype": "seperate",
        "dimmer": 3,
        "id": 3,
        "nicestrobe": 250,
        "shutter": 4,
        "shutters": {
          "open": 0
        },
        "strobe": 4,
        "stroberange": [
          20,
          255
        ]
      },
      "4": {
        "address": 22,
        "colorchannels": {
          "blue": 2,
          "green": 1,
          "red": 0
        },
        "colortype": "seperate",
        "dimmer": 3,
        "id": 4,
        "nicestrobe": 250,
        "shutter": 4,
        "shutters": {
          "open": 0
        },
        "strobe": 4,
        "stroberange": [
          20,
          255
        ]
      }
    },
    "strobe": {
      "1": {
        "address": 1,
        "colorchannels": {
          "blue": 4,
          "green": 3,
          "red": 2
        },
        "colortype": "seperate",
        "dimmer": 0,
        "id": 0,
        "nicestrobe": 211,
        "shutter": 1,
        "shutters": {
          "closed": 7,
          "open": 0
        },
        "strobe": 1,
        "stroberange": [
          130,
          249
        ]
      },
      "2": {
        "address": 27,
        "colorchannels": {
          "blue": 4,
          "green": 3,
          "red": 2
        },
        "colortype": "seperate",
        "dimmer": 0,
        "id": 5,
        "nicestrobe": 211,
        "shutter": 1,
        "shutters": {
          "closed": 7,
          "open": 0
        },
        "strobe": 1,
        "stroberange": [
          130,
          249
        ]
      }
    }
  }
    
    compiled_chaser = color_pulse(universe, interval=128)
    print(compiled_chaser)
