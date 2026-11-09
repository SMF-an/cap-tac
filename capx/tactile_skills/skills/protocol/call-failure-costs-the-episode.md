---
id: R17
name: call-failure-costs-the-episode
family: protocol
summary: A call that fails, times out or is refused bought one missed attempt — left uncaught it ends the program, which scores 0 exactly like a wrong answer.
task_types: []
tasks: [tactile_memory_match]
channels: [depth]
objects: any
evidence: >-
  2026-11-07 gpt-6-astra trial 28 ended on its first can: the release path
  called move_delta outside the try, the call was abandoned 31 s in, and the
  client's read timeout ended the trial with both candidates untouched. The
  same batch's trials 26 and 27 caught nothing either, but every failure they
  swallowed — a refused move, a close that met nothing, a re-approach to a can
  a previous close had pushed — was one attempt, and the program went on.
---

## How

Wrap every call, the retract and the release included: those are the ones most
often written outside the try, and a refused rise from a failed grasp ends the
episode exactly as surely as a refused grasp does. A failure is one missed
attempt on that object — open if the hand may be shut, go back to cruise
height, and take the next object or one new approach
([[search-continues-after-miss]], [[cruise-height]]).

A failed motion is not a motion that did nothing. The timeout message says so:
the sim operation is still running on the server's main thread and will still
write its rows. Re-read the pose before retrying rather than retrying blind
([[tactile-cannot-locate]]).

## Traps

- **A close that met nothing does not raise.** It returns the 34.00 mm far
  plane on both pads, which looks like a reading and is not one
  ([[depth-grip-scale]]). A try/except will not catch it.
- **A reflex stop is not an exception either.** It returns a reading with
  `terminated_by` set; read that before treating the stop as an error
  ([[reflex-stop-on-events]]).
