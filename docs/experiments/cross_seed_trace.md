# Cross-seed trace chain

Does a model get better across independent seeds when each seed's prompt
carries the previous seed's execution trace?

The experiment runs seeds `1..N` in order, one process per seed.  Seed `N`'s
initial prompt ends with seed `N-1`'s final program, execution output and
outcome.  Everything else is a fresh episode: the scene is re-randomized, the
code namespace is re-created, and no Python state crosses the boundary.

Every arm starts from the **tactile readings its task needs**, not the full
original-21 route.  The readings carry the task's judgment constants; what is
missing is the procedure around them (grasp, transport, placement, wrist
reset, search, recovery).  The trace is meant to teach that procedure.  A
recipe that exists only to obtain the reading -- the bulb's legs/lead/seat,
the strap's grasp/ramp/hold, the one-grasp protocol, the weigh-by-lift
protocol -- is part of the reading set.  Generic scales and channel rules
that the task-specific reading already names are withheld.

For grasp_classify the route is one reading: `texture-slope-two-grips` -- the
texture calibration and its 0.06 split.  The generic depth scale, the weight
channel, the yield caveat and the channel rule are not readings this task
needs, so they are withheld.

No placement recipe is delivered: the old library has no `place` family.  The
prompt's placement promise is therefore rewritten, while its promise about
the channel, the two grips and the split value stays true.

## The tactile_memory_match arm

`UniVTACTactileMemoryMatchTraceCodeEnv` follows the same rule on the harder
task.  The task is a three-property match (2 textures x 2 weights x 2
hardnesses), and the hidden pair is drawn per seed.

The task's `check_success()` returns a bare bool, so its
`check_success_detail()` was added on 2026-11-08 to report where each candidate
physically ended up:

```
CHECK DETAIL:
  candidate_left: placed=False, on_match_pad=False, xy error 4.22 cm, z error 1.10 cm
  candidate_right: placed=True, on_match_pad=True, xy error 0.84 cm, z error 0.20 cm
```

That splits a failure into "carried the wrong can" against "carried the right
one but did not place it" without naming the twin or either hidden class --
`filter_check_detail` allowlists the two candidate names and those four fields,
so `reference_class_key`/`distractor_class_key` can never travel.  The detail
also makes the prompt's request to print the comparison still worth doing: the
placement tells the next seed what went wrong, the stdout tells it what the
readings were.

The route is four readings:

1. `memory-match-one-grasp` -- the one-grasp protocol, 0.50 loose / 0.25 firm,
   then the lift and sag pair;
2. `can-texture-difference` -- the 0.02 anisotropy split;
3. `follow-is-signal` -- the load response is the lift follow-through;
4. `position-mode-for-yield` -- the servo caveat.

`gel-shear-weight` is withheld: this task's weight axis is the can's own z --
`follow` (how far it rode the lift) and `sag_mm` (its stick-slip spread while
suspended), see `DIMENSION_AXES` in the task.  The pads' shear difference is
not a feature of the scorer, so the rule would teach a read that decides
nothing here, and it contradicts `follow-is-signal` in the same block.  The
generic depth scale and the channel rule are not readings this task needs
either.

No approach, transport or placement rule is delivered (the old library has
none), and the prompt still points at the one-grasp protocol because it is
there.

Run it with its own config; the 8120 server has to be restarted on the new
task, and a seed costs roughly 10 minutes (the 2026-11-07 standard run measured
694 s per trial), so 30 seeds is about five hours:

```bash
bash scripts/launch_univtac_server.sh tactile_memory_match 8120 <gpu> 0
bash scripts/run_trace_chain.sh tactile_memory_match 1 30 <model> 8120 12000
```

## What one seed leaves for the next

`<output_dir>/trace_chain/` holds:

| file | contents |
|---|---|
| `seed_NNN.json` | final program, stdout/stderr, check detail, `task_completed`, reward, sandbox rc, block/regeneration counts, token usage, wall time, `injected_from_seed` |
| `seed_NNN_injected_block.txt` | the exact text appended to that seed's prompt |

The program in the record has the framework-appended `check_success()` call
removed.  That function is deliberately absent from the API document, and a
trace that carried it would teach the next seed about a hidden function.  On a
traced run the call is not executed at all: its own stdout prints the
ground-truth texture and expected pad when the task fails, and that output
would travel into the next seed's prompt.  The verdict is unaffected --
`env.step` already calls `/check_success` to compute `task_completed`.

The trace does carry a **failure reason**, because "it failed" is not enough to
learn from: the task's check detail supplies the geometric reasons ("Y off by
3.2 cm (max 2 cm)", "tilted 24° (max 15°)") and the measurements behind them
(per-axis position error, tilt cosine).  Only those measurement-like fields
reach the trace.  The same detail also carries `ground_truth_texture`,
`expected_pad` and raw object poses; those are filtered out by an allowlist in
`capx.envs.trace_chain.filter_check_detail`, so the next seed is told what
went wrong, not what the answer was.

The trace arm writes to `./outputs/univtac_grasp_classify_trace`, not to the
baseline task directory: `runner._setup_output_dir` inserts the model name
before the last component, so the run lands in
`outputs/<model>/univtac_grasp_classify_trace/`.  `run_trace_chain.sh` reads
`output_dir` from the config it is given and derives the same path, so analysis
follows the config rather than a hard-coded name.

The chain is bounded: `--trace-max-chars` (default 12000) keeps the program and
the tails of both output streams, and marks whatever it drops.  Seed 1 has no
trace and is the cold start.  If a seed is missing, the next one reads the most
recent earlier record and says so in `injected_from_seed`.

Generated code runs unattended.  `input()` is shadowed in the exec namespace
and raises, so a program that tries to ask the operator fails its block with a
readable reason instead of parking the chain on stdin.  That happened on
2026-11-08: seed 1, told the split must come from its own attempts, wrote
`json.loads(input('Enter measured calibration slopes...'))` and blocked the
first seed of a 30-seed run until the exec alarm.  The prompt now also says
the program is unattended, and that with no earlier attempt the model should
choose a split, act on it, and print the slope it measured.

## Trace modes

`--trace-mode` (or `CAPX_TRACE_MODE` for `run_trace_chain.sh`) selects what the
next seed receives.  The default is the original arm and is unchanged:

| mode | what the next seed gets |
|---|---|
| `off` | nothing; the prompt is the standard one |
| `last` | the most recent record only (the arm the first eight chains ran) |
| `card` | the model's four-line experience card for the last record |
| `window_card` | cards for the last `--trace-window` records plus the best success, with a streak line |

`window_card` with `--trace-window 1` is the card pair (the last card plus the
best success).  Two older arms, `dual` and `dual_card`, are gone: neither was
ever run end to end, and `dual_card` was exactly that one-record window.

`card` adds one reflection call per seed, made after the trace record is
written.  The model sees the same filtered trace the next seed would have
received and writes four lines (`DID`, `MEASURED`, `WHY IT FAILED` or
`WHY IT PASSED`, `NEXT TIME`); the card is stored on the record as `card` and
the next seed gets it instead of the raw program.  The filtered check detail is
still appended verbatim under the card, so compression cannot lose the exact
numbers.  If the reflection call fails -- or the record predates card mode --
the renderer falls back to that seed's raw trace, marked as such, so the chain
never goes blind.  The card call's token usage is kept under
`meta["reflection"]`, separate from the trial's own usage.

The record names the slots it was injected from: `injected_from_seed`, and
`injected_best_seed` when `window_card` added a best success outside the
window.  A missing seed still falls back to the latest earlier record, so a
killed seed does not break the chain, but the record names the seed it
actually used.

`window_card` extends the card arm to the last `--trace-window` (default 3)
records and records the seeds it used in `injected_window_seeds`.  The window
is cards, not raw programs, on purpose: one blind_pick_balls program is
already ~12k characters at the cap, so three raw programs would either double
the prompt or cut every program's middle.  The one piece of evidence a window
adds over `card` is repetition: when every attempt in the window ended the
same way, a `STREAK:` line says so once, and the model does not have to infer
it from three similar cards.  When the streak is absent the window is more of
the same evidence, which is exactly what the arm tests.

## The tension_strap arm

`UniVTACTensionStrapTraceCodeEnv` runs the force/time variant: the pair of
targets and their hold durations are drawn from the seed
(`capx/serving/tension_strap_spec.py`), the server binds the same draw into
`StrapAcceptanceLimits`, and the client re-renders the prompt and the oracle
from it -- so what the model is told, what the oracle aims at and what the
scorer reads cannot drift apart.  Seed 1, for example, asks 10.5 N for 2.5 s
then 14 N for 3 s.

This task serves **exactly one seed per server process**: the band modulus is
drawn before UIPC initialises and `_reset_actors` raises on any other
combination.  Its config therefore sets `server_lifecycle: per_seed`, and
`run_trace_chain.sh` owns the server for the whole chain:

1. stop whatever is on the port -- supervisor first (it respawns a dead child
   in 2 s), then the server, both SIGKILLed;
2. start `scripts/launch_univtac_server.sh <task> <port> <gpu> <seed>`;
3. poll `/health` until it reports the right `task` **and** `seed` (the server
   now returns its seed);
4. run the seed's trial, and stop the server again when the chain exits.

GPUs come from `CAPX_GPU` (default 0), the startup budget from
`$CAPX_SERVER_START_TIMEOUT` (default 900 s -- a force task's first reset runs
the full scene build, measured at ~330 s on a loaded machine).

The route is four readings: the gauge (`strap-load-is-shear-difference`: the
reference close, 0.93 N per px of the two pads' shear difference and the
2.3 mm take-up), the two control recipes (`strap-grasp-and-ramp`: climb in
<=2 mm steps and re-grasp after three dead raises; `strap-hold-creeps`: hold
the height and the clock, never the reading), and the slip baseline
(`slip-pair-baseline`: the first `detect_slip` call only sets the reference).
The depth scale and the channel rule are not readings this task needs.

The old library has no approach, grip or recovery rule, so what is left to
learn is the grasp itself and surviving a refused motion.  The prompt says
only that a hold is made of commands.

The task's `check_success_detail()` was added for this arm and reports
`failure` (`strap_ruptured`, `grasp_lost`, `travel_limit`, `timeout`, ...),
`stages_completed`, `current_hold_s` and `elapsed_s`.  It never reports the
measured tension: estimating the load from the gel is the task, and the
physical value would hand that estimate over.

An episode that simply stops before both holds latches none of those strings
-- the policy clock never reaches the 60 s timeout, so the detail used to
report no reason at all.  Since 2026-11-08 an empty latch with a False
verdict is reported as `incomplete_two_stage_sequence`, the string the task's
own controller path already uses, so a later seed is told why the attempt
failed and not only how far it got.

A seed costs 520-820 s of episode plus the per-seed server start, so 30 seeds
is roughly 7-8 hours.

## The bulb_tightening arm

`UniVTACBulbTighteningTraceCodeEnv` runs the insertion.  What transfers between
seeds is a procedure, not a value: the thread lead is drawn per seed
(`default_rng(seed + 71823)`) before UIPC initialises, so a program that
carries the previous attempt's lead is wrong by construction on the next one,
and a program that measures it in-episode is right on all of them.  The task
raises on any other seed (`Create one force-task process per seed ...`), so the
config sets `server_lifecycle: per_seed`.

The route is seven readings: the direction legs (`thread-direction-by-legs`),
the shear-to-millimetre conversion (`mirrored-shear-slide`), the lead
measurement (`lead-by-overshoot`), the seat reading (`seat-by-slide-collapse`),
the bulb's own close recipe (`bulb-close-force`, 27.2 mm), and the two rules
that say when a shear reading is usable (`confidence-gate`,
`lattice-pitch-budget`).  The generic depth scale and the channel rule are
withheld: the depth scale's 20.5 mm rigid-part example is a crush target on
this glass, and the 2026-11-09..11 trace runs lost every seed at the first
close to it.  The close target is a constant the run cannot derive from its
failures, so it is delivered rather than learned.

The old library has no grasp-pose, feed, wrist-reset or recovery rule, so what
is left to learn is the operating procedure: how to take the glass, how far to
feed, how to re-set the wrist, how many blocks to run, and what to do after a
dead raise or a refused motion.  Of the prompt's promises, only the
wrist-reset one is rewritten, because the old library has no
`wrist-limit-segmentation`; the direction, lead and seat promises stay true.

The failure detail is ViTaForge's existing `check_success_detail`.
`filter_check_detail` keeps `failure`, `held_steps` and `reasons`, and drops
`measured`, `nominal_free_travel_mm` and `target_torque_Nm`.  The last two are
the answers: the lead and the torque bar.  The `advance` and `seat_force`
reason strings were reworded on 2026-11-08 for the same reason -- they used to
print `<travel-0.002> mm required` and `<= 3`, which is exactly the per-seed
lead and the seat threshold.  `phase` is dropped too: only the expert path
writes it, so on a policy run it reads `setup` on every seed.

A seed costs 52-108 minutes on the oracle's own path (the lead is drawn per
seed: seed 1 about 108 min, seed 9 about 52), plus a server start per seed.
Ten seeds is 10-20 hours; thirty is 45-110.  Start with a short range.

## The blind_pick_balls arm

`UniVTACBlindPickBallsTraceCodeEnv` runs the blind retrieval.  The twelve
balls' materials are drawn from `cfg.seed` in `Task.__init__` and never
redrawn, and `reset(seed)` only re-samples which cells are occupied -- so a
program that carries a cell or a material from the previous attempt is wrong by
construction, and a second seed needs a second server process.  The config sets
`server_lifecycle: per_seed`.

The route is two readings: the material reading with its 0.06 relief bar
(`texture-in-relief`) and what fully open pads read
(`wide-pads-see-nothing`).  The depth scale and the channel rule are not
readings this task needs.

The old library has no search, transport or placement family, so what is left
to learn is the procedure: how to sweep the rows, how to leave a met ball where
it is, how to lift and carry, and where to release.

The failure detail is the adapter's `check_success_detail`, which returns the
task's own `_info()`: `filter_check_detail` keeps `failure` (one of
`wrong_material`, `unverified_grasp`, `timeout`, `grasp_lost`, ...) and
`physics_step`, and drops the rest.  The task's stdout is where the probe
readings travel, so the trace prompt asks for them to be printed.

The task's own clock is 120 s of simulated time (halved from 240 s on
2026-11-04), about 2750-3600 s of wall clock at 0.38-0.5 s per control step.
The oracle's path is much shorter, so ten seeds is roughly 4-10 hours and
thirty is 12-30.

## The weight_classify arm

`UniVTACWeightClassifyTraceCodeEnv` runs the light/heavy classification.  The
label is drawn per reset from the reset seed (`_reset_actors` ->
`rng.choice(['light', 'heavy'])`), so one server process serves every seed and
the config leaves `server_lifecycle` at `external`.

The route is three readings: `grip-force-by-mass` (the ~28.5 weighing grip and
the 0.2 px shear split), `weigh-grasp-by-lift` (lift 3 cm, check the block
follows, re-close at ~28.0 and lift again if it does not) and
`gel-shear-weight` (the shear channel).  The depth scale, the load response,
the yield caveat and the channel rule are not readings this task needs.  The
old library has no approach, transport or placement family, so carrying and
seating are what the trace is meant to teach.

The served task originally exposed only ``check_success`` (a bool), so the
trace carried no placement reason at all.  It now has a
``check_success_detail`` mirroring hardness_classify's (position error, tilt
and the same reason strings); the trace filter keeps those three and withholds
the label, the target plate and the raw pose.  A server that was already
running when the method was added keeps the old class in memory until it is
restarted.

## The hardness_classify arm

`UniVTACHardnessClassifyTraceCodeEnv` is the same shape: the soft/hard label is
drawn in `_reset_actors` from the reset seed, so the config is
`external`.  Its route is two -- `hardness-two-closes` (the depth gained from
position 0.10 to 0.06: ~1.6 mm hard, ~0.8 soft, split 1.2 mm) and
`position-mode-for-yield` (the servo caveat) -- and what is left to learn is
again the grasp, carry and placement procedure.

## The arm_shadow_piano arm

`UniVTACArmShadowPianoTraceCodeEnv` runs the note sequence.  The sequence and
each note's dynamic are drawn from the server's seed before the task is built
(`capx/serving/piano_spec.py`), so a seed needs its own server process; the
config sets `server_lifecycle: per_seed`.

The delivered set is the one reading this task needs,
`piano-gel-force-band`: the 0.35 N light / 0.50 N heavy targets and the tilt
plateau above 0.43 N where the estimate saturates.  The library's taxonomy
routes the task to the `force` family for the config test, but its own comment
says the env class names exactly that one skill, because none of the family's
grasp / carry / load rules describes a single-finger key press.  The frozen
manifest records this skill as excluded from its own DIGIT runtime, where
`left_force_N` and `close_gripper(press_force_N=...)` do not exist; this
harness serves those fields on the GelSight Mini / UIPC rig the skill was
measured on, so the exclusion does not apply here.

What is left to learn is the playing procedure: moving between keys without
dragging the pad, releasing before the next note, and ordering the presses.  A
seed costs about 10-15 minutes (the standard seed-1 trial measured 666 s) plus
a per-seed server start, so thirty seeds is roughly 6-10 hours.

The check detail is the task's own `_info()`.  `filter_check_detail` keeps
`failure`, `terminal_reason` and the progress counters (`notes_completed`,
`notes_attempted`, `total_notes`); `piano_sequence` and `piano_instruction`
are the answer for that seed and never reach the next prompt.

## Run

```bash
# The server must already be running on the same task.  For grasp_classify,
# weight_classify and hardness_classify one server process serves every seed;
# force tasks such as bulb_tightening, blind_pick_balls and arm_shadow_piano
# need a fresh server per seed.
bash scripts/launch_univtac_server.sh grasp_classify 8120 <gpu> 0

bash scripts/run_trace_chain.sh grasp_classify 1 30 <model> 8120 12000
```

The model proxy has to be up as well; the driver takes its URL from
`CAPX_SERVER_URL` and defaults to `http://127.0.0.1:8112/chat/completions`.
That default is not cosmetic: `launch.py` falls back to port 8110 when
`--server-url` is omitted, and on this host 8110 is another user's viser
server, which accepts the connection and drops it without an HTTP reply.  The
symptom is a `RemoteDisconnected` traceback on the first model query of seed 1.
The driver now POSTs an empty body at the proxy before starting and refuses to
run unless it gets an HTTP status back.

`run_trace_chain.sh` launches one `capx/envs/launch.py` per seed with
`--total-trials N --resume-idx N`, so exactly seed `N` runs in that process.
With the default `server_lifecycle: external` the server is never started or
stopped by the script; a config that sets `per_seed` hands the server to the
script instead (see the tension_strap and bulb_tightening arms above).  It
defaults to
`env_configs/univtac/<task>/univtac_<task>_trace.yaml`; pass a different config
as the seventh argument to run another arm.  For a run with every skill
withheld, use the standard config and add `--deliver-tactile-skills False` to
the `launch.py` call.

## Analyze

```bash
python scripts/analyze_trace_chain.py \
    outputs/<model>/univtac_grasp_classify_trace/trace_chain
```

It writes `chain_summary.csv` and `chain_summary.md` and prints the aggregate.
The success measure is **`task_completed`**, not `sandbox_rc`: the latter only
says the generated code ran.  For grasp_classify the script also annotates each
seed with the simulator's `truth=rough/plain` line from
`outputs/univtac_tactile_calibration.log`; that annotation is analysis-only and
never enters a prompt.

## Known limits

- There is no control arm.  A rising curve under this setting is consistent
  with learning from traces, but seed difficulty alone can also move it.  The
  smallest useful control is the same seeds with `--trace-mode off`.
- The kept skills carry each task's judgment constants (the splits, the lead
  formula, the seat rule, the material bar).  This is not a blank slate; what
  the model has to learn is the procedure around them -- grasp, transport,
  placement, wrist reset, search -- which the frozen original-21 library does
  not contain.
- The trace can reveal the previous seed's classification and chosen pad.
  Copying it can be wrong on a differently textured seed; that is part of what
  the experiment observes.
- Temperature is unchanged, so one 30-seed chain is one sample.
