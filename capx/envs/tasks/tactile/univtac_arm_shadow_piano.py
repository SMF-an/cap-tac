import copy

from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.serving.piano_spec import goal_sentence, sample_sequence

PROMPT = """
You are controlling a Franka arm carrying a five-finger Shadow Hand. Only the
index finger moves; every other hand joint is welded, and the wrist's
orientation is fixed. The API is described below.

Goal: {goal}

get_key_position(midi) returns the two palm targets for a key -- `hover`, 40 mm
above it, and `press`, the height at which the fingertip meets it -- and
goto_pose moves between them; pass back the quaternion it returns.

To play a note: goto_pose to `press`, rise 2 mm, then close_gripper(position=0.0)
to bring the index finger down onto the key with the pad still unloaded.  Read
the gel with describe_tactile(side="left") there, then lower the palm in steps
of 0.1 mm -- goto_pose([x, y, z - 0.0001]), starting from that raised height --
and read the gel after each one.  Keep your own running height and command the
next step below your last command, not below where the hand reports it is: the
key pushes back, so the hand lags the command and re-anchoring on it would
never descend.  move_delta cannot step finer than 2 mm; use goto_pose for the
descent.

The gel is the only tactile reading this task gives you, and it is words, not a
number: it describes the surface and how firm or compliant it feels, and it
will not report the force in newtons.  Do not wait for a numeric reading.
Compare readings at the same key -- the unloaded pad, first touch, and each
lower step -- and let the change between two descriptions choose the next step.
A light note needs less descent than a heavy one.

The description is free text: it may be a sentence fragment, may start with
punctuation, may be in another language, and will not use a fixed label.  Your
program has to decide from the sentences it gets: compare the words that
changed between two readings and descend or hold accordingly.  Nothing here
asks you to name the force.  Act on the best reading you have -- a program
that waits for a person, reads from stdin, or raises instead of moving
measures nothing and scores nothing.  A description that says nothing useful
is a missed reading, not a reason to stop: reading again a step lower is a
measurement.

The score is the key's measured force: a light note counts only while that
force stays inside 0.28-0.42 N, a heavy one only while it stays inside
0.43-0.57 N, and each note must be held in its band for 0.12 s of simulated
time and released before the next one.  A force above the band for more than a
moment ends the run, so do not keep descending once the pad feels firmly
loaded.

Lift the palm a few millimetres before opening the finger: opening it while the
pad still presses the key drags the gel across it and spikes the force past the
band.

A description that names no surface at all is the unloaded pad, not a zero
reading; it is the baseline the pressed readings are compared against.

Time advances only while a motion, a close or an open is being commanded; a
reading alone does not move the clock.  So hold the key in its band by
re-issuing goto_pose at the same height a few times before you release it.

A wrong key ends the episode, and a note that is never released does not count.

Write ONLY executable Python code (no code fences). Import numpy if needed.
"""


class UniVTACArmShadowPianoCodeEnv(CodeExecutionEnvBase):
    """The shadow-hand piano task with the note sequence its trial was given.

    The sequence is a property of the seed, so it cannot be a class constant:
    the goal the model reads is re-rendered in ``reset`` from the trial id,
    which is the same seed the server drew the task's notes from
    (capx/serving/piano_spec.py).

    There is no oracle on this branch.  The scalar-channel reference played the
    notes with a calibrated force servo fed by the gel's depth statistics, and
    this branch serves the gel only as Octopi's text description -- none of
    that reference can run, so none of it is carried.
    """

    prompt = PROMPT.format(goal=goal_sentence(sample_sequence(0)))
    # The force family's grasp rules describe a two-pad grasp of a free object,
    # not a single finger on a key; the one rule this task needs is the
    # reading rule, so name it and nothing else.
    tactile_skills = ("description-needs-contact",)

    def __init__(self, cfg) -> None:
        self._sequence = sample_sequence(0)
        super().__init__(cfg)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._sequence = sample_sequence(0 if seed is None else seed)
        self._task_prompt = PROMPT.format(goal=goal_sentence(self._sequence))
        self._full_prompt = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",
             "content": [{"type": "text", "text": self._get_complete_prompt()}]},
        ]
        obs["full_prompt"] = copy.deepcopy(self._full_prompt)
        info["task_prompt"] = self._task_prompt
        return obs, info
