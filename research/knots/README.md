# Knots research (draft)

Domain design for a third model in barista's family: knot-tying Q&A on the
same untied-head, constrained-output-vocabulary architecture.

## Domain

`knots.json` lists 72 knots across 7 categories (general, climbing, boating,
fishing, camping, rescue, decorative), each with a canonical name and known
aliases. Aliases matter for the which-knot style: a question may name a knot
by an alias the model should still recognize as the same target.

## Question styles

Two styles, per the user's choice of "both":

### how_to_tie

Given a knot (by canonical name or alias), answer with the tying procedure.

Question templates (paraphrase pool, not exhaustive):
- "how do I tie a {knot}?"
- "show me how to tie a {knot}"
- "what are the steps for a {knot}?"
- "how is a {knot} made?"
- "can you walk me through tying a {knot}?"
- "steps to tie a {knot}"

Answer shape: an ordered sequence of short imperative steps, each built from
a small controlled action vocabulary (verbs: *wrap, cross, pass, tuck, pull,
dress, form, thread, loop, wrap-around*; parts: *working end, standing part,
standing end, bight, loop, turn*; connectives: *around, over, under, through,
back, then*). This is the largest consumer of output-vocabulary budget and
the main place barista's "reused vs. appended word" split matters -- see
plan step 1's vocabulary target of ~1200-2000 words.

### which_knot

Given a scenario, answer with a knot recommendation and a one-line reason.
Scenario categories to cover (drives QA generation coverage, not the output
vocabulary directly):

- joining two ropes of the same diameter
- joining two ropes of different diameter
- making a fixed (non-slipping) loop
- making an adjustable/tensioning loop
- attaching a rope to a post, ring, or rail
- needing a quick-release hitch
- a climbing tie-in or anchor
- a load-bearing or safety-critical join
- attaching fishing line to a hook or lure
- securing a tarp or tent guyline
- lashing poles together
- a decorative or finishing knot

Answer shape: `{knot name}` + a short justification clause using a small
fixed set of property words (*strong, secure, quick-release, adjustable,
non-slipping, bulky, jams under load, easy to untie*). Much smaller
per-answer vocabulary than how_to_tie -- closer to classification, as noted
when this direction was scoped.

## Answer-shape contract (for later `qa_pairs.jsonl` generation)

Both styles ultimately produce `{"question": str, "answer_words": [str, ...]}`
records (plan step 3's input to `build_dataset.py`) -- the `answer_words` list
is pre-tokenized at the word/punctuation level, matching how `vocab.json`
classes are looked up one word at a time. Every word in `answer_words` must
end up in the curated output vocabulary (plan step 3's hard-fail conformance
check); this taxonomy file exists so the vocabulary curated in step 3 is
scoped to what these question styles actually need, rather than guessed at
independently.

## Open questions before generating the corpus

- How many paraphrased questions and answer-wording variants per
  knot/scenario are needed to avoid overfitting a ~1-2M-core-param model
  (plan risk #2)? Barista's own example count per topic is not published, so
  this needs a val-loss-driven answer during training (plan step 6), not a
  number fixed up front.
- Whether `which_knot` answers should sometimes name more than one acceptable
  knot (several knots satisfy "join two ropes of different diameter") --
  affects both corpus design and how greedy decoding is judged.
