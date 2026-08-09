"""Generate qa_pairs.jsonl: full coverage of knots.json's 72-knot taxonomy
for how_to_tie, a which_knot scenario set, and a capability style that
answers "what do you know about?" with knots.json's category list.

Every answer is built from word tuples (not free prose) so vocabulary
conformance is true by construction -- though the self-check at the bottom
still verifies it, since a typo in a step tuple would otherwise fail
silently until build_dataset.py.

Procedures here are deliberately simplified to fit a ~280-word controlled
vocabulary (no left/right, no ordinals beyond "first", no plural nouns
except ends/poles/strands, no digits). They describe the general shape of
each knot correctly but are not a substitute for a real reference,
particularly for the safety-relevant categories (climbing, rescue, boating
anchor bends) -- see firmware/esp32_knots/README.md's caution note.

Extending further: add an entry to KNOT_STEPS or WHICH_KNOT using only
words in output_words.json (rerun build_word_list.py first if a step
genuinely needs a new word).

  uv run python research/knots/generate_corpus.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Each value is a list of steps; each step is a tuple of output-vocabulary
# words (no trailing period -- the generator appends one "." token per step).
KNOT_STEPS: dict[str, list[tuple[str, ...]]] = {
    # ---- general -------------------------------------------------------
    "Overhand Knot": [
        ("form", "a", "loop", "in", "the", "rope"),
        ("pass", "the", "working-end", "through", "the", "loop"),
        ("pull", "the", "rope", "to", "tighten"),
    ],
    "Overhand Loop": [
        ("fold", "the", "rope", "into", "a", "bight"),
        ("form", "a", "loop", "with", "the", "bight"),
        ("pass", "the", "bight", "through", "the", "loop"),
        ("pull", "to", "tighten"),
    ],
    "Figure-Eight Knot": [
        ("form", "a", "loop", "in", "the", "rope"),
        ("wrap", "the", "working-end", "around", "the", "standing-part"),
        ("pass", "the", "working-end", "back", "through", "the", "loop"),
        ("pull", "the", "rope", "to", "tighten"),
    ],
    "Figure-Eight Loop": [
        ("fold", "the", "rope", "into", "a", "bight"),
        ("form", "a", "loop", "with", "the", "bight"),
        ("wrap", "the", "bight", "around", "the", "standing-part"),
        ("pass", "the", "bight", "back", "through", "the", "loop"),
        ("pull", "to", "tighten"),
    ],
    "Square Knot": [
        ("cross", "one", "end", "over", "the", "other"),
        ("tuck", "it", "under", "and", "pull", "through"),
        ("mirror", "the", "cross", "and", "tuck", "it", "under", "again"),
        ("pull", "to", "tighten"),
    ],
    "Slip Knot": [
        ("form", "a", "bight", "in", "the", "rope"),
        ("pass", "the", "bight", "through", "a", "loop", "in", "the", "standing-part"),
        ("pull", "the", "standing-part", "to", "tighten"),
        ("pull", "the", "tail", "to", "release"),
    ],
    "Half Hitch": [
        ("pass", "the", "working-end", "around", "the", "standing-part"),
        ("tuck", "it", "back", "through", "the", "loop"),
        ("pull", "to", "tighten"),
    ],
    "Two Half Hitches": [
        ("wrap", "the", "working-end", "around", "the", "post"),
        ("tuck", "it", "back", "through", "the", "loop", "to", "form", "a", "half", "hitch"),
        ("repeat", "the", "half", "hitch", "again"),
        ("pull", "to", "tighten"),
    ],
    "Clove Hitch": [
        ("wrap", "the", "rope", "around", "the", "post"),
        ("wrap", "it", "around", "again"),
        ("tuck", "the", "working-end", "under", "itself"),
        ("pull", "to", "tighten"),
    ],
    "Constrictor Knot": [
        ("wrap", "the", "rope", "around", "the", "post"),
        ("cross", "over", "and", "wrap", "it", "around", "again"),
        ("tuck", "the", "working-end", "under", "both"),
        ("pull", "to", "tighten"),
    ],
    "Surgeon's Knot": [
        ("cross", "one", "end", "over", "the", "other", "and", "wrap", "it", "around", "twice"),
        ("pull", "to", "snug"),
        ("mirror", "the", "cross", "and", "tuck", "it", "under", "again"),
        ("pull", "to", "tighten"),
    ],
    "Double Overhand Stopper": [
        ("form", "a", "loop", "in", "the", "rope"),
        ("pass", "the", "working-end", "through", "the", "loop", "twice"),
        ("pull", "the", "rope", "to", "tighten"),
    ],
    # ---- climbing --------------------------------------------------------
    "Figure-Eight Follow-Through": [
        ("form", "a", "loop", "in", "the", "rope"),
        ("wrap", "the", "working-end", "around", "the", "standing-part"),
        ("pass", "the", "working-end", "back", "through", "the", "loop", "to", "finish", "a", "figure-eight", "knot"),
        ("pass", "the", "tail", "around", "the", "harness"),
        ("follow", "the", "figure-eight", "knot", "back", "through", "with", "the", "tail"),
        ("pull", "to", "tighten"),
    ],
    "Double Figure-Eight Loop": [
        ("fold", "the", "rope", "into", "a", "bight"),
        ("form", "a", "loop", "with", "the", "bight"),
        ("wrap", "the", "bight", "around", "the", "standing-part", "twice"),
        ("pass", "the", "bight", "back", "through", "the", "loop"),
        ("pull", "to", "tighten"),
    ],
    "Alpine Butterfly Loop": [
        ("form", "a", "bight", "in", "the", "middle", "of", "the", "rope"),
        ("twist", "the", "bight", "to", "form", "a", "loop"),
        ("pass", "the", "bight", "through", "the", "loop"),
        ("pull", "to", "tighten"),
    ],
    "Bowline on a Bight": [
        ("fold", "the", "rope", "into", "a", "bight"),
        ("form", "a", "loop", "with", "the", "bight"),
        ("pass", "the", "bight", "up", "through", "the", "loop"),
        ("wrap", "the", "bight", "around", "the", "loop", "and", "pull", "to", "tighten"),
    ],
    "Double Bowline": [
        ("form", "a", "round-turn", "in", "the", "standing-part"),
        ("pass", "the", "working-end", "up", "through", "the", "round-turn"),
        ("wrap", "the", "working-end", "around", "the", "standing-part"),
        ("pass", "the", "working-end", "back", "down", "through", "the", "round-turn"),
        ("pull", "to", "tighten"),
    ],
    "Prusik Knot": [
        ("wrap", "the", "cord", "around", "the", "rope", "three", "times"),
        ("pass", "the", "tail", "through", "the", "coil"),
        ("dress", "the", "coil"),
        ("pull", "to", "check", "that", "it", "jams"),
    ],
    "Klemheist Knot": [
        ("wrap", "the", "cord", "around", "the", "rope", "three", "times"),
        ("pass", "the", "bight", "through", "the", "coil"),
        ("dress", "the", "coil"),
        ("pull", "to", "check", "that", "it", "jams"),
    ],
    "Autoblock Knot": [
        ("wrap", "the", "cord", "around", "the", "rope", "three", "times"),
        ("join", "the", "ends", "with", "a", "carabiner"),
        ("pull", "to", "check", "that", "it", "jams"),
    ],
    "Munter Hitch": [
        ("cross", "the", "rope", "to", "form", "a", "loop"),
        ("pass", "the", "loop", "into", "a", "carabiner"),
        ("pull", "either", "end", "to", "lock", "the", "hitch"),
    ],
    "Double Fisherman's Knot": [
        ("wrap", "the", "tail", "of", "one", "rope", "around", "the", "other", "rope", "twice"),
        ("tuck", "the", "tail", "back", "through", "the", "coil"),
        ("pull", "to", "tighten"),
        ("repeat", "the", "same", "wrap", "and", "tuck", "with", "the", "other", "rope"),
        ("pull", "to", "finish"),
    ],
    "Water Knot": [
        ("form", "an", "overhand", "knot", "in", "the", "webbing"),
        ("follow", "it", "back", "through", "with", "the", "other", "end", "of", "webbing"),
        ("pull", "to", "tighten"),
    ],
    "Girth Hitch": [
        ("pass", "the", "sling", "around", "the", "harness"),
        ("thread", "one", "end", "of", "the", "sling", "through", "the", "other", "end"),
        ("pull", "to", "tighten"),
    ],
    # ---- boating -----------------------------------------------------
    "Bowline": [
        ("form", "a", "loop", "in", "the", "standing-part"),
        ("pass", "the", "working-end", "up", "through", "the", "loop"),
        ("wrap", "the", "working-end", "around", "the", "standing-part"),
        ("pass", "the", "working-end", "back", "down", "through", "the", "loop"),
        ("pull", "to", "tighten"),
    ],
    "Cleat Hitch": [
        ("wrap", "the", "rope", "around", "the", "cleat"),
        ("cross", "the", "rope", "over", "itself", "and", "wrap", "it", "around", "again"),
        ("tuck", "the", "working-end", "under", "itself", "to", "lock", "it"),
        ("pull", "to", "tighten"),
    ],
    "Round Turn and Two Half Hitches": [
        ("wrap", "the", "rope", "around", "the", "post", "twice", "for", "a", "round-turn"),
        (
            "wrap",
            "the",
            "working-end",
            "around",
            "the",
            "standing-part",
            "and",
            "tuck",
            "it",
            "back",
            "through",
            "to",
            "form",
            "a",
            "half",
            "hitch",
        ),
        ("repeat", "the", "half", "hitch", "again"),
        ("pull", "to", "tighten"),
    ],
    "Rolling Hitch": [
        ("wrap", "the", "rope", "around", "the", "pole", "twice"),
        ("cross", "over", "and", "wrap", "it", "around", "again"),
        ("tuck", "the", "working-end", "under", "itself"),
        ("pull", "to", "check", "that", "it", "jams"),
    ],
    "Anchor Bend": [
        ("wrap", "the", "rope", "around", "the", "ring", "twice", "for", "a", "round-turn"),
        ("tuck", "the", "working-end", "under", "the", "round-turn"),
        (
            "wrap",
            "the",
            "working-end",
            "around",
            "the",
            "standing-part",
            "and",
            "tuck",
            "it",
            "under",
            "itself",
            "to",
            "form",
            "a",
            "half",
            "hitch",
        ),
        ("pull", "to", "tighten"),
    ],
    "Sheet Bend": [
        ("form", "a", "bight", "in", "one", "rope"),
        ("pass", "the", "working-end", "of", "the", "other", "rope", "up", "through", "the", "bight"),
        ("wrap", "it", "around", "the", "bight"),
        ("tuck", "it", "under", "itself"),
        ("pull", "to", "tighten"),
    ],
    "Double Sheet Bend": [
        ("form", "a", "bight", "in", "one", "rope"),
        ("pass", "the", "working-end", "of", "the", "other", "rope", "up", "through", "the", "bight"),
        ("wrap", "it", "around", "the", "bight", "twice"),
        ("tuck", "it", "under", "itself"),
        ("pull", "to", "tighten"),
    ],
    "Buntline Hitch": [
        ("pass", "the", "working-end", "through", "the", "ring"),
        ("wrap", "it", "around", "the", "standing-part"),
        ("tuck", "it", "back", "through", "itself", "to", "form", "a", "clove", "hitch"),
        ("pull", "to", "tighten"),
    ],
    "Carrick Bend": [
        ("form", "a", "loop", "with", "one", "rope"),
        ("weave", "the", "other", "rope", "over", "and", "under", "through", "the", "loop"),
        ("dress", "the", "coil"),
        ("pull", "to", "tighten"),
    ],
    "Fisherman's Knot": [
        (
            "form",
            "an",
            "overhand",
            "knot",
            "with",
            "the",
            "tail",
            "of",
            "one",
            "rope",
            "around",
            "the",
            "other",
            "rope",
        ),
        (
            "repeat",
            "the",
            "same",
            "overhand",
            "knot",
            "with",
            "the",
            "tail",
            "of",
            "the",
            "other",
            "rope",
            "around",
            "the",
            "first",
            "rope",
        ),
        ("pull", "to", "tighten"),
    ],
    "Zeppelin Bend": [
        ("form", "a", "loop", "in", "each", "rope"),
        ("pass", "one", "loop", "through", "the", "other", "loop"),
        ("dress", "the", "coil"),
        ("pull", "to", "tighten"),
    ],
    "Midshipman's Hitch": [
        ("wrap", "the", "working-end", "around", "the", "standing-part", "twice"),
        ("wrap", "it", "around", "again", "the", "other", "way"),
        ("tuck", "it", "under", "itself", "to", "lock", "the", "hitch"),
        ("pull", "to", "tighten"),
        ("slide", "the", "hitch", "to", "make", "the", "line", "loose", "or", "tight"),
    ],
    # ---- fishing -----------------------------------------------------
    "Improved Clinch Knot": [
        ("pass", "the", "line", "through", "the", "eye", "of", "the", "hook"),
        ("wrap", "the", "tail", "around", "the", "standing-part", "five", "times"),
        ("thread", "the", "tail", "back", "through", "the", "loop", "next", "to", "the", "eye"),
        ("dress", "the", "coil", "and", "pull", "to", "tighten"),
    ],
    "Palomar Knot": [
        ("fold", "the", "line", "into", "a", "bight"),
        ("pass", "the", "bight", "through", "the", "eye", "of", "the", "hook"),
        ("form", "an", "overhand", "knot", "with", "the", "bight"),
        ("pass", "the", "hook", "through", "the", "loop", "of", "the", "bight"),
        ("pull", "to", "tighten"),
    ],
    "Blood Knot": [
        ("wrap", "the", "tail", "of", "one", "line", "around", "the", "other", "line", "five", "times"),
        ("pass", "the", "tail", "back", "through", "the", "middle"),
        ("repeat", "the", "same", "wrap", "with", "the", "tail", "of", "the", "other", "line"),
        ("pull", "to", "tighten"),
    ],
    "Uni Knot": [
        ("pass", "the", "line", "through", "the", "eye", "of", "the", "hook"),
        ("form", "a", "loop", "and", "lay", "it", "over", "the", "standing-part"),
        ("wrap", "the", "tail", "around", "the", "standing-part", "and", "through", "the", "loop", "five", "times"),
        ("pull", "the", "tail", "to", "cinch", "the", "coil"),
        ("dress", "the", "coil", "and", "pull", "to", "tighten"),
    ],
    "Albright Knot": [
        ("fold", "the", "thick", "line", "into", "a", "bight"),
        ("pass", "the", "thin", "line", "through", "the", "bight"),
        ("wrap", "the", "thin", "line", "around", "the", "bight", "and", "itself", "five", "times"),
        ("thread", "the", "thin", "line", "back", "through", "the", "bight"),
        ("pull", "to", "tighten"),
    ],
    "Non-Slip Loop Knot": [
        ("form", "an", "overhand", "knot", "in", "the", "line"),
        ("pass", "the", "tail", "through", "the", "eye", "of", "the", "hook"),
        ("thread", "the", "tail", "back", "through", "the", "overhand", "knot"),
        ("wrap", "the", "tail", "around", "the", "standing-part", "three", "times"),
        ("thread", "the", "tail", "back", "through", "the", "overhand", "knot", "again"),
        ("pull", "to", "tighten"),
    ],
    "Snell Knot": [
        ("pass", "the", "line", "through", "the", "eye", "of", "the", "hook"),
        ("lay", "the", "tail", "over", "the", "hook", "and", "the", "standing-part"),
        ("wrap", "the", "tail", "around", "both", "five", "times"),
        ("pull", "the", "tail", "to", "cinch", "the", "coil"),
        ("dress", "the", "coil", "and", "pull", "to", "tighten"),
    ],
    "Nail Knot": [
        ("lay", "the", "leader", "over", "the", "line"),
        ("wrap", "the", "leader", "around", "the", "line", "five", "times"),
        ("thread", "the", "tail", "back", "through", "the", "coil"),
        ("pull", "to", "tighten"),
    ],
    "Perfection Loop": [
        ("form", "a", "loop", "in", "the", "line"),
        ("wrap", "the", "tail", "around", "the", "standing-part", "to", "form", "a", "loop", "again"),
        ("pass", "one", "loop", "through", "the", "other", "loop"),
        ("pull", "to", "tighten"),
    ],
    "Dropper Loop": [
        ("form", "a", "loop", "in", "the", "middle", "of", "the", "line"),
        ("twist", "the", "loop", "around", "the", "standing-part", "five", "times"),
        ("pass", "the", "loop", "through", "the", "coil"),
        ("pull", "to", "tighten"),
    ],
    # ---- camping -----------------------------------------------------
    "Taut-Line Hitch": [
        ("wrap", "the", "working-end", "around", "the", "standing-part", "twice"),
        ("wrap", "it", "around", "again", "the", "other", "way"),
        ("pull", "to", "tighten"),
        ("slide", "the", "hitch", "to", "make", "the", "line", "loose", "or", "tight"),
    ],
    "Trucker's Hitch": [
        ("form", "a", "loop", "in", "the", "middle", "of", "the", "rope"),
        ("pass", "the", "working-end", "around", "the", "post", "and", "back", "through", "the", "loop"),
        ("pull", "the", "working-end", "to", "cinch", "the", "rope", "tight"),
        ("tuck", "the", "working-end", "under", "itself", "to", "lock", "it"),
        ("pull", "to", "tighten"),
    ],
    "Timber Hitch": [
        ("wrap", "the", "rope", "around", "the", "pole"),
        ("pass", "the", "working-end", "around", "the", "standing-part"),
        ("twist", "the", "working-end", "around", "the", "standing-part", "three", "times"),
        ("pull", "to", "tighten"),
    ],
    "Highwayman's Hitch": [
        ("form", "a", "bight", "in", "the", "rope", "and", "lay", "it", "over", "the", "post"),
        ("pass", "another", "bight", "through", "the", "first", "bight"),
        ("pull", "the", "standing-part", "to", "tighten"),
        ("pull", "the", "tail", "to", "release"),
    ],
    "Siberian Hitch": [
        ("wrap", "the", "rope", "around", "the", "post"),
        ("form", "a", "bight", "and", "tuck", "it", "through", "the", "wrap"),
        ("pull", "to", "tighten"),
        ("pull", "the", "tail", "to", "release"),
    ],
    "Evenk Knot": [
        ("fold", "the", "rope", "into", "a", "bight"),
        ("wrap", "the", "bight", "around", "the", "post", "twice"),
        ("tuck", "the", "tail", "through", "the", "bight", "to", "lock", "it"),
        ("pull", "the", "tail", "to", "release"),
    ],
    "Marlinspike Hitch": [
        ("form", "a", "loop", "in", "the", "rope"),
        ("twist", "the", "loop", "again"),
        ("insert", "the", "spike", "through", "the", "loop"),
        ("pull", "to", "tighten"),
    ],
    "Tripod Lashing": [
        ("lay", "the", "three", "poles", "together"),
        ("wrap", "the", "rope", "around", "all", "three", "poles", "five", "times"),
        ("wrap", "the", "rope", "between", "the", "poles", "to", "lock", "it"),
        ("finish", "with", "a", "clove", "hitch"),
    ],
    "Square Lashing": [
        ("cross", "the", "two", "poles"),
        ("wrap", "the", "rope", "over", "and", "under", "the", "poles", "four", "times"),
        ("wrap", "the", "rope", "between", "the", "poles", "to", "lock", "it"),
        ("finish", "with", "a", "clove", "hitch"),
    ],
    "Diagonal Lashing": [
        ("wrap", "a", "timber", "hitch", "around", "both", "poles"),
        ("wrap", "the", "rope", "over", "and", "under", "the", "poles", "four", "times"),
        ("wrap", "the", "rope", "between", "the", "poles", "to", "lock", "it"),
        ("finish", "with", "a", "clove", "hitch"),
    ],
    "Sheepshank": [
        ("fold", "the", "rope", "into", "a", "bight"),
        ("fold", "the", "rope", "again", "to", "form", "another", "bight"),
        ("wrap", "the", "standing-part", "around", "the", "first", "bight", "and", "tuck", "it", "through"),
        ("wrap", "the", "other", "standing-part", "around", "the", "other", "bight", "and", "tuck", "it", "through"),
        ("pull", "to", "tighten"),
    ],
    "Farmer's Loop": [
        ("twist", "the", "middle", "of", "the", "rope", "to", "form", "a", "coil"),
        ("twist", "it", "again", "to", "form", "another", "coil"),
        ("pull", "the", "middle", "coil", "through", "the", "other", "coil"),
        ("pull", "to", "tighten"),
    ],
    # ---- rescue --------------------------------------------------------
    "Handcuff Knot": [
        ("form", "a", "loop", "in", "the", "rope"),
        ("cross", "it", "to", "form", "another", "loop"),
        ("pass", "each", "loop", "through", "the", "other"),
        ("pull", "to", "tighten"),
    ],
    "Distel Hitch": [
        ("wrap", "the", "cord", "around", "the", "rope", "four", "times"),
        ("tuck", "the", "tail", "under", "itself"),
        ("dress", "the", "coil"),
        ("pull", "to", "check", "that", "it", "jams"),
    ],
    "Blake's Hitch": [
        ("wrap", "the", "cord", "around", "the", "rope", "four", "times"),
        ("weave", "the", "tail", "over", "and", "under", "the", "coil"),
        ("tuck", "the", "tail", "under", "itself"),
        ("pull", "to", "check", "that", "it", "jams"),
    ],
    "Ashley's Stopper Knot": [
        ("form", "a", "loop", "in", "the", "rope"),
        ("wrap", "the", "working-end", "around", "the", "standing-part"),
        ("tuck", "the", "working-end", "through", "the", "loop", "twice"),
        ("pull", "to", "tighten"),
    ],
    "Icicle Hitch": [
        ("wrap", "the", "cord", "around", "the", "pole", "five", "times"),
        ("tuck", "the", "tail", "back", "through", "the", "coil"),
        ("pull", "to", "check", "that", "it", "jams"),
    ],
    "Klemheist Sling": [
        ("wrap", "the", "sling", "around", "the", "rope", "three", "times"),
        ("pass", "the", "bight", "of", "the", "sling", "through", "the", "coil"),
        ("dress", "the", "coil"),
        ("pull", "to", "check", "that", "it", "jams"),
    ],
    "Butterfly Bend": [
        ("form", "a", "loop", "in", "each", "rope"),
        ("pass", "one", "loop", "through", "the", "other", "loop"),
        ("dress", "the", "coil"),
        ("pull", "to", "tighten"),
    ],
    "Double Overhand Bend": [
        (
            "wrap",
            "the",
            "tail",
            "of",
            "one",
            "rope",
            "around",
            "the",
            "other",
            "rope",
            "twice",
            "and",
            "tuck",
            "it",
            "through",
            "itself",
        ),
        ("repeat", "the", "same", "wrap", "and", "tuck", "with", "the", "tail", "of", "the", "other", "rope"),
        ("pull", "to", "tighten"),
    ],
    # ---- decorative ----------------------------------------------------
    "Monkey Fist": [
        ("wrap", "the", "rope", "around", "itself", "three", "times"),
        ("wrap", "it", "around", "the", "same", "coil", "again", "the", "other", "way"),
        ("dress", "the", "coil"),
        ("pull", "the", "ends", "to", "tighten"),
    ],
    "Turk's Head": [
        ("wrap", "the", "cord", "around", "the", "pole", "three", "times"),
        ("weave", "the", "tail", "over", "and", "under", "the", "coil"),
        ("follow", "the", "same", "weave", "around", "again"),
        ("dress", "the", "coil", "and", "pull", "to", "tighten"),
    ],
    "Chain Sinnet": [
        ("form", "a", "bight", "in", "the", "rope"),
        ("pull", "another", "bight", "through", "the", "first", "bight"),
        ("repeat", "the", "same", "pull", "again", "and", "again"),
        ("thread", "the", "tail", "through", "the", "loop", "to", "finish"),
    ],
    "Crown Knot": [
        ("take", "the", "three", "strands", "at", "the", "end", "of", "the", "rope"),
        ("cross", "each", "strand", "over", "the", "next", "strand"),
        ("pull", "the", "strands", "to", "tighten"),
    ],
    "Matthew Walker Knot": [
        ("take", "the", "three", "strands", "at", "the", "end", "of", "the", "rope"),
        ("wrap", "each", "strand", "around", "the", "standing-part"),
        ("tuck", "each", "strand", "through", "the", "next", "loop"),
        ("pull", "the", "strands", "to", "tighten"),
    ],
    "Wall Knot": [
        ("take", "the", "three", "strands", "at", "the", "end", "of", "the", "rope"),
        ("tuck", "each", "strand", "under", "the", "next", "strand"),
        ("pull", "the", "strands", "to", "tighten"),
    ],
}

HOW_TO_TEMPLATES: list[str] = [
    "how do I tie a {knot}?",
    "show me how to tie a {knot}",
    "what are the steps for a {knot}?",
    "how is a {knot} tied?",
    "can you explain how to tie a {knot}?",
    "walk me through tying a {knot}",
]

# Each entry: scenario phrase, knot name, one-sentence justification as a
# word tuple.
WHICH_KNOT: list[tuple[str, str, tuple[str, ...]]] = [
    (
        "join two rope of different diameter",
        "Sheet Bend",
        ("sheet", "bend", "it", "is", "reliable", "for", "rope", "of", "different", "diameter"),
    ),
    (
        "make a fixed loop that will not slip",
        "Bowline",
        ("bowline", "it", "makes", "a", "strong", "loop", "that", "will", "not", "slip"),
    ),
    (
        "attach a rope to a post or rail",
        "Clove Hitch",
        ("clove", "hitch", "it", "is", "quick", "to", "tie", "around", "a", "post", "or", "rail"),
    ),
    (
        "make an adjustable loop for a tarp guyline",
        "Taut-Line Hitch",
        ("taut-line", "hitch", "it", "is", "adjustable", "and", "useful", "for", "a", "tarp", "guyline"),
    ),
    (
        "back up a rappel on a climbing rope",
        "Prusik Knot",
        ("prusik", "knot", "it", "jams", "on", "the", "rope", "and", "is", "reliable", "for", "a", "rappel"),
    ),
    (
        "tie fishing line to a hook",
        "Improved Clinch Knot",
        ("improved", "clinch", "knot", "it", "is", "strong", "and", "popular", "to", "tie", "line", "to", "a", "hook"),
    ),
    (
        "make a decorative weight on the end of a throw line",
        "Monkey Fist",
        ("monkey", "fist", "it", "is", "decorative", "and", "can", "add", "weight", "to", "a", "line"),
    ),
    (
        "join two climbing ropes before a rappel",
        "Double Fisherman's Knot",
        (
            "double",
            "fisherman's",
            "knot",
            "it",
            "is",
            "strong",
            "and",
            "reliable",
            "to",
            "join",
            "two",
            "rope",
            "before",
            "a",
            "rappel",
        ),
    ),
    (
        "join two rope of the same diameter",
        "Fisherman's Knot",
        (
            "fisherman's",
            "knot",
            "it",
            "is",
            "simple",
            "and",
            "reliable",
            "for",
            "rope",
            "of",
            "the",
            "same",
            "diameter",
        ),
    ),
    (
        "tie a quick-release hitch for a guyline",
        "Siberian Hitch",
        ("siberian", "hitch", "it", "is", "quick-release", "and", "useful", "for", "a", "guyline"),
    ),
    (
        "control a rope without a mechanical device",
        "Munter Hitch",
        ("munter", "hitch", "it", "is", "versatile", "and", "useful", "with", "a", "carabiner", "for", "climbing"),
    ),
    (
        "lash three poles together into a tripod",
        "Tripod Lashing",
        ("tripod", "lashing", "it", "is", "essential", "to", "join", "three", "poles", "together"),
    ),
    (
        "shorten a rope without cutting it",
        "Sheepshank",
        ("sheepshank", "it", "is", "a", "classic", "temporary", "way", "to", "shorten", "a", "rope"),
    ),
    (
        "attach a sling to a climbing harness",
        "Girth Hitch",
        ("girth", "hitch", "it", "is", "simple", "and", "reliable", "to", "join", "a", "sling", "to", "a", "harness"),
    ),
    (
        "attach a rope to a hook for hauling",
        "Anchor Bend",
        ("anchor", "bend", "it", "is", "secure", "and", "reliable", "for", "load-bearing", "use"),
    ),
    (
        "grip a rope under load with a cord",
        "Distel Hitch",
        ("distel", "hitch", "it", "jams", "on", "the", "rope", "and", "is", "easy", "to", "release"),
    ),
]

WHICH_KNOT_TEMPLATES: list[str] = [
    "what knot should I use to {scenario}?",
    "which knot works best to {scenario}?",
    "what is a good knot for {scenario}?",
    "can you recommend a knot to {scenario}?",
    "which knot would you use to {scenario}?",
    "what knot do you suggest to {scenario}?",
]

# The capability style answers "what do you know about?" with knots.json's
# own category list (built in build_examples(), not hardcoded here, so it
# cannot drift from the taxonomy file -- same reasoning as
# build_word_list.py pulling the category words from the same place).
CAPABILITY_TEMPLATES: list[str] = [
    "what type of knots do you know?",
    "what knots can you help with?",
    "what categories of knots do you know?",
    "what kinds of knots do you know about?",
    "what topics do you cover?",
    "what areas of knot tying do you know?",
    "what can you tell me about?",
]

# The category style answers "tell me about X knots" with the names of every
# knot knots.json files under that category (built in build_examples(), not
# hardcoded, for the same drift reason as CAPABILITY_TEMPLATES above).
# {category} is filled with knots.json's category strings verbatim (already
# lowercase), which is also why "general" reads a little flatly as a
# question -- it is the literal taxonomy label, not a rewritten noun phrase.
CATEGORY_TEMPLATES: list[str] = [
    "tell me about {category} knots",
    "what {category} knots do you know?",
    "what knots are used for {category}?",
    "list some {category} knots",
    "which knots are good for {category}?",
    "what knots do you know for {category}?",
]


def steps_to_answer(steps: list[tuple[str, ...]]) -> list[str]:
    """Flatten a KNOT_STEPS entry into one answer_words list, "." after each step."""
    words = []
    for step in steps:
        words.extend(step)
        words.append(".")
    return words


def build_examples() -> list[dict]:
    """Build every {style, knot, question, answer_words} example, all four styles."""
    taxonomy = json.loads((HERE / "knots.json").read_text())
    examples = []
    for knot, steps in KNOT_STEPS.items():
        answer = steps_to_answer(steps)
        for template in HOW_TO_TEMPLATES:
            examples.append(
                {
                    "style": "how_to_tie",
                    "knot": knot,
                    "question": template.format(knot=knot.lower()),
                    "answer_words": answer,
                }
            )
    for scenario, knot, justification in WHICH_KNOT:
        answer = list(justification) + ["."]
        for template in WHICH_KNOT_TEMPLATES:
            examples.append(
                {
                    "style": "which_knot",
                    "knot": knot,
                    "question": template.format(scenario=scenario),
                    "answer_words": answer,
                }
            )
    # One word per category, comma-separated, matching the terse telegraphic
    # style every other answer already uses rather than a full sentence --
    # avoids needing a first-person pronoun or a verb like "know" in the
    # output vocabulary just for this one question style.
    categories = taxonomy["categories"]
    answer = []
    for i, category in enumerate(categories):
        answer.append(category)
        answer.append("," if i < len(categories) - 1 else ".")
    for template in CAPABILITY_TEMPLATES:
        examples.append(
            {
                "style": "capability",
                "knot": None,
                "question": template,
                "answer_words": answer,
            }
        )
    # One category answers with the names of every knot filed under it in
    # knots.json, comma-separated between knots (not between the words of one
    # multi-word name), period at the end -- same shape as the capability
    # answer above, one level down the taxonomy.
    by_category = {c: [] for c in categories}
    for k in taxonomy["knots"]:
        by_category[k["category"]].append(k["name"].lower())
    for category, names in by_category.items():
        answer = []
        for i, name in enumerate(names):
            answer.extend(name.split(" "))
            answer.append("," if i < len(names) - 1 else ".")
        for template in CATEGORY_TEMPLATES:
            examples.append(
                {
                    "style": "category",
                    "knot": None,
                    "question": template.format(category=category),
                    "answer_words": answer,
                }
            )
    return examples


def main() -> None:
    """Build every example, vocabulary-check it, then write qa_pairs.jsonl."""
    vocab = set(json.loads((HERE / "output_words.json").read_text())["words"])
    known_knots = {k["name"] for k in json.loads((HERE / "knots.json").read_text())["knots"]}

    missing_knots = known_knots - set(KNOT_STEPS)
    extra_knots = set(KNOT_STEPS) - known_knots
    if extra_knots:
        raise SystemExit(f"KNOT_STEPS has knots not in knots.json: {sorted(extra_knots)}")

    examples = build_examples()

    bad = []
    for ex in examples:
        for w in ex["answer_words"]:
            if w not in vocab:
                bad.append((ex["question"], w))
    if bad:
        lines = "\n".join(f"  {q!r}: {w!r}" for q, w in bad)
        raise SystemExit(f"{len(bad)} answer word(s) fall outside output_words.json:\n{lines}")

    path = HERE / "qa_pairs.jsonl"
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    n_categories = len(json.loads((HERE / "knots.json").read_text())["categories"])
    print(
        f"wrote {path}: {len(examples)} examples "
        f"({len(KNOT_STEPS)}/{len(known_knots)} knots covered x "
        f"{len(HOW_TO_TEMPLATES)} phrasings, {len(WHICH_KNOT)} which_knot "
        f"facts x {len(WHICH_KNOT_TEMPLATES)} phrasings, 1 capability fact x "
        f"{len(CAPABILITY_TEMPLATES)} phrasings, {n_categories} categories x "
        f"{len(CATEGORY_TEMPLATES)} phrasings), all vocabulary-clean"
    )
    if missing_knots:
        print(f"not yet covered ({len(missing_knots)}): {sorted(missing_knots)}")


if __name__ == "__main__":
    main()
