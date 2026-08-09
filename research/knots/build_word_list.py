"""Assemble the curated output word list for the knots model.

This is the vocabulary-first step: the word list produced here is the hard
constraint every generated answer in qa_pairs.jsonl must be built from
(enforced later by build_dataset.py's conformance check). It is deliberately
NOT vocab.json/layout.json yet -- those need out2in, which needs the trained
input tokenizer (prepare.py), which needs the generated corpus text. This
script only produces the word list; assembling the final barista-shaped
vocab.json/layout.json is build_vocab.py, run after the tokenizer exists.

Three sources, merged and deduplicated:
  - every word in every canonical knot name in knots.json (split on
    whitespace; internal hyphens/apostrophes kept as part of the word, e.g.
    "figure-eight", "surgeon's"), lowercased. Extracted rather than
    hand-transcribed so it cannot drift from the taxonomy file.
  - every category in knots.json's top-level "categories" list (e.g.
    "general", "climbing"). generate_corpus.py's capability answer ("what
    type of knots do you know?") is built from that same list, so pulling
    the words from here too means the two cannot disagree about what a
    category is called.
  - a hand-authored supplement: specials, punctuation, function words,
    bounded number words, rope-anatomy nouns, action verbs, property/
    justification words, and scenario nouns, per research/knots/README.md's
    question taxonomy.

  uv run python research/knots/build_word_list.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

PUNCTUATION = [".", ",", ";", ":"]

# Bounded number words, barista-style: spelled out so the output alphabet
# never needs digit tokens. Only as many as procedures plausibly need (wrap
# counts, turn counts).
NUMBERS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "twice",
    "double",
    "again",
]

FUNCTION_WORDS = [
    "a",
    "an",
    "the",
    "to",
    "of",
    "in",
    "on",
    "with",
    "around",
    "through",
    "over",
    "under",
    "back",
    "then",
    "and",
    "or",
    "for",
    "from",
    "at",
    "into",
    "out",
    "off",
    "up",
    "down",
    "this",
    "that",
    "it",
    "its",
    "itself",
    "other",
    "another",
    "your",
    "you",
    "before",
    "after",
    "when",
    "so",
    "is",
    "are",
    "be",
    "not",
    "no",
    "yes",
    "both",
    "either",
    "need",
    "make",
    "makes",
    "use",
    "used",
    "using",
    "called",
    "also",
    "which",
    "most",
    "first",
    "next",
    "finally",
    "same",
    "each",
    "way",
    "as",
    "if",
    "can",
    "will",
    "should",
    "times",
    "middle",
    "between",
    "all",
    "together",
]

# "ends", "poles" and "strands" are the only plural nouns in the vocabulary:
# lashings (multiple poles) and strand-work knots (crown, wall, matthew
# walker) genuinely need to refer to more than one at once, unlike
# everything else here which stays singular by rephrasing around it.
ROPE_ANATOMY = [
    "rope",
    "line",
    "cord",
    "webbing",
    "sling",
    "strand",
    "strands",
    "end",
    "ends",
    "working-end",
    "standing-end",
    "standing-part",
    "tail",
    "bight",
    "loop",
    "turn",
    "round-turn",
    "coil",
    "eye",
    "elbow",
    "knot",
    "hitch",
    "bend",
    "spike",
]

ACTIONS = [
    "tie",
    "form",
    "take",
    "hold",
    "pass",
    "cross",
    "wrap",
    "coil",
    "bring",
    "fold",
    "tuck",
    "thread",
    "feed",
    "pull",
    "draw",
    "tighten",
    "dress",
    "snug",
    "cinch",
    "twist",
    "lay",
    "place",
    "insert",
    "run",
    "lead",
    "finish",
    "secure",
    "lock",
    "seize",
    "whip",
    "follow",
    "retrace",
    "mirror",
    "weave",
    "repeat",
    "check",
    "load",
    "slide",
    "join",
    "add",
    "release",
    "shorten",
]

PROPERTIES = [
    "strong",
    "secure",
    "quick-release",
    "adjustable",
    "non-slipping",
    "bulky",
    "jams",
    "reliable",
    "versatile",
    "classic",
    "traditional",
    "decorative",
    "load-bearing",
    "permanent",
    "temporary",
    "simple",
    "popular",
    "essential",
    "useful",
    "general-purpose",
    "critical",
    "tight",
    "loose",
    "easy",
    "quick",
    "smooth",
    "thick",
    "thin",
]

SCENARIOS = [
    "climbing",
    "sailing",
    "boating",
    "fishing",
    "camping",
    "rescue",
    "tarp",
    "tent",
    "guyline",
    "post",
    "ring",
    "rail",
    "dock",
    "harness",
    "lure",
    "hook",
    "pole",
    "poles",
    "mast",
    "boat",
    "sail",
    "diameter",
    "anchor",
    "mountaineering",
    "hiking",
    "backpacking",
    "scouting",
    "emergency",
    "safety",
    "weight",
    "carabiner",
    "rappel",
    "leader",
    "different",
    "same",
]

SUPPLEMENT = PUNCTUATION + NUMBERS + FUNCTION_WORDS + ROPE_ANATOMY + ACTIONS + PROPERTIES + SCENARIOS


def name_words(name: str) -> list[str]:
    """Split a canonical knot name into its lowercase component words."""
    return [w.lower() for w in name.split(" ")]


def main() -> None:
    """Assemble output_words.json from knots.json's names/categories plus SUPPLEMENT."""
    taxonomy = json.loads((HERE / "knots.json").read_text())
    from_names = []
    for k in taxonomy["knots"]:
        from_names.extend(name_words(k["name"]))
    categories = list(taxonomy["categories"])

    # Dedupe, preserving first-seen order within each source, names before
    # categories before supplement so an earlier source keeps the slot on
    # a collision (e.g. "hitch", or "climbing" appearing in both categories
    # and the hand-authored scenario nouns).
    seen = set(SPECIALS)
    ordered = list(SPECIALS)
    for word in from_names + categories + SUPPLEMENT:
        if word not in seen:
            seen.add(word)
            ordered.append(word)

    out = {
        "_comment": (
            "Curated output word list for the knots model. Specials first "
            "(4), then words derived from knots.json's canonical names, "
            "then knots.json's category names, then the hand-authored "
            "supplement (punctuation, numbers, function words, rope "
            "anatomy, actions, properties, scenario nouns). Every answer "
            "word in qa_pairs.jsonl must come from this list -- see "
            "build_dataset.py's conformance check."
        ),
        "total": len(ordered),
        "words": ordered,
    }
    path = HERE / "output_words.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"wrote {path}: {len(ordered)} words "
        f"({len(from_names)} raw from names, {len(categories)} categories, "
        f"{len(SUPPLEMENT)} supplement, {len(ordered) - len(SPECIALS)} "
        f"unique after specials)"
    )


if __name__ == "__main__":
    main()
