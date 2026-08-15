"""The index's own counts, against the index.

`docs/reports/README.md` opens by saying how many questions the record answers and how many
of them belong to phase 4, and then lists them. The two went out of step twice: questions 19
and 20 were added, each with a row and a detail section and a link that resolved, and the
sentence above them still said eighteen. Nothing was wrong with either new entry -- the
defect was in a paragraph neither commit touched, which is exactly the kind a reader trusts
and a reviewer skims.

So the counts are asserted against what is under them. The prose stays prose; what it claims
is arithmetic, and arithmetic is checkable.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX = REPO_ROOT / "docs" / "reports" / "README.md"

#: Only as far as the record is likely to reach. A count that outgrows this fails loudly on
#: the lookup rather than quietly matching nothing.
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty-one": 21,
    "twenty-two": 22,
    "twenty-three": 23,
    "twenty-four": 24,
    "twenty-five": 25,
    "twenty-six": 26,
    "twenty-seven": 27,
    "twenty-eight": 28,
    "twenty-nine": 29,
    "thirty": 30,
}


@pytest.fixture(scope="module")
def index() -> str:
    return INDEX.read_text(encoding="utf-8")


def _spelled(text: str, pattern: str, what: str) -> int:
    """The integer behind one spelled-out count in the intro.

    The pattern is matched against the intro with its line breaks flattened, because the
    sentence wraps at 100 columns and a rewrap must not be able to turn this test green by
    making the claim unfindable -- a missing match is a failure, not a skip.
    """
    flat = " ".join(text.split())
    match = re.search(pattern, flat)
    assert match, f"the intro no longer states {what}; it must, or this test guards nothing"
    word = match.group(1).lower()
    assert word in NUMBER_WORDS, f"unmapped number word {word!r} for {what}"
    return NUMBER_WORDS[word]


def _rows(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"^\| (\d+) \|", text, flags=re.MULTILINE)]


def _sections(text: str) -> dict[int, str]:
    heads = re.findall(r"^## (\d+)\. (.+)$", text, flags=re.MULTILINE)
    return {int(number): title for number, title in heads}


def test_the_intro_counts_the_questions_that_are_actually_listed(index: str) -> None:
    """Turns red when: a question is added or removed without the sentence following it."""
    rows = _rows(index)
    assert len(rows) > 1, "no numbered rows parsed -- the table format changed"
    assert rows == list(range(1, len(rows) + 1)), f"rows are not 1..n in order: {rows}"
    stated = _spelled(index, r"They answer (\S+) questions", "how many questions it answers")
    assert stated == len(rows), f"the intro says {stated} questions and the table lists {len(rows)}"


def test_the_intro_counts_phase_fours_share_of_them_twice(index: str) -> None:
    """The sentence states the phase-4 count twice -- both must match the sections.

    It opens with "phase 4 answers N of them" and closes by calling the enumerated failures
    "N separate failures". Two numbers for one quantity is two chances to go stale, and the
    second is the one a writer extending the list forgets.

    Turns red when: a phase-4 question is added and either count is left behind.
    """
    sections = _sections(index)
    phase4 = sorted(number for number, title in sections.items() if title.startswith("Phase 4"))
    assert phase4, "no phase-4 sections found -- the heading format changed"
    opening = _spelled(index, r"phase 4 answers (\S+) of them", "phase 4's share")
    closing = _spelled(index, r"are (\S+) separate failures", "the count of enumerated failures")
    assert opening == len(phase4), (
        f"the intro says phase 4 answers {opening}; {len(phase4)} sections are phase 4: {phase4}"
    )
    assert closing == len(phase4), (
        f"the intro enumerates {closing} failures for {len(phase4)} phase-4 questions"
    )


def _campaigns(text: str) -> list[str]:
    """One entry per distinct record the table points at, in the order first cited.

    A campaign is a document, not a question: six rows cite `phase4-text2sql-mixture.md` and
    they are one campaign between them. So the last cell of each row is reduced to its first
    link target -- the section markers that distinguish those six sit outside the link, which
    is why deduplicating on the target is the same thing as deduplicating on the record.
    """
    seen: list[str] = []
    for line in text.splitlines():
        if not re.match(r"^\| \d+ \|", line):
            continue
        cell = line.rstrip("|").rsplit("|", 1)[-1]
        target = re.search(r"\]\(([^)]+)\)", cell)
        assert target, f"a numbered row cites no record: {line[:60]!r}"
        if target.group(1) not in seen:
            seen.append(target.group(1))
    return seen


def test_the_intro_counts_the_campaigns_the_table_cites(index: str) -> None:
    """The third count in that sentence, and the one the other two tests do not reach.

    It was already stale when this test was written: the intro said fifteen over sixteen
    records, because a campaign can be added without adding a question and the question count
    is what a writer checks. Counting rows or sections cannot catch that -- both were correct
    -- so the quantity has to be derived from the column nobody edits when extending the prose.

    Turns red when: a new report is linked from the table without the sentence following it,
    or a row is repointed at a record that already had one and the count is not reduced.
    """
    campaigns = _campaigns(index)
    assert len(campaigns) > 1, "no records parsed from the table -- the last column changed"
    stated = _spelled(index, r"There are (\S+) campaigns", "how many campaigns there are")
    assert stated == len(campaigns), (
        f"the intro says {stated} campaigns and the table cites {len(campaigns)}: {campaigns}"
    )


def test_every_question_has_a_row_and_a_section(index: str) -> None:
    """Turns red when: a row is added without its write-up, or a write-up without its row."""
    rows = set(_rows(index))
    sections = set(_sections(index))
    assert rows == sections, (
        f"rows without a section: {sorted(rows - sections)}; "
        f"sections without a row: {sorted(sections - rows)}"
    )


def test_every_link_in_the_index_resolves(index: str) -> None:
    """Turns red when: a report is renamed or moved and the index still points at where it was.

    The index is the only file that links every other one, so a rename that misses it leaves
    a record no reader can reach from the entry point -- and markdown links do not fail loudly.
    """
    targets = set(re.findall(r"\]\(([^)]+)\)", index))
    assert len(targets) > 10, f"only {len(targets)} links parsed -- the link syntax changed"
    missing = sorted(
        target
        for target in targets
        if not target.startswith(("http", "#", "mailto:"))
        and not (INDEX.parent / target.split("#")[0]).exists()
    )
    assert not missing, f"index links point at nothing: {missing}"
