"""Citation grounding: verify every [Sn] tag the model emits actually maps
to a source we retrieved, and strip the ones that don't.

This is the concrete mechanism behind "reduce hallucinated citations" -
the model is prompted to only cite listed sources (see prompt.py), and this
module is the backstop that catches it when it doesn't.
"""
import re
from dataclasses import dataclass

_CITATION_RE = re.compile(r"\[S(\d+)\]")


@dataclass
class CitationReport:
    cleaned_text: str
    valid_tags: list[int]
    invalid_tags: list[int]

    @property
    def total_citations(self) -> int:
        return len(self.valid_tags) + len(self.invalid_tags)

    @property
    def hallucination_rate(self) -> float:
        if self.total_citations == 0:
            return 0.0
        return len(self.invalid_tags) / self.total_citations


def validate_citations(answer_text: str, num_sources: int) -> CitationReport:
    valid_tags: list[int] = []
    invalid_tags: list[int] = []

    def _replace(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= num_sources:
            valid_tags.append(n)
            return match.group(0)
        invalid_tags.append(n)
        return ""

    cleaned_text = _CITATION_RE.sub(_replace, answer_text)
    cleaned_text = re.sub(r"[ \t]+([.,])", r"\1", cleaned_text)  # tidy up stray spaces left by removal
    return CitationReport(cleaned_text=cleaned_text, valid_tags=valid_tags, invalid_tags=invalid_tags)
