from app.generation.citations import validate_citations


def test_valid_citations_pass_through_unchanged():
    answer = "Retries use exponential backoff [S1]. Timeouts default to 30s [S2]."
    report = validate_citations(answer, num_sources=2)

    assert report.valid_tags == [1, 2]
    assert report.invalid_tags == []
    assert report.cleaned_text == answer
    assert report.hallucination_rate == 0.0


def test_out_of_range_citation_is_stripped_and_flagged():
    answer = "The client retries automatically [S1]. It also caches results [S9]."
    report = validate_citations(answer, num_sources=2)

    assert report.valid_tags == [1]
    assert report.invalid_tags == [9]
    assert "[S9]" not in report.cleaned_text
    assert "[S1]" in report.cleaned_text
    assert report.hallucination_rate == 0.5


def test_no_citations_gives_zero_rate_not_division_error():
    report = validate_citations("I don't know based on the provided sources.", num_sources=3)

    assert report.total_citations == 0
    assert report.hallucination_rate == 0.0
