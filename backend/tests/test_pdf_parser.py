import pytest

from app.errors import InvalidUpload, UnprocessableDocument
from app.pdf_parser import extract_pages, validate_upload

from conftest import HANDBOOK_PAGES, make_encrypted_pdf, make_pdf, make_scanned_pdf

MB = 1024 * 1024


def test_page_numbers_are_preserved(handbook_pdf):
    parsed = extract_pages(handbook_pdf, max_pages=10)
    assert parsed.page_count == 3
    assert [p.number for p in parsed.pages] == [1, 2, 3]
    assert "founded in 1987" in parsed.pages[0].text
    assert "40 dollars" in parsed.pages[1].text
    assert "Helmets are mandatory" in parsed.pages[2].text


def test_blank_pages_keep_their_position():
    parsed = extract_pages(make_pdf(["First page has text here.", "", "Third page text is here."]), max_pages=10)
    assert [p.number for p in parsed.pages] == [1, 2, 3]
    assert parsed.pages[1].text == ""
    assert "Third page" in parsed.pages[2].text


def test_line_wrap_hyphenation_is_repaired():
    long_text = "The committee reviewed the infra-\nstructure budget for next year carefully."
    parsed = extract_pages(make_pdf([long_text]), max_pages=10)
    assert "infrastructure" in parsed.pages[0].text


def test_running_footers_are_removed_but_body_text_kept():
    import pymupdf

    bodies = ["Revenue grew in the northern region.", "Costs fell after the merger.", "Hiring slowed in spring.", "The board approved a dividend."]
    doc = pymupdf.open()
    for n, body in enumerate(bodies, start=1):
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 540, 400), body)
        page.insert_text((72, 760), f"Acme Corp annual report | page {n}")
    parsed = extract_pages(doc.tobytes(), max_pages=10)
    assert all("Acme Corp annual report" not in p.text for p in parsed.pages)
    assert [p.text for p in parsed.pages] == bodies


def test_footer_detection_needs_three_pages():
    line = "This identical short line appears on both pages."
    parsed = extract_pages(make_pdf([line, line]), max_pages=10)
    assert all(p.text == line for p in parsed.pages)


def test_empty_pdf_is_rejected():
    with pytest.raises(UnprocessableDocument) as err:
        extract_pages(make_pdf(["", ""]), max_pages=10)
    assert err.value.code == "no_text"


def test_scanned_pdf_is_detected():
    with pytest.raises(UnprocessableDocument) as err:
        extract_pages(make_scanned_pdf(), max_pages=10)
    assert err.value.code == "scanned_pdf"
    assert "OCR" in err.value.message


def test_partially_scanned_pdf_reports_image_pages():
    import pymupdf

    doc = pymupdf.open(stream=make_pdf(HANDBOOK_PAGES), filetype="pdf")
    doc.insert_pdf(pymupdf.open(stream=make_scanned_pdf(1), filetype="pdf"))
    parsed = extract_pages(doc.tobytes(), max_pages=10)
    assert parsed.image_only_pages == [4]


def test_encrypted_pdf_is_rejected():
    with pytest.raises(UnprocessableDocument) as err:
        extract_pages(make_encrypted_pdf(), max_pages=10)
    assert err.value.code == "encrypted_pdf"


def test_corrupted_pdf_is_rejected():
    with pytest.raises(UnprocessableDocument) as err:
        extract_pages(b"%PDF-1.7\n this is not really a pdf \x00\x01", max_pages=10)
    assert err.value.code == "corrupted_pdf"


def test_page_limit_is_enforced():
    with pytest.raises(UnprocessableDocument) as err:
        extract_pages(make_pdf(["text page"] * 5), max_pages=3)
    assert err.value.code == "too_many_pages"


@pytest.mark.parametrize(
    ("filename", "data", "code"),
    [
        ("notes.txt", b"%PDF-1.7 ...", "invalid_upload"),
        (None, b"%PDF-1.7 ...", "invalid_upload"),
        ("empty.pdf", b"", "invalid_upload"),
        ("image.pdf", b"\x89PNG\r\n\x1a\n....", "invalid_upload"),
        ("big.pdf", "OVERSIZED", "file_too_large"),
    ],
    ids=["wrong-extension", "no-filename", "empty", "not-a-pdf", "too-large"],
)
def test_validate_upload_rejects_bad_files(filename, data, code):
    if data == "OVERSIZED":
        data = b"%PDF-" + b"0" * (2 * MB)
    with pytest.raises(InvalidUpload) as err:
        validate_upload(filename, data, max_bytes=1 * MB)
    assert err.value.code == code


def test_validate_upload_accepts_pdf_with_uppercase_extension(handbook_pdf):
    validate_upload("REPORT.PDF", handbook_pdf, max_bytes=1 * MB)
