"""Generate the bundled sample PDF: a synthetic handbook for a fictional public library.

All names, numbers, phone numbers and addresses are invented for demonstration.
Run from the backend folder:  python scripts/make_sample_pdf.py
"""

from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parent.parent / "sample_data" / "harbor-city-library-handbook.pdf"

PAGES: list[tuple[str, list[str]]] = [
    (
        "Harbor City Public Library: Member Handbook (2026 edition)",
        [
            "This handbook explains how to use the Harbor City Public Library. Harbor City is a fictional town, "
            "and this document is synthetic sample content created to demonstrate DocuMind AI.",
            "The library system has four branches: Central Library, Northshore, Eastgate and Mill Creek. "
            "Central Library opened in 1962 and is the largest branch, holding about 180,000 items. "
            "Mill Creek is the newest branch; it opened in September 2019 inside the Mill Creek Community Center.",
            "Our mission is to give every resident free access to books, information and quiet space to learn. "
            "The library is governed by a seven-member Board of Trustees that meets on the second Tuesday of each month.",
        ],
    ),
    (
        "Library cards",
        [
            "Library cards are free for anyone who lives, works or attends school in Harbor City. "
            "To get a card, bring a photo ID and proof of a local address, such as a utility bill dated within the last 60 days.",
            "Children under 13 need a parent or guardian to sign their card application. "
            "Teens aged 13 to 17 can sign up on their own with a school ID.",
            "People who live outside Harbor City can buy a non-resident card for $35 per year. "
            "A replacement for a lost card costs $2. Cards must be renewed every three years so we can confirm your address.",
        ],
    ),
    (
        "Borrowing and renewals",
        [
            "Most books can be borrowed for 21 days. DVDs and video games are loaned for 7 days, "
            "and magazines for 14 days. Reference books and the newest issue of each magazine stay in the library.",
            "You can have up to 30 items checked out at once, including no more than 10 DVDs.",
            "Items can be renewed up to two times, online or at any desk, as long as no one else has placed a hold on them. "
            "When an item you placed on hold arrives, we keep it on the hold shelf for 5 days before passing it on.",
            "Items can be returned to any branch or to the 24-hour book drop outside Central Library.",
        ],
    ),
    (
        "Fines, fees and lost items",
        [
            "Harbor City Public Library has been fine-free since March 2023: there are no daily fines for overdue items.",
            "An item that is 45 days overdue is marked as lost, and the account holder is billed the item's replacement cost. "
            "If the item is returned in good condition within six months, the charge is removed.",
            "Borrowing is paused when an account owes $25 or more in fees. Damaged items are charged at replacement cost, "
            "and a $5 processing fee applies to each lost or damaged item.",
        ],
    ),
    (
        "Computers, printing and study rooms",
        [
            "Free Wi-Fi is available at every branch on the network named HCPL-Public; no password is required. "
            "Public computers can be booked in 60-minute sessions, and a session can be extended if no one is waiting.",
            "Printing costs 10 cents per page for black and white and 50 cents per page for color. "
            "Scanning to email is free.",
            "Study rooms seat up to six people and can be booked for up to two hours per day, up to 14 days in advance. "
            "Rooms not claimed within 15 minutes of the booking start are released.",
            "The Makerspace at Central Library has two 3D printers. Members must complete a free 45-minute safety "
            "orientation before using them, and printing costs 25 cents per gram of filament.",
        ],
    ),
    (
        "Opening hours and contact",
        [
            "Central Library is open Monday to Thursday from 9 a.m. to 8 p.m., Friday and Saturday from 9 a.m. to 6 p.m., "
            "and Sunday from 1 p.m. to 5 p.m.",
            "Northshore, Eastgate and Mill Creek are open Tuesday to Saturday from 10 a.m. to 6 p.m. and are closed on Sundays and Mondays.",
            "All branches close on New Year's Day, Independence Day, Thanksgiving Day and Christmas Day.",
            "For help, call the Central Library information desk at (555) 010-4422 or email help@harborcitylibrary.example.",
        ],
    ),
    (
        "Privacy and conduct",
        [
            "We keep your borrowing history only until an item is returned, unless you turn on the optional reading-history "
            "feature in your account. Records of unpaid charges are kept until the charge is resolved.",
            "Covered drinks are welcome in all public areas. Food is allowed only in the Central Library cafe and in study rooms.",
            "Service animals are welcome in all branches. Please keep phone calls in the lobby so others can study.",
        ],
    ),
]


def build(out: Path = OUT) -> Path:
    doc = pymupdf.open()
    width, height, margin = 612, 792, 72
    for number, (heading, paragraphs) in enumerate(PAGES, start=1):
        page = doc.new_page(width=width, height=height)
        y = margin
        page.insert_textbox(pymupdf.Rect(margin, y, width - margin, y + 60), heading, fontsize=17, fontname="hebo")
        y += 48
        for paragraph in paragraphs:
            rect = pymupdf.Rect(margin, y, width - margin, height - margin)
            remaining = page.insert_textbox(rect, paragraph, fontsize=11, fontname="helv", lineheight=1.35)
            used = rect.height - remaining
            y += used + 12
        page.insert_text((margin, height - 40), f"Harbor City Public Library handbook  |  page {number}", fontsize=8, fontname="helv", color=(0.45, 0.45, 0.45))
    doc.set_metadata({"title": "Harbor City Public Library: Member Handbook", "author": "DocuMind AI sample (synthetic)"})
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    doc.close()
    return out


if __name__ == "__main__":
    print(f"Wrote {build()}")
