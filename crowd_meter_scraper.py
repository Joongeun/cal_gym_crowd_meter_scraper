import csv
import os
from datetime import datetime
from playwright.sync_api import sync_playwright

URL = "https://recwell.berkeley.edu/facilities/recreational-sports-facility-rsf/rsf-weight-room-crowd-meter/"
MHTML_FILE = "page.mhtml"
CSV_FILE = "crowd_meter_data.csv"

def save_page_as_mhtml(url, mhtml_file):
    """Save a webpage as MHTML using Playwright."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto(url, wait_until="networkidle")

        # Use CDP to capture MHTML
        session = context.new_cdp_session(page)
        mhtml_data = session.send("Page.captureSnapshot", {"format": "mhtml"})["data"]

        with open(mhtml_file, "w", encoding="utf-8") as f:
            f.write(mhtml_data)

        browser.close()
        print(f"Saved MHTML to {mhtml_file}")


def extract_preceding_chars(mhtml_file, keyword):
    """Print the n characters preceding each occurrence of keyword in MHTML text."""
    with open(mhtml_file, "r", encoding="utf-8") as f:
        text = f.read()

    index = 0
    results = []
    while True:
        index = text.find(keyword, index)
        if index == -1:
            break
        # get up to n preceding chars (if available)
        preceding = text[max(0, index - 4):index]
        if preceding[0] != "1": 
            preceding = preceding[1:]
        results.append(preceding)
        index += len(keyword)
    print(results)
    return results

def log_to_csv(csv_file, keyword, preceding_list):
    """Append timestamp + preceding chars to CSV."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create file with headers if it doesn't exist yet
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Time", "RSF_percent_full", "CMS_percent_full"])  # headers
        writer.writerow([now, preceding_list[0], preceding_list[1]])


if __name__ == "__main__":
    save_page_as_mhtml(URL, MHTML_FILE)

    keyword = " Full"  # change this
    preceding_chars = extract_preceding_chars(MHTML_FILE, keyword)

    if preceding_chars:
        log_to_csv(CSV_FILE, keyword, preceding_chars)
        print(f"Logged {len(preceding_chars)} matches to {CSV_FILE}")
    else:
        print("No matches found.")
