import os
import re
import pandas as pd
import fitz  # PyMuPDF (FAST)
import unicodedata
from multiprocessing import Pool, cpu_count
from datetime import datetime

senate_header_pattern = r"(?:\n|\r|\f)\s*Senate\s*(?:\n|\r)"
house_header_pattern  = r"(?:\n|\r|\f)\s*House of Representatives\s*(?:\n|\r)"

def extract_text(pdf_path):
    """Extract text using PyMuPDF — MUCH faster & more accurate than PDFMiner."""
    text = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text.append(page.get_text("text"))
        return "\n".join(text)
    except Exception as e:
        print(f"   ❌ Error extracting text from {pdf_path}: {e}")
        return ""

def normalize_text(txt):
    if not txt:
        return txt
    
    txt = txt.encode("latin1", errors="ignore").decode("utf8", errors="ignore")

    # Fix typical UTF-8 → Latin1 decode bugs
    replacements = {
        "â€™": "'",    # apostrophe
        "â€œ": "\"",   # opening quote
        "â€�": "\"",   # closing quote
        "â€˜": "'",    # opening apostrophe
        "â€”": "-",    # em dash
        "â€“": "-",    # en dash
        "â€¢": "*",    # bullet
        "â€¦": "...",  # ellipsis
        "Â": "",       # stray control character
        "â€": "",      # garbage prefix
    }

    for bad, good in replacements.items():
        txt = txt.replace(bad, good)

    return txt

def clean(text):
    text = text.replace("\f", " ")
    text = re.sub(r"\b[HSE]\s?\d{1,4}\b", " ", text)
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"(Vol\.\s*\d+.*?\n)", "", text)
    text = re.sub(r"(PO\s*\d+.*?\n)", "", text)
    text = re.sub(r"(VerDate.*?\n)", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def split_into_paragraphs(text):
    paras = re.split(
        r"(?:\n\s*\n)|(?<=[\.!?])\s+(?=[A-Z])",
        text
    )
    return [p.strip() for p in paras if p.strip()]

def merge_to_target_length(paragraphs, target=180, max_len=260):
    merged = []
    buffer = ""

    for p in paragraphs:
        if len((buffer + " " + p).split()) < target:
            buffer = (buffer + " " + p).strip()
        else:
            if buffer:
                merged.append(buffer.strip())
            buffer = p

    if buffer:
        merged.append(buffer.strip())

    merged = [p for p in merged if len(p.split()) <= max_len]
    return merged

def extract_year_from_path(pdf_path):
    filename = os.path.basename(pdf_path)
    m = re.search(r"CREC_(\d{4})_", filename)
    if m:
        return m.group(1)
    return os.path.basename(os.path.dirname(pdf_path))

def process_pdf(pdf_path, output_root):
    try:
        filename = os.path.basename(pdf_path)
        year = extract_year_from_path(pdf_path)

        print(f"\n📄 Processing {year}/{filename}")

        raw_text = extract_text(pdf_path)
        if not raw_text:
            print("   ⚠️ Empty text extracted, skipping.")
            return

        raw_text = normalize_text(raw_text)

        senate_match = re.search(senate_header_pattern, raw_text)
        house_match  = re.search(house_header_pattern, raw_text)

        if not senate_match or not house_match:
            print("   ⚠️ No Senate/House header. Skipping.")
            return

        if senate_match.start() < house_match.start():
            senate_text = raw_text[senate_match.end():house_match.start()]
            house_text  = raw_text[house_match.end():]
        else:
            house_text  = raw_text[house_match.end():senate_match.start()]
            senate_text = raw_text[senate_match.end():]

        clean_senate = clean(normalize_text(senate_text))
        clean_house  = clean(normalize_text(house_text))

        senate_paras = split_into_paragraphs(clean_senate)
        house_paras  = split_into_paragraphs(clean_house)

        # ======================
        # MERGE
        # ======================
        senate_final = merge_to_target_length(senate_paras)
        house_final  = merge_to_target_length(house_paras)

        if not senate_final and not house_final:
            print("   ⚠️ No final paragraphs. Skipping.")
            return

        df = pd.DataFrame({
            "chamber": ["Senate"]*len(senate_final) + ["House"]*len(house_final),
            "paragraph": senate_final + house_final
        })

        year_folder = os.path.join(output_root, year)
        os.makedirs(year_folder, exist_ok=True)

        out_name = os.path.splitext(filename)[0] + "_clean_paragraphs.csv"
        out_path = os.path.join(year_folder, out_name)

        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"   ✅ Saved → {out_path}")

    except Exception as e:
        print(f"   ❌ Fatal error processing {pdf_path}: {e}")
        return

if __name__ == "__main__":

    input_root  = r"D:/RA with Steven Silver/CongressionalRecords_EntireIssue/1997"
    output_root = r"D:/RA with Steven Silver/CongressionalRecords_EntireIssue_cleaned"

    pdf_files = []

    for root, dirs, files in os.walk(input_root):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, fname))

    print(f"\n🔥 FOUND {len(pdf_files)} PDF FILES TO PROCESS\n")

    workers = max(2, cpu_count() - 1)   # use N-1 cores
    print(f"🚀 Using {workers} parallel workers\n")

    with Pool(processes=workers) as pool:
        pool.starmap(process_pdf, [(pdf, output_root) for pdf in pdf_files])

    print("\n🎉 DONE — All PDFs Processed!")