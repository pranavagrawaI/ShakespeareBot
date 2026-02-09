import re
from pypdf import PdfReader

def preprocess_shakespeare_pdf(input_path, output_path):
    print(f"Processing: {input_path}")
    
    # 1. Load the PDF
    try:
        reader = PdfReader(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    full_text = ""
    
    # 2. Extract text from each page
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    # 3. Define Regex Patterns for Noise Removal
    # Matches timestamps like "07/02/2026, 14:28"
    timestamp_pattern = r"\d{2}/\d{2}/\d{4}, \d{2}:\d{2}"
    
    # Matches the specific header "Hamlet (complete text) (OpenSourceShakespeare.org)"
    header_pattern = r"Hamlet \(complete text\) \(OpenSourceShakespeare\.org\)"
    
    # Matches URLs starting with https://www.opensourceshakespeare.org
    url_pattern = r"https?://www\.opensourceshakespeare\.org\S+"
    
    # Matches page indicators like "1/84", "2/84", "84/84"
    page_number_pattern = r"\b\d+/84\b"
    
    # Matches isolated line numbers (e.g., "5", "10", "4060") on their own lines
    line_number_pattern = r"^\s*\d+\s*$"

    # 4. Apply Cleaning
    lines = full_text.split('\n')
    cleaned_lines = []

    for line in lines:
        # Apply regex substitutions
        line = re.sub(timestamp_pattern, "", line)
        line = re.sub(header_pattern, "", line)
        line = re.sub(url_pattern, "", line)
        line = re.sub(page_number_pattern, "", line)
        
        # Remove lines that are just line numbers (e.g., " 5 ")
        if re.match(line_number_pattern, line):
            continue
            
        # Remove "The following table:" artifacts if present from extraction
        if "The following table:" in line:
            continue

        # Strip whitespace and only keep non-empty lines
        clean_line = line.strip()
        if clean_line:
            cleaned_lines.append(clean_line)

    # 5. Save to Output
    final_text = "\n".join(cleaned_lines)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)
        
    print(f"Success! Cleaned text saved to: {output_path}")

# --- Execution ---
if __name__ == "__main__":
    input_pdf = "/Users/vipulsharma/shakespearebot/ShakespeareBot/dataset.pdf"
    output_txt = "/Users/vipulsharma/shakespearebot/ShakespeareBot/cleaned_hamlet.txt"
    
    preprocess_shakespeare_pdf(input_pdf, output_txt)