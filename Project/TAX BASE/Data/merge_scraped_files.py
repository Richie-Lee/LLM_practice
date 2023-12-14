def read_file_with_fallback(file_path):
    try:
        # First attempt: Read with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        try:
            # Second attempt: Read with ISO-8859-1 encoding
            with open(file_path, 'r', encoding='ISO-8859-1') as file:
                return file.read()
        except UnicodeDecodeError:
            # Final attempt: Ignore errors with UTF-8 encoding
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()

def str_to_txt_file(export_str, export_directory):
    with open(export_directory, 'w', encoding='utf-8') as text_file:
        text_file.write(export_str)

EXPORT = True
export_directory = "C:/Users/RLee/Desktop/TAX BASE/bdo_scrape_full.txt"

# All member firms of which we have individually scraped the BDO insights page
mfs = ["us", "uk", "nl", "au"]

# Create dictionary with member firms as key and the scraped pages as value
all_docs = {}
global_index = 0
merged_string = ""

for mf in mfs:
    import_directory = f"C:/Users/RLee/Desktop/TAX BASE/bdo_{mf}_scrape.txt"
    
    # Read the contents of the file using the fallback mechanism
    all_docs[mf] = read_file_with_fallback(import_directory)

    # Split the document into individual entries
    entries = all_docs[mf].split('-\n')
    local_index = 0
    for entry in entries:
        # Skip empty entries
        if not entry.strip():
            continue

        # Replace the original index with the new format
        lines = entry.split('\n')
        lines[0] = f"Index: {global_index} ({mf.upper()} {local_index})"  # Reformat index line

        # Rejoin the entry and add it to the merged string
        reformatted_entry = '\n'.join(lines) + '\n-\n'
        merged_string += reformatted_entry

        # Increment global and local indices
        global_index += 1
        local_index += 1

if EXPORT:
    str_to_txt_file(merged_string, export_directory)
