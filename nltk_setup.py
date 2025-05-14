import nltk

PACKAGES = [
    ('tokenizers/punkt', 'punkt'),
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet', 'wordnet'), 
    ('sentiment/vader_lexicon', 'vader_lexicon')
]

def download_nltk_resources():
    print("Checking and downloading NLTK resources...")
    all_downloaded = True
    for path, package_id in PACKAGES:
        try:
            nltk.data.find(path)
            print(f"[nltk_setup] {package_id} is already downloaded.")
        except nltk.downloader.DownloadError:
            print(f"[nltk_setup] Downloading {package_id}...")
            try:
                nltk.download(package_id, quiet=False) # Set quiet=True for less verbose output
                print(f"[nltk_setup] Successfully downloaded {package_id}.")
            except Exception as e:
                print(f"[nltk_setup] ERROR: Could not download {package_id}. Reason: {e}")
                all_downloaded = False
        except Exception as e: # Handle other potential errors like network issues during find
            print(f"[nltk_setup] ERROR: An issue occurred while checking {package_id}. Reason: {e}")
            all_downloaded = False


    if all_downloaded:
        print("All NLTK resources are available.")
    else:
        print("Some NLTK resources might be missing. Please check the logs.")

if __name__ == '__main__':
    # This allows you to run `python nltk_setup.py` directly
    download_nltk_resources()