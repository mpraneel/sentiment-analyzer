"""
nltk_setup.py  —  Download required NLTK data packages.

Called automatically on app startup (from app.py) and can also be run
directly:  python nltk_setup.py
"""

import nltk


# (data_path, package_id) pairs.
# punkt_tab is required by NLTK >= 3.9; punkt is kept as a fallback for
# older installations.
_PACKAGES = [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("tokenizers/punkt",     "punkt"),
    ("corpora/stopwords",    "stopwords"),
    ("corpora/wordnet",      "wordnet"),
    ("sentiment/vader_lexicon", "vader_lexicon"),
]


def download_nltk_resources() -> bool:
    """
    Ensure all required NLTK data packages are present.

    Returns True when every package is available, False if any failed.
    Downloads are skipped silently when a package is already on disk.
    """
    all_ok = True
    for data_path, package_id in _PACKAGES:
        try:
            nltk.data.find(data_path)
        except LookupError:
            try:
                nltk.download(package_id, quiet=True)
            except Exception as exc:
                print(f"[nltk_setup] WARNING: could not download '{package_id}': {exc}")
                all_ok = False
        except Exception as exc:
            print(f"[nltk_setup] WARNING: error checking '{package_id}': {exc}")
            all_ok = False

    return all_ok


if __name__ == "__main__":
    ok = download_nltk_resources()
    if ok:
        print("All NLTK resources are ready.")
    else:
        print("Some NLTK resources could not be downloaded. Check the warnings above.")
