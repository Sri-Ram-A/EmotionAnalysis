# Text processing
import re
import emoji
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pathlib import Path
import nltk
nltk.download('punkt_tab')
BASE_DIR =  Path(__file__).resolve().parents[2]

#  1
def remove_html_tags(text):
    """Remove HTML tags from text"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# 2 
def clean_text(text):
    """REMOVE UNECESSARY SPACES AND REPEATED CHARS"""
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)  # "<p>hi</p>" → "hi"
    # remove space BEFORE special chars
    text = re.sub(r'\s+([@#&!?.,:/\-_*=(){}\[\]\'"…])', r'\1', text)
    # remove space AFTER special chars
    text = re.sub(r'([@#&!?.,:/\-_*=(){}\[\]\'"…])\s+', r'\1', text)
    # normalize spaced slashes in URLs: "buff.ly / abc" → "buff.ly/abc"
    text = re.sub(r'(\w)\s*\/\s*(\w)', r'\1/\2', text)
    # remove URLs fully
    text = re.sub(r'https?://\S+|www\.\S+|\w+\.\w+/\S+', '', text)
    # remove mentions: @user, @ user, @abc_xyz, messy @ mompou _ mumpow
    text = re.sub(r'@\s*[\w_]+', '', text)
    text = re.sub(r'@\s*[^#\s]+', '', text)
    # remove hashtag symbol but keep text
    text = re.sub(r'#(\w+)', r'\1', text)
    # collapse spaced ellipses: ". . ." → "..."
    text = re.sub(r'\.\s+\.\s+\.', '...', text)
    # collapse repeated spaced punctuations: "? ? ?" → "???"
    text = re.sub(r'([!?])\s+(?=\1)', r'\1', text)
    # NEW: collapse repeated special characters → keep ONE
    text = re.sub(r'([@#&!?.,:/\-_*=(){}\[\]\'"…])\1+', r'\1', text)
    return text

# 3 
def convert_emojis_to_text(text):
    """CONVERT EMOJIS TO TEXT"""
    """Convert emojis to their text description"""
    return emoji.demojize(text, delimiters=(" ", " "))

# 4
def filter_english_ascii(paragraph: str) -> str:
    """ KEEP ONLY ENGLISH LANGUAGE"""
    tokens = re.findall(r"[A-Za-z']+", paragraph)
    return " ".join(tokens)

# 5 
def replace_punctuation_with_space(text):
    """REPLACE PUNCTUATION WITH SPACE"""
    """Replace punctuation characters with a space instead of removing them."""
    return re.sub(r'[^\w\s]', ' ', text)  # punctuation → " "

# 6 
def remove_extra_whitespace(text):
    """REMOVE EXTRA WHITESPACE"""
    """Remove extra whitespace"""
    return ' '.join(text.split())

# 7 
def convert_to_lowercase(text):
    """LOWERCASE CONVERSION"""
    """Convert text to lowercase"""
    return text.lower()

# 8 
def expand_contractions(text):
    """Expand Contacted Words"""
    """Expand contractions like don't -> do not"""
    return contractions.fix(text)

def load_slang_dict(file_path):
    slang_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):   # skip empty/comment 
                continue
            parts = line.split(maxsplit=1)         # split into 2 parts ONLY
            if len(parts) == 2:
                slang, expansion = parts[0], parts[1]
                slang_map[slang.lower()] = expansion.lower()
    return slang_map

# 9 
# This shit takes time,god knows why -7min
slang_dict = load_slang_dict(BASE_DIR / "src" / "data" / "slangs.txt")
def expand_slang(text):
    """CONVERT SHORT TO LONG FORM"""
    text_lower = text.lower()
    for slang, expansion in slang_dict.items():
        # Escape special regex characters like /, -, *, etc.
        slang_escaped = re.escape(slang)
        # Replace whole-word slang with expansion
        pattern = r'\b' + slang_escaped + r'\b'
        text_lower = re.sub(pattern, expansion, text_lower)
    return text_lower

#  10 
def clean_single_letters_and_repeats(text):
    """Removing single and repeated letters"""
    # 1. Normalize repeated characters inside words: "coooool" -> "cool"
    text = re.sub(r'(\w)\1+', r'\1', text)
    # 2. Remove isolated single letters EXCEPT meaningful ones
    keep_letters = {'a', 'i', 'o', 'u'}  # customize as needed
    # Regex catches isolated letters between boundaries
    def remove_bad_single_letters(match):
        letter = match.group(1)
        return letter if letter in keep_letters else ''
    text = re.sub(r'\b([a-zA-Z])\b', remove_bad_single_letters, text)
    return text

# 11 
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    """STOPWORDS REMOVAL"""
    list_of_words =  word_tokenize(text)
    filtered_tokens = [word for word in list_of_words if word not in stop_words]
    text = " ".join(filtered_tokens)
    return text
