"""Builds words.js (the game's word bank) from WordNet + a frequency list.
Usage:  pip install nltk wordfreq   then   python build_wordbank.py
"""
import json, random
from collections import Counter
import nltk
nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

def stem(w):
    return w[:max(4, len(w) - 3)]

rows = []
for w in wn.all_lemma_names():
    # single lowercase words only: no proper nouns, phrases, hyphens
    if not (w.isalpha() and w.islower() and 4 <= len(w) <= 14):
        continue
    z = zipf_frequency(w, "en")
    if z < 2.5:                      # too obscure to be fun
        continue
    if wn.morphy(w) not in (None, w):  # inflected form (cats, running)
        continue
    s = wn.synsets(w)[0]             # most common sense
    d = s.definition()
    dl = d.lower()
    if len(d) < 15 or dl.startswith(("plural of", "past tense")):
        continue
    # reject definitions that leak the answer or its synonyms
    if any(stem(l) in dl for l in s.lemma_names() if l.isalpha()):
        continue
    ex = next((e for e in s.examples() if w in e.lower()), "")
    tier = "easy" if z >= 4 else "medium" if z >= 3.3 else "hard"
    rows.append([w, d, s.pos(), ex, tier])

random.shuffle(rows)
with open("words.js", "w", encoding="utf-8") as f:
    f.write("window.WORDS=" + json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + ";")
print(len(rows), "words", dict(Counter(r[4] for r in rows)))
