import pronouncing
from poesy import Poem
from rhyme_highlighter import highlight_rhymes  # from Rhyme-Highlighter tool

def get_syllable_count(word):
    phones = pronouncing.phones_for_word(word)
    return pronouncing.syllable_count(phones[0]) if phones else 0

def rhymes(word1, word2):
    return word2 in pronouncing.rhymes(word1)

def analyze_meter_and_rhyme_block(lines):
    poem = Poem("\n".join(lines))
    meter = poem.meter()
    rhyme_scheme = poem.rhyme_scheme()
    return meter, rhyme_scheme

def highlight_full_rhymes(text):
    return highlight_rhymes(text)


