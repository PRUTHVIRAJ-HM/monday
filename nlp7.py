import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
def get_synonyms_antonyms(word):
    synonyms = set()
    antonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name())
    return synonyms, antonyms

synonyms, antonyms = get_synonyms_antonyms("active")
print("Synonyms of 'active':", synonyms)
print("Antonyms of 'active':", antonyms)