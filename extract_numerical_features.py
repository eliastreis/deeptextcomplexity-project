import nltk
nltk.download('punkt')
from textblob_de import TextBlobDE as TextBlob

import textstat

def extract_numerical_features(sentence):
    features = []

    blob = TextBlob(sentence)
    sentence_length = len(blob)


    # sentence length
    features.append(len(sentence.split(" ")))
    features.append(sentence_length)

    # (avg) number of syllables per sentence
    features.append(textstat.syllable_count(sentence))
    features.append(textstat.avg_syllables_per_word(sentence))

    # Number of words with more then 3 syllables
    def more_then_3_syllables(sentence):
      count = 0
      for word in sentence.split(" "):
          syllable_count = textstat.syllable_count(word)
          if syllable_count >= 3: count += 1
      return count
    features.append(more_then_3_syllables(sentence))

    # all tags; see https://github.com/markuskiller/textblob-de/blob/dev/textblob_de/ext/_pattern/text/search.py
    TAGS = ["CC", "CD", "CJ", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "JJ*",
            "LS", "MD", "NN", "NNS", "NNP", "NNPS", "NN*", "NO", "PDT", "PR",
            "PRP", "PRP$", "PR*", "PRP*", "PT", "RB", "RBR", "RBS", "RB*", "RP",
            "SYM", "TO", "UH", "VB", "VBZ", "VBP", "VBD", "VBN", "VBG", "VB*",
            "WDT", "WP*", "WRB", "X", ".", ",", ":", "(", ")"]

    # todo: can be optimized to be map
    # get all tags in sentence
    tags_in_sentence = [e[1] for e in blob.pos_tags]
    for e in blob.tokens:
        if e in [".", ",", ":", "(", ")"]:
            tags_in_sentence.append(e)

    for tag in TAGS:
        appearances = tags_in_sentence.count(tag)
        features.append(appearances)
        features.append(appearances/sentence_length)

    return np.array(features)
