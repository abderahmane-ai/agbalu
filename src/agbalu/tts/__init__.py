"""Speech synthesis: the front-end and the voice corpus for Matoub.

`g2p` is the reference grapheme-to-phoneme table. It exists because the published
pronunciation lexicon is a dictionary — it covers 93.75% of word tokens but leaves
26.92% of speech-corpus clips carrying an unattested word, and an acoustic model
needs phonemes for every utterance it trains on.
"""
