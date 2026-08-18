"""Script conversion between Neo-Tifinagh and Kabyle Latin, and schwa restoration.

Tifinagh as it is written for Kabyle omits `e`, so the mapping back to Latin is not a
lookup: the vowel has to be predicted from the consonants around it. That is why this is
a model and not a table, and why the table is kept beside it as the baseline to beat
(`agbalu.bench.tifinagh`).

Nothing is imported here. `torch` is an optional extra, and `config` and `tokenizer` are
useful without it.
"""
