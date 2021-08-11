#!/usr/bin/env python
# coding: utf-8

# In[109]:


import scispacy
import spacy
import pandas as pd
from pathlib import Path

nlp = spacy.load("en_core_sci_scibert")


def process_file(file):
    with file.open("r") as f:
        title = f.readline()
        abstract = f.readline()

    nlp_title = nlp(title.strip())
    nlp_abstract = nlp(abstract.strip())
    sents = list(nlp_title.sents) + list(nlp_abstract.sents)

    data = []
    sent_offset = 0
    for sent_id, sent in enumerate(sents):
        for token_id, token in enumerate(sent):
            data.append(
                (
                    sent_id + 1,
                    token_id + 1,
                    sent_offset + token.idx,
                    sent_offset + token.idx + len(token),
                    token,
                )
            )

        sent_offset += len(sent.text) + 1  # Do I really need + 1 for SANTO parser?

    df = pd.DataFrame(
        data=data, columns=["Sent ID", "Token ID", "Char start", "Char end", "Text"]
    ).astype(str)

    with open(f"pilot_data/{file.stem}.csv", "w") as f:
        for i, sent in enumerate(sents):
            f.write("#" + sent.text + "\n")

            mask = df["Sent ID"] == str(i + 1)
            for values in df[mask].values.tolist():
                f.write("\t".join(values) + "\n")


if __name__ == "__main__":
    files = Path("pilot_brat_data").glob("*.txt")
    for file in files:
        process_file(file)
