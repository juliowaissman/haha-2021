"""Convert textcat annotation from JSONL to spaCy v3 .spacy format."""
import pandas as pd
import typer
import warnings
from pathlib import Path

import spacy
from spacy.tokens import DocBin


def convert(lang: str, input_path: Path, training_path: Path, validation_path: Path):
    nlp = spacy.blank(lang)
    db_train = DocBin()
    db_test = DocBin()
    
    df = pd.read_csv(input_path)
    df_si = df[df.is_humor > 0]
    train_si = df_si.sample(frac=0.8, random_state=31416) 
    test_si = df_si.drop(train_si.index)

    df_no = df[df.is_humor == 0]    
    train_no = df_no.sample(frac=0.8, random_state=31416) 
    test_no = df_no.drop(train_no.index)

    db_train = genera(nlp, train_si.text, {'humor': 1.0, 'no_humor':0.0}, db_train)
    db_train = genera(nlp, train_no.text, {'humor': 0.0, 'no_humor':1.0}, db_train)
    db_train.to_disk(training_path)
    
    db_test = genera(nlp, test_si.text, {'humor': 1.0, 'no_humor':0.0}, db_test)
    db_test = genera(nlp, test_no.text, {'humor': 0.0, 'no_humor':1.0}, db_test)
    db_test.to_disk(validation_path)


def genera(nlp, serie, cat, db):
    for text in serie:
        doc = nlp.make_doc(text)
        doc.cats = cat
        db.add(doc)
    return db
    
    
if __name__ == "__main__":
    typer.run(convert)
