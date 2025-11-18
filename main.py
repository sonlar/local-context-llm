import keyword_spacy
import os
import pymupdf
import spacy
class Build_Corpus:
    def find_files(self, path: str) -> list:
        return os.listdir(path)

    def read_pdf(self, document: str) -> str:
        reader = pymupdf.open(document)
        doc = str()
        for page in reader:
            doc += str(page.get_text())
        return doc
    def extract_keywords(self, text: str ) -> list:
        nlp = spacy.load("en_core_web_md")
        nlp.add_pipe("keyword_extractor", last=True, config={"top_n": 10, "min_ngram": 3, "max_ngram": 3, "strict": True})
        doc = nlp(text)
        return [word[0] for word in doc._.keywords]
