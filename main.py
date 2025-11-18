import keyword_spacy
import os
import pymupdf
import spacy


class Build_Corpus:
    def __find_files(self, path: str) -> list:
        return os.listdir(path)

    def __read_pdf(self, document: str) -> str:
        reader = pymupdf.open(document)
        doc = str()
        for page in reader:
            doc += str(page.get_text())
        return doc

    def __extract_keywords(self, text: str ) -> list:
        nlp = spacy.load("en_core_web_md")
        nlp.add_pipe("keyword_extractor", last=True, config={"top_n": 5, "min_ngram": 1, "max_ngram": 3, "strict": True})
        doc = nlp(text)
        return [word[0] for word in doc._.keywords]

    def extract(self, path: str) -> None:
        files = self.__find_files(path)
        for file in files:
            text = self.__read_pdf(path+file)
            keywords = self.__extract_keywords(text)
            print(f"{file}: {keywords}\n")


# TODO: Implement chroma db
# TODO: Find appropriate model on huggingface for RAG based on corpus extracted from pdfs

if __name__ == "__main__":
    corpo = Build_Corpus()
    corpo_extract = corpo.extract(path="./data/")

    

