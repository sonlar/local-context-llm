import pymupdf
class Build_Corpus:
    def __init__(self):
        pass
    def read_pdf(document: str) -> str:
        reader = pymupdf.open(document)
        doc = str()
        for page in reader:
            doc += str(page.get_text())
        return doc
