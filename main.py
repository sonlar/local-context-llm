import pymupdf
class Build_Corpus:
    def find_files(self, path: str) -> list:
        return os.listdir(path)

    def read_pdf(self, document: str) -> str:
        reader = pymupdf.open(document)
        doc = str()
        for page in reader:
            doc += str(page.get_text())
        return doc
