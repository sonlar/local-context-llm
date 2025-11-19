import chromadb
import keyword_spacy
import os
import pymupdf
import spacy


class Build_Corpus:
    def __find_files(self, path: str) -> list:
        """Returns a list of files in the given path"""
        return os.listdir(path)

    def __read_pdf(self, document: str) -> str:
        """
        Attempts to read a given path/to/document,
        and returns the content of the given doc to the best of its abilities
        """
        try:
            reader = pymupdf.open(document)
            doc = str()
            for page in reader:
                doc += str(page.get_text())
            return doc
        except Exception:
            print(Exception)
            return ""

    def __extract_keywords(self, text: str ) -> list:
        """Expects a text: str and returns a list of keywords that best describes it"""
        nlp = spacy.load("en_core_web_md")
        nlp.add_pipe("keyword_extractor", last=True, config={"top_n": 1, "min_ngram": 1, "max_ngram": 3, "strict": True})
        doc = nlp(text)
        return [word[0] for word in doc._.keywords]

    def extract(self, path: str) -> list:
        """
        Reads files in a given path
        Returns a list of tuples: (file(name): str, text: str, keywords: list)
        """
        corpus = list()
        files = self.__find_files(path)
        for file in files:
            text = self.__read_pdf(path+file)
            keywords = self.__extract_keywords(text)
            corpus.append((file, text, keywords))
        return corpus


# TODO: Implement chroma db
class Database:
    def collect_data(self, path="./data/") -> list:
        """Extracts corpus from files to database"""
        extractor = Build_Corpus()
        corpus = extractor.extract(path=path)
        return corpus

    def connect_to_db(self, persistent: bool = False, path: str = "./chroma") -> None:
        """
        Attempts to create or connect to database.
        Has two parameters: persistent: bool, path: str
        persistent determines if the db is only in memory(False) or saved to disk
        path only matters if persistent
        """
        if persistent:
            self.client = chromadb.PersistentClient(path=path)
        else:
            self.client = chromadb.Client()

    def write_to_db(self, corpus: list, name: str = "my_collection") -> None: 
        """
        Writes to collection, and inserts corpus
        Does not overwrite existing collection
        Parameters: 
            corpus: list
                Expects a list of tuples containing three values in the following order:
                filename: str used as id in the collection
                text: str content used to create embeddings
                keywords: list can be used to filter queries
            name: str name of collection we are creating/writing to
        """
        file, text, keywords = zip(*corpus)
        self.collection = self.client.get_or_create_collection(name=name)
        self.collection.upsert(
            ids = list(file),
            documents=list(text),
            metadatas = [{"keyword": str(keyword)} for keyword in keywords]
        )

    def read_db(self, name: str = "my_collection") -> None:
        """Reads content of persistent collection"""
        self.collection = self.client.get_collection(name=name)
        

    def query(self, query: str, n_results: int = 1) -> None:
        """Returns the most relevant content"""
        results = self.collection.query(query_texts=list(query), n_results=n_results)
        ids_res = results["ids"][0]
        dists = results["distances"][0]
        docs_res = results["documents"][0]
        metas = results["metadatas"][0]
        for doc_id, dist, doc, meta in zip(ids_res, dists, docs_res, metas):
            print(f"id:{doc_id}, dist:{dist}, metadata: {meta}, doc: {doc[:100]}") 



# TODO: Find appropriate model on huggingface for RAG based on corpus extracted from pdfs

if __name__ == "__main__":
    db = Database()
    # corpus = db.collect_data()
    # db.connect_to_db(persistent=True)
    # db.write_to_db(corpus)
    # db.read_db()
    # db.query("I enjoy ERDiagrams, they are fun to make, especially when they use the union operator")


