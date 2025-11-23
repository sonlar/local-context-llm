import os

import chromadb
import pymupdf
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class Build_Corpus:
    def __find_files(self, path: str) -> list[str]:
        """Returns a list of files in the given path"""

        return os.listdir(path)

    def __read_pdf(self, document: str) -> list[str] | None:
        """
        Attempts to read a given path/to/document,
        and returns the content of the given doc to the best of its abilities
        """
        try:
            reader = pymupdf.open(document)
            doc = list()
            for page in reader:
                doc.append(page.get_text())
            return doc
        except Exception:
            print(Exception)

    def extract(self, path: str) -> list[tuple[str, str, str]]:
        """
        Reads files in a given path
        Returns a list of tuples:
            (filename-page_number: str, text: str, filename: str)
        """
        corpus = list()
        files = self.__find_files(path)
        for file in files:
            filename, _ = os.path.splitext(file)
            text = self.__read_pdf(path + file)
            for page_number, page in enumerate(text):
                if len(page.strip()) == 0:
                    continue
                corpus.append((f"{filename}-{page_number}", page, filename))
        return corpus


class Database:
    def collect_data(self, path: str = "./data/") -> list[tuple[str, str, str]]:
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

    def write_to_db(
        self, corpus: list[tuple[str, str, str]], name: str = "my_collection"
    ) -> None:
        """
        Writes to collection, and inserts corpus
        Does not overwrite existing collection
        Parameters:
            corpus: list
                filename-page_number: str used as id in the collection
                text: str content used to create embeddings
                filename: list can be used to filter queries
            name: str name of collection we are creating/writing to
        """
        file, text, filename = zip(*corpus)
        self.collection = self.client.get_or_create_collection(name=name)
        self.collection.upsert(
            ids=list(file),
            documents=list(text),
            metadatas=[{"filename": str(name)} for name in filename],
        )

    def read_db(self, name: str = "my_collection") -> None:
        """Reads content of persistent collection"""
        self.collection = self.client.get_collection(name=name)

    def query(self, query: str, n_results: int = 5) -> list:
        """Returns the most relevant content in a list"""
        results = self.collection.query(query_texts=list(query), n_results=n_results)
        ids_res = results["ids"][0]
        dists = results["distances"][0]
        docs_res = results["documents"][0]
        metas = results["metadatas"][0]
        return docs_res


class LLM:
    def __init__(self, model_id: str, db: Database) -> None:
        """
        Installs local LLM and builds pipeline
        Expects two parameters:
            model_id: str -> model name from huggingface
            db: Database-> Database class to enable prompting Chroma
        """
        self.query = db.query
        model_id = model_id
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        self.pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200
        )

    def prompt(self, question: str) -> None:
        """
        Expects question: str which is prompted to LLM
        Calls on get_context which is appended to prompt
        """
        template = f"Question: {question}\n\n"
        context = self.get_context(question)
        prompt = template + context
        answer = self.pipe(prompt)
        print(answer[0]["generated_text"])

    def get_context(self, question: str) -> str:
        """use self.query from db.query to retrieve additional context for prompt"""
        retrieved_docs = self.query(question)
        docs_content = "\n\n".join(doc for doc in retrieved_docs)
        system_prompt = (
            "Use the given context to answer the original given question.\n"
            "Do NOT generate new questions."
            "Use three sentences maximum and keep the answer concise.\n"
            f"Context: {docs_content}"
        )
        return system_prompt


if __name__ == "__main__":
    db = Database()
    # corpus = db.collect_data()
    db.connect_to_db(persistent=True)
    # db.write_to_db(corpus)
    db.read_db()
    # db.query("ER diagram", n_results=10)

    llm = LLM("Qwen/Qwen3-0.6B", db=db)
    llm.prompt("What is nosql?")
