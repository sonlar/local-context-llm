import chromadb
import os
import langchain_chroma
import pymupdf

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate

class Build_Corpus:
    def __find_files(self, path: str) -> list:
        """Returns a list of files in the given path"""
        return os.listdir(path)

    def __read_pdf(self, document: str) -> list:
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
            return list()

    def extract(self, path: str) -> list:
        """
        Reads files in a given path
        Returns a list of tuples: (filename-page_number: str, text: str, filename: str)
        """
        corpus = list()
        files = self.__find_files(path)
        for file in files:
            filename, _ = os.path.splitext(file)
            text = self.__read_pdf(path+file)
            for page_number, page in enumerate(text):
                corpus.append((f"{filename}-{page_number}", page, filename))
        return corpus


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
                filename-page_number: str used as id in the collection
                text: str content used to create embeddings
                filename: list can be used to filter queries
            name: str name of collection we are creating/writing to
        """
        file, text, filename = zip(*corpus)
        self.collection = self.client.get_or_create_collection(name=name)
        self.collection.upsert(
            ids = list(file),
            documents=list(text),
            metadatas = [{"filename": str(name)} for name in filename]
        )

    def read_db(self, name: str = "my_collection") -> None:
        """Reads content of persistent collection"""
        self.collection = self.client.get_collection(name=name)
        
    def create_vectorstore(self, name: str = "my_collection", path: str = "./chroma") -> Chroma:
        """Create vectorstore for  LLM"""
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(
            collection_name=name,
            persist_directory=path,
            embedding_function=embeddings
        )
        return vectorstore


class LLM:
        def __init__(self, model_id) -> None:
            model_id = model_id
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
            self.hf = HuggingFacePipeline(pipeline=pipe)

        def prompt(self, question) -> None:
            template = "Question: {question}"
            prompt = PromptTemplate.from_template(template)
            chain = prompt | self.hf
            question = question
            print(chain.invoke({"question": question}))


if __name__ == "__main__":
    db = Database()
    # corpus = db.collect_data()
    # db.connect_to_db(persistent=True)
    # db.write_to_db(corpus)
    # db.read_db()
    # db.query("nosql, sql, chroma, chromadb, vectors")

    # llm = LLM("Qwen/Qwen3-0.6B")
    # llm.prompt("What is nosql?")

    vectorstore = db.create_vectorstore()
    print("\n")
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "fetch_k": 5}
    )
    print(retriever.invoke("Tell me something about nosql"))

