import os
from typing import Literal

import chromadb
import keyword_spacy
import pymupdf
import spacy
import wikipedia
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class Build_Corpus:
    def __find_files(self, path: str) -> list[str]:
        """Returns a list of files in the given path"""
        return os.listdir(path)

    def __read_pdf(self, document: str) -> list[str]:
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

        except Exception as e:
            print(e)
            return list()

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


class Wiki:
    """Searches Wikipedia for keywords extracted from question"""

    # WARNING: Requires: python -m spacy download en_core_web_md
    def __extract_keywords(self, question: str, count: int) -> list:
        """Uses Keyword spaCy to extract keywords from question"""
        try:
            nlp = spacy.load("en_core_web_md")
            nlp.add_pipe(
                "keyword_extractor",
                last=True,
                config={"top_n": count, "min_ngram": 1, "max_ngram": 3, "strict": True},
            )
            doc = nlp(question)
            return doc._.keywords

        except Exception as e:
            print(e)
            return list()

    def search(
        self, question: str, count: int = 3, sentences: int = 5, article_count: int = 2
    ) -> list:
        """Searches wikipedia for keywords"""
        context = list()
        if not question.strip():
            print("No question found")
            return list()

        keywords = self.__extract_keywords(question, count)
        for keyword in keywords:
            searchterm = keyword[0]
            articles = wikipedia.search(searchterm)[:article_count]

            for article in articles:
                summary = wikipedia.summary(
                    article, sentences=sentences, auto_suggest=False
                )
                context.append(summary)

        return context


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

    def prompt(
        self,
        question: str,
        source: Literal["database", "wikipedia", "no_context"] = "database",
    ) -> None:
        """
        Expects question: str which is prompted to LLM
        Calls on get_context which is appended to prompt
        """
        template = f"Question: {question}\n\n"

        if source == "no_context":
            prompt = template

        else:
            context = self.__get_context(question, source)
            prompt = template + context

        answer = self.pipe(prompt)
        print(answer[0]["generated_text"])

    def __get_context(
        self, question: str, source: Literal["database", "wikipedia"]
    ) -> str:
        """Query given source to retrieve additional context for prompt"""
        match source:
            case "database":
                retrieved_docs = self.query(question)
            case "wikipedia":
                retrieved_docs = self.__wikipedia(question)

        docs_content = "\n\n".join(doc for doc in retrieved_docs)
        system_prompt = (
            "Use the given context to answer the original given question.\n"
            "Do NOT generate new questions.\n"
            "Use three sentences maximum and keep the answer concise.\n"
            f"Context: {docs_content}\n\n"
            "Answer: "
        )
        return system_prompt

    def __wikipedia(self, question: str) -> list:
        """Searches Wikipedia for keywords found in question"""
        wiki = Wiki()
        return wiki.search(question)


if __name__ == "__main__":
    db = Database()
    corpus = db.collect_data()
    db.connect_to_db(persistent=False)
    db.write_to_db(corpus)
    db.read_db()
    # db.query("ER diagram", n_results=10)

    # wiki = Wiki()
    # wiki.search(question="What is NoSQL?")

    llm = LLM("Qwen/Qwen3-0.6B", db=db)
    print("\nThe following question is answered using database\n")
    llm.prompt("What is NoSQL?", "database")

    print("\nThe following question is answered using wikipedia\n")
    llm.prompt("What is NoSQL?", "wikipedia")

    print("\nThe following question is answered using no context\n")
    llm.prompt("What is NoSQL?", "no_context")
