# Local Context LLM

A Python-based Retrieval-Augmented Generation (RAG) system that enables you to query local documents using a locally-running large language model (LLM). This project extracts content from PDF files, stores them in a vector database, and uses semantic search to provide relevant context to an LLM for answering questions.

## Features

- **PDF Document Processing**: Automatically extract text from PDF files in a specified directory
- **Vector Database Storage**: Store document embeddings in ChromaDB for efficient semantic search
- **Local LLM Integration**: Use Hugging Face transformers to run LLMs locally (default: Qwen3-0.6B)
- **Retrieval-Augmented Generation**: Enhance LLM responses with relevant context from your documents
- **Persistent Storage**: Optional persistent database storage for reusing embeddings across sessions
- **MMR Search**: Uses Maximal Marginal Relevance for diverse and relevant document retrieval

## Prerequisites

- Python 3.8 or higher
- Sufficient RAM for running the LLM model (minimum 4GB recommended)
- Storage space for model weights and document embeddings

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sonlar/local-context-llm.git
cd local-context-llm
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. **Prepare your documents**: Place your PDF files in a `data/` directory in the project root:
```bash
mkdir data
# Copy your PDF files to the data/ directory
```

2. **Run the application**:
```bash
python main.py
```

### Step-by-Step Workflow

The application follows these steps:

1. **Extract documents** from the `data/` directory
2. **Create embeddings** using the `all-MiniLM-L6-v2` model
3. **Store embeddings** in ChromaDB
4. **Query the LLM** with context retrieved from your documents

### Example Code

```python
from main import Database, LLM

# Initialize database
db = Database()

# Option 1: Create database from PDF files
corpus = db.collect_data(path="./data/")
db.connect_to_db(persistent=True)
db.write_to_db(corpus)

# Option 2: Use existing database
db.connect_to_db(persistent=True)
db.read_db()

# Create vectorstore and LLM
vectorstore = db.create_vectorstore()
llm = LLM("Qwen/Qwen3-0.6B", vectorstore=vectorstore)

# Ask questions
llm.prompt("What is nosql?")
```

## Project Structure

```
local-context-llm/
├── main.py              # Main application with three classes:
│                        # - Build_Corpus: PDF extraction
│                        # - Database: ChromaDB management
│                        # - LLM: Query processing
├── requirements.txt     # Python dependencies
├── data/               # Directory for PDF documents (create this)
└── chroma/             # ChromaDB persistent storage (auto-created)
```

## Classes and Methods

### Build_Corpus
- `extract(path)`: Extracts text from all PDF files in the specified path
- Returns list of tuples: `(filename-page_number, text, filename)`

### Database
- `collect_data(path)`: Extracts corpus from PDF files
- `connect_to_db(persistent, path)`: Connects to ChromaDB (in-memory or persistent)
- `write_to_db(corpus, name)`: Writes corpus to vector database
- `read_db(name)`: Reads existing collection from database
- `create_vectorstore(name, path)`: Creates LangChain vectorstore for querying

### LLM
- `__init__(model_id, vectorstore)`: Initializes LLM with specified model and vectorstore
- `prompt(question)`: Processes a question and returns an answer with retrieved context
- `get_context(question)`: Retrieves relevant documents for a given question

## Configuration

### Changing the LLM Model

You can use any Hugging Face model compatible with text generation:

```python
llm = LLM("microsoft/phi-2", vectorstore=vectorstore)
# or
llm = LLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0", vectorstore=vectorstore)
```

### Adjusting Retrieval Parameters

Modify the retriever settings in the `LLM.__init__` method:

```python
self.retrieve = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,        # Number of documents to retrieve
        "fetch_k": 50  # Number of documents to fetch before MMR
    }
)
```

### Customizing the Prompt Template

Edit the `get_context` method in the `LLM` class to customize how the system prompts the model.

## Dependencies

- **chromadb**: Vector database for storing embeddings
- **PyMuPDF**: PDF text extraction
- **langchain**: Framework for LLM applications
- **langchain-chroma**: ChromaDB integration for LangChain
- **langchain-huggingface**: Hugging Face integration for LangChain
- **transformers**: Hugging Face transformer models
- **sentence-transformers**: Embedding models
- **torch**: PyTorch for model inference
- **accelerate**: Model optimization and loading

## Performance Tips

1. **Use smaller models** for faster inference (e.g., Qwen3-0.6B, TinyLlama)
2. **Enable persistent storage** to avoid re-processing documents
3. **Adjust `max_new_tokens`** in the pipeline for shorter/longer responses
4. **Use GPU** if available by ensuring PyTorch is installed with CUDA support

## Limitations

- Currently only supports PDF files
- Requires sufficient RAM to load the LLM model
- Processing large document collections may take time

## Future Enhancements

- Support for additional file formats (DOCX, TXT, HTML)
- Web interface for easier interaction
- Support for multiple LLM backends
- Document update and deletion capabilities
- Batch processing for large document sets

## License

This project is open source. Please check the repository for license details.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [ChromaDB](https://www.trychroma.com/)
- Uses models from [Hugging Face](https://huggingface.co/)
