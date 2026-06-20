# RAG_AI

A dedicated attempt to learn AI engineering.

1: -Created a RAG_AI folder  

2:- check if python is available or not. python --version

3:- get https://www.python.org/  download LSR Python 3.14:  

4:- install and check again python3 --version  

5:- Install dependencies pip3 install langchain langchain-community langchain-text-splitters chromadb sentence-transformers 



—> More details  1. The framework •	langchain: This is the project manager. It doesn’t generate text or store data itself; its only job is to provide the structure (like a template pipeline) that connects your database to your AI model so they can talk to each other seamlessly.
* langchain-community: This is the translator/connector. LangChain on its own doesn't know what Ollama or ChromaDB are. The community package contains the specific custom bridges built by other developers to let LangChain plug directly into local tools like Ollama.
* langchain-text-splitters: This is the smart paper shredder. If you give an AI a massive wall of text all at once, it gets confused or slows down. This library takes your sample.txt file and neatly chops it into small, clean paragraphs (chunks) without cutting words in half.
2. The Smart Memory Bank
* chromadb: This is your Vector Database. Standard databases (like Excel or SQL) look for exact keyword matches. If you search for "automobile", they might miss a sentence containing "car". ChromaDB is different; it saves text based on meaning and concepts. It will create a hidden folder right inside your project directory to store your text chunks so they can be searched instantly.
3. The Mathematical Translator
* sentence-transformers: This is the hidden engine under the hood. Computers don't actually understand English words; they only understand numbers. This library downloads a tiny, ultra-lightweight AI model to your machine whose only job is to read your text chunks and convert them into long lists of coordinates (called embeddings or vectors). ChromaDB uses these coordinates to figure out how close the meaning of your text is to the user's question.
How they work together when you run your script:
Plaintext

[Your raw sample.txt]
       │
       ▼ (langchain-text-splitters chops it up)
[Tidy Text Chunks]
       │
       ▼ (sentence-transformers turns words into math coordinates)
[Vector Math Data]
       │
       ▼ (chromadb saves it in a local folder)
[Your Local Vector Database Store]


Check if llama is running 
Install and run LLM 
curl -fsSL https://ollama.com/install.sh | sh  
ollama --version  

🚀 Run app file python3 app.py      
