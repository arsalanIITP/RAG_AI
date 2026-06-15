from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# =====================================================================
# STEP 1: INGESTION (Load the raw files)
# =====================================================================
print("\n🔄 Step 1: Reading files from './knowledge_base'...")
loader = DirectoryLoader('./knowledge_base', glob="*.txt", loader_cls=TextLoader)
documents = loader.load()
print(f"✅ Successfully loaded {len(documents)} document(s).")

# =====================================================================
# STEP 2: CHUNKING (Chop the text up)
# =====================================================================
print("\n✂️ Step 2: Fragmenting text into clean semantic chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} text chunks.")

# =====================================================================
# STEP 3: EMBED & VECTORIZE (Turn words into mathematical space)
# =====================================================================
print("\n🧠 Step 3: Initializing local embedding engine (all-MiniLM-L6-v2)...")
print("📥 Note: On the very first run, this will download a tiny 90MB math model.")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("\n🗄️ Step 4: Indexing vectors into a local ChromaDB directory...")
# This sets up a vector store and builds the local folder './my_vdb'
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./my_vdb")
retriever = vector_store.as_retriever(search_kwargs={"k": 1}) # Grabs the #1 single best match
print("✅ Vector database initialized and persisted.")

# =====================================================================
# STEP 4: MODEL CONFIGURATION (Connect to your local Ollama server)
# =====================================================================
print("\n🤝 Step 5: Establishing connection to local Ollama instance (phi3)...")
# Note: Ensure you have run 'ollama run phi3' or 'ollama run llama3' in a terminal
llm = Ollama(model="llama3.2")

# =====================================================================
# STEP 5: PROMPT DESIGN & CHAIN CONSTRUCTION (The RAG Pipeline)
# =====================================================================
# This defines the template that forces the AI to stay grounded in your data
template = """Answer the question based only on the following context. If you do not know the answer based on the context, say clearly that the information is missing.

Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Helper function to join the retrieved document text strings together
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# The modern LCEL Chain pipeline:
# 1. Take the user query, pass it to retriever, format the output as context string.
# 2. Pass both context and query into our structural prompt template.
# 3. Feed the filled prompt to the local LLM.
# 4. Clean up the response as a string.
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# =====================================================================
# STEP 6: EXECUTION
# =====================================================================
print("\n🚀 System Ready! Enter interactive chat mode.")

COMMANDS = {
    "end_chat": "Exit the interactive chat",
    "commands": "List available commands",
}

def print_commands():
    print("\nAvailable commands:")
    for c, desc in COMMANDS.items():
        print(f" - {c}: {desc}")
    print("")

print("Type 'commands' to see available commands. Type 'end_chat' to exit.")

try:
    while True:
        user_query = input("\nYou: ").strip()
        if not user_query:
            continue

        if user_query == "end_chat":
            print("Ending chat. Goodbye!")
            break

        if user_query == "commands":
            print_commands()
            continue

        print("🤔 Searching vector database and generating answer...")
        try:
            # send the query through the RAG chain and print the result
            result = rag_chain.invoke(user_query)
            print("\n✨ --- Local LLM Answer --- ✨")
            print(result)
            print("-----------------------------\n")
        except Exception as e:
            print(f"Error during model invocation: {e}")
            # continue the loop after the error
except (KeyboardInterrupt, EOFError):
    print("\nChat terminated by user.")