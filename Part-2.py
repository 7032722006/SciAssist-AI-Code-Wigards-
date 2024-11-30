# Function to prepare and split documents
def prepare_and_split_docs(pdf_directory):
    split_docs = []
    for pdf in pdf_directory:

        with open(pdf.name, "wb") as f:
            f.write(pdf.getbuffer())
        
        # Use PyPDFLoader with the saved file path
        loader = PyPDFLoader(pdf.name)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=512,
            chunk_overlap=256,
            disallowed_special=(),
            separators=["\n\n", "\n", " "]
        )
        split_docs.extend(splitter.split_documents(documents))
    return split_docs

# Function to ingest documents into the vector database
def ingest_into_vectordb(split_docs):
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
    
    db = FAISS.from_documents(split_docs, embeddings)
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    return db

# Function to get the conversation chain
def get_conversation_chain(retriever):
    llm = Ollama(model="llama3.2")
    contextualize_q_system_prompt = (
        "Given the chat history and the latest user question, "
        "provide a response that directly addresses the user's query based on the provided  documents. "
        "Do not rephrase the question or ask follow-up questions."
    )


    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    ### Answer question ###
    system_prompt = (
        """As a personal chat assistant, You are programmed to deliver precise and pertinent information strictly 
        based on the provided documents. Each response will be concise, limited to 150 words and formulated in 2-3 sentences, 
        without any options for selection, standalone questions, or queries included. use this {context}"""
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    ### Statefully manage chat history ###
    store = {}


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain