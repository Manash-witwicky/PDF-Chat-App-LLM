import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv
import faiss
import os
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain


with st.sidebar:
    st.title("LLM Powered PDF Chat App")
    add_vertical_space(38)
    st.markdown(
    '<div style="text-align: center;">Made with ❤️ by Manash</div>',
    unsafe_allow_html=True
    )

def main():
    load_dotenv()
    st.header("Upload and Chat with PDF")
    uploaded_file = st.file_uploader("Upload a File", type='pdf')
    # st.write(uploaded_file.name)
    # st.write(uploaded_file.file_id)
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)

        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

        ## creating the chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text=text)
    
        # embeddings
        
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        file_name = uploaded_file.name[:-4]

        # if os.path.exists(f"{file_name}.index"):
        #     vector_store = faiss.read_index(f"{file_name}.index")
        #     st.write("Embeddings loaded from disk")
        # else:
        #     faiss.write_index(vector_store.index, f"{file_name}.index")
        #     st.write("Embeddings saved to disk")

        if os.path.exists(f"{file_name}.faiss"):
            vector_store = FAISS.load_local(file_name, embeddings)
            st.write("Embeddings loaded from disk")
        else:
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            vector_store.save_local(file_name)
            st.write("Embeddings saved to disk")

        # accept user query

        query = st.text_input("Ask Question about your PDF file")

        if query:
            docs = vector_store.similarity_search(query=query, k=1)
            # st.write(docs)
        
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)


        # Create and open a .pkl file, then save the vectors in it.
        # .pkl file used to store python objects
        #with open(f"{file_name}.pkl", "wb") as f:
        #    pickle.dump(vectore_store, f)

if __name__ == "__main__":
    main()
