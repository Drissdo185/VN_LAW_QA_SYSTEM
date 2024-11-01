import streamlit as st
from indexing import Indexing

# Initialize the Indexing class
indexer = Indexing(path="./chroma_db", chunk_size=512, chunk_overlap=40)

def main():
    st.title("Document Retrieval System")
    
    # Display text input for querying
    query = st.text_input("Enter your query:")
    
    if st.button("Search"):
        # Retrieve the answer based on the query
        answer = indexer.retrieve(query=query)
        
        # Extract the answer text (assuming it's inside an object or response field)
        if isinstance(answer, str):
            st.write("Answer:", answer)
        else:
            st.write("Answer:", answer.response)  # Adjust this line based on how your response is structured.

if __name__ == "__main__":
    # Embed and index documents when starting the app
    indexer.embed_and_index()
    main()
