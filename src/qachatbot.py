import psycopg2
import torch
from transformers import BertTokenizer, BertModel

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_db_connection():
    return psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="postgres123",
        host="localhost"
    )

def get_bert_embedding(text):
    # Tokenize and encode the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the [CLS] token embedding as the sentence embedding
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings[0]  # Return as a 1D numpy array

def find_most_similar(query_embedding, cur, top_k=3):
    # Convert numpy array to a vector format for the SQL query
    query_embedding_vector = query_embedding.tolist()

    # Use the vector type in the SQL query
    cur.execute("""
    SELECT id, question, answer, embedding <-> %s::vector AS distance
    FROM university_qa
    ORDER BY distance
    LIMIT %s;
    """, (query_embedding_vector, top_k))

    return cur.fetchall()


def answer_question(question):
    # Encode the question using BERT
    query_embedding = get_bert_embedding(question)
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            results = find_most_similar(query_embedding, cur)
            
            print(f"Query: {question}\n")
            print("Top matches:")
            for i, (id, db_question, answer, distance) in enumerate(results, 1):
                print(f"\n{i}. Question: {db_question}")
                print(f"   Answer: {answer}")
                print(f"   Similarity Score: {1 - distance:.4f}")  # Convert distance to similarity
    finally:
        conn.close()

# Example usage
if __name__ == "__main__":
    while True:
        user_question = input("\nEnter your question (or 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break
        answer_question(user_question)