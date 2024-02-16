import openai

def create_embedding(text: str, embedding_model: str = "text-embedding-3-small"): 
    # Creates embedding vector from user query
    embedded_query = openai.embeddings.create(input=text,
                                                    model=embedding_model,
                                                    ).data[0].embedding
    return embedded_query


# Function to display the chat history
def display_chat_history(messages):
    for message in messages:
        print(f'''{message['role'].capitalize()}: {message['response']}''')