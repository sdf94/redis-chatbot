import openai
import os
from redis.commands.search.indexDefinition import (
    IndexDefinition,
    IndexType
)
from redis.commands.search.field import (
    TextField,
    VectorField)
from helpers.redis_handler import RedisHandler
from helpers.llm_memory import LLMMemory
from helpers.helper import display_chat_history

try:
    redis_host = os.environ['HOST']
    redis_pass = os.environ['PASS']
except KeyError:
    print("We need a host, password, and port in order to login to Redis.")

try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
except KeyError:
    print("We need a openai API key")

def setup_schema(redis_link) -> None:
    # Constants
    VECTOR_DIM = 1536
    INDEX_NAME = "embeddings-index"                 # name of the search index
    PREFIX = "doc"                                  # prefix for the document keys
    DISTANCE_METRIC = "COSINE"                      # distance metric for the vectors (ex. COSINE, IP, L2)

    # Define RediSearch fields for each of the columns in the dataset
    role = TextField(name="role")
    response = TextField(name="response") 
    response_embedding = VectorField("response_vector",
        "FLAT", {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": DISTANCE_METRIC,
        }
    )
    fields = [role, response, response_embedding]

    try:
        redis_link.ft(INDEX_NAME).info()
        print("Index already exists")
    except:
        # Create RediSearch Index
        redis_link.ft(INDEX_NAME).create_index(
            fields = fields,
            definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
        )

# Function to get the assistant's response
def get_assistant_response(user_prompt, context) -> str:
    prompt = f'''

        Use ONLY the context below to answer the question. If you do not know the answer, make something up.

        Context:
        {context}

        Question: {user_prompt}
        Answer:
        '''
    
    if not context:
        context = [{"role": "user", "content": prompt}] 
       
    r = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": m["role"], "content": m["content"]} for m in context],
        )
    response = r.choices[0].message.content
    return response


if __name__ == "__main__":
    redis_client = RedisHandler(host = redis_host, password = redis_pass)
    redis_conn = redis_client.connect()
    setup_schema(redis_conn)
    llm_memory = LLMMemory(redis_conn)

    while True:
        # Get user input
        user_input = input("User: ")
        if user_input.lower() == 'bye':
            break
        llm_memory.add({"role": "user", "response": user_input})

        history = llm_memory.fetch(user_input)

        # Get assistant response
        response_text = get_assistant_response(user_input, history)

        print(f'Assistant: {response_text}')
        llm_memory.add({"role": "assistant", "response": response_text})

    display_chat_history(history)