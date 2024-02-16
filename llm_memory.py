import time
import numpy as np
from redis.commands.search.query import Query
from typing import List
from redis_handler import RedisHandler
from helper import create_embedding


class LLMMemory(RedisHandler):
    def __init__(self, connection: RedisHandler):
        self.redis_client = connection

    def add(self,message: dict, prefix="doc"):

        """
        Add conversation interactions to the memory layer.

        Args:
            message (str): Message to be stored.
        """

        key = f"{prefix}:{str(message['role'])}:{str(int(time.time()))}"
        message['response_vector'] = np.array(create_embedding(message['response'])).tobytes()
        self.redis_client.hset(key, mapping=message)

    def fetch(self, user_query: str, 
              index_name: str = "embeddings-index", 
              vector_field: str = "response_vector",
              return_fields: list = ["role", "response", "vector_score"],
              hybrid_fields = "*",
              k: int = 5,
              embedding_model: str = "text-embedding-3-small") -> List[dict]:
        """
            Fetch conversation history relevant to the provided context.

            Args:
                user_query (str): The query for which relevant conversation history is to be fetched.
                index_name (str): The name of the Redis index to search in. Default is "embeddings-index".
                vector_field (str): The field containing the vectors in the Redis index. Default is "response_vector".
                return_fields (list): The list of fields to return for each interaction. Default is ["role", "response", "vector_score"].
                hybrid_fields (str): The hybrid fields to be used in the search query. Default is "*".
                k (int): The number of nearest neighbors to retrieve. Default is 5.
                embedding_model (str): The name of the embedding model to use. Default is "text-embedding-3-small".

            Returns:
                list: List of conversation interactions relevant to the user query.
            """

        # Creates embedding vector from user query
        embedded_query = create_embedding(user_query)
            
        # Prepare the Query
        base_query = f'{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]'
        query = (
                Query(base_query)
                .return_fields(*return_fields)
                .sort_by("vector_score")
                .paging(0, k)
                .dialect(2)
            )
        params_dict = {"vector": np.array(embedded_query).astype(dtype=np.float32).tobytes()}

        # perform vector search
        results = self.redis_client.ft(index_name).search(query, params_dict)
        return results.docs
   
    def clear(self):
        """
        Clear the memory layer.
        """
        self.redis_client.flushdb()
