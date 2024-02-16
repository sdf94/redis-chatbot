import redis


class RedisHandler:
    def __init__(self, host: str = 'localhost', port: int = 6379 , password: str = None) -> None:
        self.host = host
        self.port = port
        self.password = password

    def connect(self) -> None:
        """
        Connect to the Redis server.
        """
        self.redis_client = redis.Redis(host=self.host, port=self.port, password=self.password)
        return  self.redis_client

    def disconnect(self) -> None:
        """
        Disconnect from the Redis server.
        """
        if self.redis_client is not None:
            self.redis_client.close()
