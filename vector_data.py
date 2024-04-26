import redis
import numpy as np

class RedisImageSearch:
    def __init__(self, host='localhost', port=6379, db=0):
        # Initialize the RedisImageSearch class with a connection to a Redis database.
        self.redis = redis.StrictRedis(host=host, port=port, db=db)

    def add_image_embedding(self, image_id, embedding):
        """ Store an image embedding in Redis. """
        # Ensure the embedding is a numpy array and flatten it to a list for storage
        if not isinstance(embedding, np.ndarray):
            raise ValueError("Embedding must be a numpy array")
        
        # Convert the numpy array to a list and store it in Redis as a comma-separated string
        embedding_list = embedding.flatten().tolist()
        # Store the embedding in Redis as a comma-separated string
        self.redis.hset("image_embeddings", image_id, ",".join(map(str, embedding_list)))
        
    def get_all_embeddings(self):
        """Retrieve all image embeddings stored in Redis and convert them back to numpy arrays.."""
        # Fetch all embeddings from Redis and convert them back to numpy arrays
        embeddings = {}
        for k, v in self.redis.hgetall("image_embeddings").items():
            # Decode the stored string and convert it back into a numpy array
            embedding_str = v.decode()
            embedding_list = list(map(float, embedding_str.split(',')))
            embedding_array = np.array(embedding_list)
            embeddings[k.decode()] = embedding_array
        return embeddings

    def find_similar_images(self, query_embedding, top_k=5, similarity_threshold=0.8):
        """Find similar images based on cosine similarity, applying a similarity threshold."""
        embeddings = self.get_all_embeddings()
        similarities = {}

        for image_id, embedding in embeddings.items():
            # Normalize embeddings and compute dot product for cosine similarity
            norm_query = query_embedding.squeeze() / np.linalg.norm(query_embedding.squeeze())
            norm_embedding = embedding / np.linalg.norm(embedding)
            similarity = np.dot(norm_query, norm_embedding)
            
            # Only consider embeddings that meet the similarity threshold
            if similarity > similarity_threshold:
                similarities[image_id] = similarity

        # Sort by similarity and return the top_k results
        sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        return sorted_similarities[:top_k]


# sec

