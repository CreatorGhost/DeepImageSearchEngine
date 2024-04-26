import redis
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np
class RedisVectorDatabase:
    def __init__(self, host='localhost', port=6379, password=None):
        self.client = redis.Redis(host=host, port=port, password=password, decode_responses=True)
        self.check_connection()

    def check_connection(self):
        try:
            self.client.ping()
            print("Connected to Redis successfully.")
        except redis.exceptions.ConnectionError:
            print("Failed to connect to Redis.")

    def create_index(self, index_name, vector_dimension):
        # Check if the index already exists
        try:
            # Attempt to get information about the index
            self.client.ft(index_name).info()
            # If the command above does not raise an exception, the index exists
            print(f"Index {index_name} already exists.")
            return
        except redis.exceptions.ResponseError as e:
            # If an exception is caught, it's likely because the index does not exist
            print(f"Index {index_name} does not exist. Creating now.")

        schema = (
            VectorField(
                "vector",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": vector_dimension,
                    "DISTANCE_METRIC": "COSINE",
                }
            ),
        )
        definition = IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH)
        self.client.ft(index_name).create_index(fields=schema, definition=definition)
        print(f"Index {index_name} created successfully.")

    def store_vector(self,index_name, doc_id, vector, metadata):
        # Ensure the vector is a NumPy array and convert it to bytes
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        vector_data = vector.tobytes()
        
        # Initialize a pipeline
        pipe = self.client.pipeline()
        
        # Use the correct prefix and store data using the pipeline
        redis_key = f"doc:{doc_id}"  # Adjusted to use the 'doc:' prefix
        metadata_bytes = {k: str(v) for k, v in metadata.items()}
        metadata_bytes["vector"] = vector_data
        pipe.hset(redis_key, mapping=metadata_bytes)
        
        # Execute the pipeline
        res = pipe.execute()
        print(f"Vector {doc_id} stored successfully.")
        
    def fetch_all_documents(self, index_name):
        print("Fetching all")
        try:
            results = self.client.ft(index_name).search("*")
            print("the res : ", results)
            documents = results.docs
            for doc in documents:
                doc_dict = doc.__dict__
                # Optionally, skip displaying the vector field if it's not meaningful as text
                if 'vector' in doc_dict:
                    doc_dict['vector'] = 'Vector data not displayed'
                print(doc.id, doc_dict)
            return documents
        except Exception as e:
            print(f"Error fetching documents from index {index_name}: {e}")
            return []
    
        
    def search_similar_vectors(self, index_name, query_vector, top_k=3, threshold=None):
        # Convert the query vector to bytes as expected by the query
        query_vector_bytes = query_vector.astype(np.float32).tobytes()
        query = (
                Query(f"*=>[KNN {top_k} @vector $vec AS score]")
                .sort_by("score")
                .return_fields("id", "score")
                .paging(0, top_k)
                .dialect(2)
            )
        query_params = {
                "vec": query_vector_bytes
            }
        # Execute the search query
        try:
            results = self.client.ft(index_name).search(query, query_params).docs
            if threshold is not None:
                results = [result for result in results if hasattr(result, 'score') and float(result.score) < threshold]
        except Exception as e:
            print(f"Error executing search: {e}")
            return []

        # Process and return the results
        return results

if __name__ == "__main__":
    db = RedisVectorDatabase()
    index_name = "myVectorIndex"
    vector_dimension = 512  # Example dimension

    # Create an index for storing vectors
    db.create_index(index_name, vector_dimension)

    # # Example vector and metadata
    # example_vector = []  # Your 512-dimension vector here
    # metadata = {"name": "Example", "type": "ExampleType"}

    # # Store a vector
    # db.store_vector(index_name, "doc1", example_vector, metadata)

    # # Perform a similarity search
    # results = db.search_similar_vectors(index_name, query_vector=example_vector)
    # for result in results:
    #     print(result.id, result.vector_score)