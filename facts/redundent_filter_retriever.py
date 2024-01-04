from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever

class RedundentFilterRetriever(BaseRetriever):
    embeddings: Embeddings # expect user to provide embedding algorithm
    chroma: Chroma # expect user to provide configured Chroma instance

    def get_relevant_documents(self, query):
        # calculate embeddings for the query string
        emb = self.embeddings.embed_query(query)

        # take embeddings and feed them into that
        # max_marginal_relevance_search_by_vector
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8
        )
    
    async def aget_relevant_documents(self):
        return []