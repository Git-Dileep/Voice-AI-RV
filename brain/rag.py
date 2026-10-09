import os
from typing import List

class KnowledgeBase:
    def __init__(self, knowledge_dir: str = "brain/knowledge"):
        self.knowledge_dir = knowledge_dir
        self.documents = []
        self._load_documents()

    def _load_documents(self):
        """Loads text documents from the knowledge directory for local RAG."""
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir, exist_ok=True)
            return

        for filename in os.listdir(self.knowledge_dir):
            filepath = os.path.join(self.knowledge_dir, filename)
            if os.path.isfile(filepath) and filepath.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.documents.append(f.read())
                    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieves relevant document chunks based on the query.
        Returns a list of text chunks.
        """
        if not self.documents:
            return []
            
        results = []
        query_words = set(query.lower().split())
        scored_docs = []
        for doc in self.documents:
            doc_words = set(doc.lower().split())
            # Basic term overlap for baseline
            score = len(query_words.intersection(doc_words))
            if score > 0:
                scored_docs.append((score, doc))
        
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scored_docs[:top_k]]
