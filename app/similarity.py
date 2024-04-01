# app/similarity.py
from sentence_transformers import SentenceTransformer, util

class TextSimilarityModel:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def compute_similarity(self, text1: str, text2: str) -> float:
        embedding1 = self.model.encode(text1, convert_to_tensor=True)
        embedding2 = self.model.encode(text2, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()
        return similarity_score