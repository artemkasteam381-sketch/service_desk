"""Optional model adapters. Downloads occur only on explicit construction."""
import numpy as np


class E5Encoder:
    def __init__(self, model_name="intfloat/multilingual-e5-small", revision=None, device="cpu", batch_size=16):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, revision=revision, device=device, trust_remote_code=False)
        resolved = getattr(self.model[0].auto_model.config, "_commit_hash", None)
        self.signature = f"{model_name}@{resolved or revision or 'UNPINNED'}:mean-chunks384-overlap64:v1"
        self.batch_size = batch_size

    def _encode(self, texts, prefix):
        chunks, owners = [], []
        tokenizer = self.model.tokenizer
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text or "пустое обращение", add_special_tokens=False)
            for start in range(0, len(ids), 320):
                chunks.append(prefix + tokenizer.decode(ids[start:start+384], skip_special_tokens=True))
                owners.append(i)
                if start+384 >= len(ids): break
        vectors = self.model.encode(chunks, batch_size=self.batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
        out = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
        np.add.at(out, owners, vectors)
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)

    def encode_documents(self, texts): return self._encode(texts, "passage: ")
    def encode_queries(self, texts): return self._encode(texts, "query: ")


class MiniLMReranker:
    def __init__(self, model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", revision=None, device="cpu"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, revision=revision, device=device, max_length=512, trust_remote_code=False)
        resolved = getattr(self.model.model.config, "_commit_hash", None)
        self.signature = f"{model_name}@{resolved or revision or 'UNPINNED'}:q128-d350-max:v1"
        self.warnings = []

    def score(self, query, documents):
        import torch
        tokenizer = self.model.tokenizer
        qids = tokenizer.encode(query, add_special_tokens=False)
        self.warnings = ["reranker_query_truncated_to_128_tokens"] if len(qids)>128 else []
        query = tokenizer.decode(qids[:128], skip_special_tokens=True)
        pairs, owners = [], []
        for i, document in enumerate(documents):
            ids = tokenizer.encode(document, add_special_tokens=False)
            for start in range(0, max(1,len(ids)), 300):
                pairs.append((query, tokenizer.decode(ids[start:start+350], skip_special_tokens=True)))
                owners.append(i)
                if start+350 >= len(ids): break
        # Identity exposes logits explicitly; only their ORDER is used by SearchIndex.
        scores = np.asarray(self.model.predict(pairs, batch_size=16, show_progress_bar=False,
                                               activation_fn=torch.nn.Identity())).reshape(-1)
        out = np.full(len(documents), -np.inf)
        np.maximum.at(out, owners, scores)
        return out


class LocalTfidfEncoder:
    """Offline lexical baseline for testing plumbing, NOT an E5 replacement benchmark."""
    cacheable = False
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=12000)
        self.signature = "local-char-tfidf-3-5:12000:v1"
    def encode_documents(self, texts):
        # Persisted vector cache cannot restore the fitted TF-IDF vocabulary: rebuild baseline.
        return self.vectorizer.fit_transform(texts).astype(np.float32).toarray()
    def encode_queries(self, texts):
        return self.vectorizer.transform(texts).astype(np.float32).toarray()
