"""Same windowing as supplied service; BGE has no E5 query/passage prefixes."""
import numpy as np


class BGEEncoder:
    def __init__(self, model_name='BAAI/bge-m3', revision=None, device='cpu', batch_size=8):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, revision=revision, device=device,
                                         trust_remote_code=False)
        self.model.max_seq_length = 512
        self.batch_size = batch_size

    def encode(self, texts):
        if not len(texts):
            return np.empty((0,self.model.get_sentence_embedding_dimension()),dtype=np.float32)
        chunks, owners = [], []
        tokenizer = self.model.tokenizer
        for i,text in enumerate(texts):
            ids = tokenizer.encode(text or 'пустое обращение',add_special_tokens=False)
            for start in range(0,max(1,len(ids)),320):
                chunks.append(tokenizer.decode(ids[start:start+384],skip_special_tokens=True))
                owners.append(i)
                if start+384 >= len(ids): break
        vectors = self.model.encode(chunks,batch_size=self.batch_size,normalize_embeddings=True,
                                    convert_to_numpy=True,show_progress_bar=True)
        out = np.zeros((len(texts),vectors.shape[1]),dtype=np.float32)
        np.add.at(out,owners,vectors)
        out /= np.maximum(np.linalg.norm(out,axis=1,keepdims=True),1e-12)
        if not np.isfinite(out).all(): raise ValueError('Invalid embeddings')
        return out


class MiniLMReranker:
    def __init__(self, model_name='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
                 revision=None,device='cpu'):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name,revision=revision,device=device,max_length=512,
                                  trust_remote_code=False)
        self.warnings = []

    def score(self, query, documents):
        import torch
        if not documents: return np.empty(0,dtype=np.float32)
        tokenizer = self.model.tokenizer
        qids = tokenizer.encode(query,add_special_tokens=False)
        self.warnings = ['reranker_query_truncated_to_128_tokens'] if len(qids)>128 else []
        query = tokenizer.decode(qids[:128],skip_special_tokens=True)
        pairs,owners = [],[]
        for i,document in enumerate(documents):
            ids = tokenizer.encode(document,add_special_tokens=False)
            for start in range(0,max(1,len(ids)),300):
                pairs.append((query,tokenizer.decode(ids[start:start+350],skip_special_tokens=True)))
                owners.append(i)
                if start+350 >= len(ids): break
        scores = np.asarray(self.model.predict(pairs,batch_size=16,show_progress_bar=False,
                          activation_fn=torch.nn.Identity())).reshape(-1)
        out = np.full(len(documents),-np.inf)
        np.maximum.at(out,owners,scores)
        return out
