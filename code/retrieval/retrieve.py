import json
import os
import numpy as np
import math
from tqdm import tqdm
from collections import defaultdict
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from gritlm import GritLM
from promptriever import Promptriever
import time
from openai import OpenAI
from google import genai
from dotenv import load_dotenv
load_dotenv()

#client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])
# gclient = genai.Client(
#     vertexai=True, 
#     project=os.environ["GEMINI_PROJECT"],
#     location=os.environ["GEMINI_LOCATION"],
#     http_options={"api_version": "v1"}
# )

model_to_path = {
    "nv1": "nvidia/NV-Embed-v1",
    "nv2": "nvidia/NV-Embed-v2",
    "qwen-1-5": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "qwen-7": "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "contriever": "facebook/contriever-msmarco",
    "tuned-contriever": "/scratch/dq2024/models/diverse_retriever/checkpoints/qampari_contriever_standard_only_random_0123/checkpoint/best_model",
    'qwen3-0-6': "Qwen/Qwen3-Embedding-0.6B",
}
openai_model_to_api = {
    "text-3-small": "text-embedding-3-small",
    "text-3-large": "text-embedding-3-large",
    "text-ada": "text-embedding-ada-002",
}
gemini_model_to_api = {
    # "gemini-embedding": "text-embedding-large-exp-03-07",
    "gemini-embedding": "gemini-embedding-exp-03-07"
}
supported_models = [
    "nv1", "nv2", "qwen-1-5", "qwen-7", "gritlm", "text-3-small", 
    "text-3-large", "text-ada", "promptriever", "gemini-embedding", 
    "bm25", "contriever", "tuned-contriever", "qwen3-0-6"
]

prompt = "Given a question, retrieve passages that answer the question"
query_prefix = "Instruct: " + prompt + "\nQuery: "

def add_eos(input_examples):
    input_examples = [
        input_example + model.tokenizer.eos_token for input_example in input_examples
    ]
    return input_examples

def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

def promptriever_query(query, domain):
    if domain == "story":
        instruction = "A relevant document would be a story chapter that answers the query. I am not interested in any chapter that appears to be from a different story than the one related to the query. Think carefully about these conditions when determining relevance."
    elif domain == "meeting":
        instruction = "A relevant document would be a meeting transcript that answers the query. I am not interested in any meeting transcript that appears to be about a different discussion than the one related to the query. Think carefully about these conditions when determining relevance."
    return f"query:  {query.strip()} {instruction.strip()}".strip()


def get_embedding(model_name, model, domain, text, max_tokens=1000, is_query=False):
    """Fetches embeddings for the given text using OpenAI.
    If the text exceeds the maximum token limit, it breaks the text into chunks.
    """
    text = text.replace("\n", " ")
    # Tokenize the text to count tokens (this is a simplistic example; a more 
    # accurate method could be used)
    tokens = text.split()
    if len(tokens) <= max_tokens:
        # If the text fits within the token limit, get the embedding as usual
        if model_name in openai_model_to_api:
            result = client.embeddings.create(
                input=text, 
                model=openai_model_to_api[model_name]
            )
            return result.data[0].embedding
        if model_name in gemini_model_to_api:
            time.sleep(1)
            result = gclient.models.embed_content(
                model=gemini_model_to_api[model_name], contents=text
            )
            return result.embeddings[0].values
        if is_query:
            if model_name in model_to_path:
                if model_name == "contriever" or model_name == "tuned-contriever":
                    # Contriever doesn't use instruction prefixes
                    return model.encode(
                        [text], 
                        batch_size=1, 
                        normalize_embeddings=False
                    )[0]
                else:
                    return model.encode(
                        add_eos([text]), 
                        batch_size=1, 
                        prompt=query_prefix, 
                        normalize_embeddings=True
                    )[0]
            elif model_name == "gritlm":
                return model.encode(
                    [text], 
                    instruction=gritlm_instruction(prompt)
                )[0]
            elif model_name == "promptriever":
                return model.encode(
                    [promptriever_query(text, domain)]
                )[0]
        else:
            if model_name in model_to_path:
                if model_name == "contriever" or model_name == "tuned-contriever":
                    # Contriever doesn't use instruction prefixes
                    return model.encode(
                        [text], 
                        batch_size=1, 
                        normalize_embeddings=False
                    )[0]
                else:
                    return model.encode(
                        add_eos([text]), 
                        batch_size=1, 
                        normalize_embeddings=True
                    )[0]
            elif model_name == "gritlm":
                return model.encode(
                    [text], 
                    instruction=gritlm_instruction("")
                )[0]
            elif model_name == "promptriever":
                return model.encode(
                    [f"passage:  {text}"]
                )[0]
    else:
        # If the text exceeds the token limit, break it into chunks
        chunks = [
            tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)
        ]
        # Convert chunks back to string
        chunks = [" ".join(chunk) for chunk in chunks]
        chunk_embeddings = [0 for _ in range(len(chunks))]
        # Get embeddings for each chunk
        if model_name in openai_model_to_api:
            chunk_embeddings = np.array([
                client.embeddings.create(
                    input=chunk, 
                    model=openai_model_to_api[model_name]
                ).data[0].embedding for chunk in chunks
            ])
        elif model_name in gemini_model_to_api:
            time.sleep(len(chunks))
            chunk_embeddings = np.array([
                gclient.models.embed_content(
                    model=gemini_model_to_api[model_name], 
                    contents=chunk
                ).embeddings[0].values for chunk in chunks
            ])
        elif model_name == "contriever" or model_name == "tuned-contriever":
            chunk_embeddings = []
            for chunk in chunks:
                chunk_embeddings.append(
                    model.encode([chunk], batch_size=1, normalize_embeddings=False)[0]
                )
            chunk_embeddings = np.array(chunk_embeddings)
        else:
            for i in range(len(chunks)):
                if is_query:
                    if model_name in model_to_path:
                        chunk_embeddings[i] = model.encode(
                            add_eos([chunks[i]]), 
                            batch_size=1, 
                            prompt=query_prefix, 
                            normalize_embeddings=True
                        )[0]
                    elif model_name == "gritlm":
                        chunk_embeddings[i] = model.encode(
                            [text],
                            instruction=gritlm_instruction(prompt)
                        )[0]
                    elif model_name == "promptriever":
                        chunk_embeddings[i] = model.encode(
                            [promptriever_query(text, domain)]
                        )[0]
                else:
                    if model_name in model_to_path:
                        chunk_embeddings[i] = model.encode(
                            add_eos([chunks[i]]), 
                            batch_size=1, 
                            normalize_embeddings=True
                        )[0]
                    elif model_name == "gritlm":
                        chunk_embeddings[i] = model.encode(
                            [text],
                            instruction=gritlm_instruction("")
                        )[0]
                    elif model_name == "promptriever":
                        chunk_embeddings[i] = model.encode(
                            [f"passage:  {text}"]
                        )[0]

        # Calculate weights based on the number of tokens in each chunk
        weights = [len(chunk.split()) for chunk in chunks]
        # Calculate the weighted average of the embeddings
        weighted_avg_embedding = np.average(
            chunk_embeddings, axis=0, weights=weights
        )
        return weighted_avg_embedding

 
class InformationRetrieval:
    def __init__(
        self, 
        query_file, 
        meeting_folder, 
        embedding_file
    ):
        with open(query_file, "r") as file:
            self.queries_dict = json.load(file)
        self.results = {"MIN": [], "MEAN": [], "MAX": []}
        self.performance_metrics = {"MIN": [], "MEAN": [], "MAX": []}
        self.meeting_ids = []
        self.meetings = []
        self.embedding_file = embedding_file

        for filename in sorted(os.listdir(meeting_folder)):
            meeting_id = filename.split(".")[0]  # Assuming filename is like "TS3009d.txt"
            self.meeting_ids.append(meeting_id)

            with open(os.path.join(meeting_folder, filename), "r") as file:
                self.meetings.append(file.read())

        # Read meetings and create a mapping from meeting IDs to texts
        self.meeting_texts = {}
        for filename in os.listdir(meeting_folder):
            meeting_id = filename.split(".")[0]
            with open(os.path.join(meeting_folder, filename), "r") as file:
                self.meeting_texts[meeting_id] = file.read()

    def precompute_embeddings(self, model_name, model, domain):
        embeddings = {}
        for meeting_id, meeting_text in tqdm(zip(self.meeting_ids, self.meetings), desc="Precomputing embeddings", total=len(self.meetings)):
            # Assuming get_embedding function is available and returns a NumPy array
            embeddings[meeting_id] = get_embedding(
                model_name, model, domain, meeting_text
            )
            if isinstance(embeddings[meeting_id], np.ndarray):
                embeddings[meeting_id] = embeddings[meeting_id].tolist()
        return embeddings

    def llm_embedding(
        self,
        model_name,
        model,
        domain,
        query,
        n,
        rankings = None
    ):
        # Compute the query embedding
        query_embedding = np.array(get_embedding(model_name, model, domain, query, is_query=True)).reshape(1, -1)
        # Create a 2D array for document embeddings
        doc_embeddings = np.array([self.embeddings[meeting_id] for meeting_id in self.meeting_ids])
        # Calculate similarity scores
        doc_scores = cosine_similarity(query_embedding, doc_embeddings)[0]
        if rankings != None:
            top_20_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:20]
            top_20_meeting_ids = [self.meeting_ids[i] for i in top_20_indices]
            rankings[query] = top_20_meeting_ids
        # Sort by similarity score and select top n indices
        top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:n]
        # Get the corresponding top n meeting IDs
        top_n_meeting_ids = [self.meeting_ids[i] for i in top_n_indices]
        return top_n_meeting_ids

    def bm25(self, query, n, rankings = None):
        tokenized_meetings = [doc.split(" ") for doc in self.meetings]
        tokenized_query = query.split(" ")
        bm25 = BM25Okapi(tokenized_meetings)
        doc_scores = bm25.get_scores(tokenized_query)
        if rankings != None:
            top_20_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:20]
            top_20_meeting_ids = [self.meeting_ids[i] for i in top_20_indices]
            rankings[query] = top_20_meeting_ids
        top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:n]
        top_n_meeting_ids = [self.meeting_ids[i] for i in top_n_indices]
        return top_n_meeting_ids

    # Function to calculate performance metrics
    def evaluate_performance(self, retrieved_meetings, ground_truth_meetings):
        retrieved_set = set(retrieved_meetings)
        ground_truth_set = set(ground_truth_meetings)

        tp = len(retrieved_set.intersection(ground_truth_set))
        fp = len(retrieved_set) - tp
        fn = len(ground_truth_set) - tp

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        # NDCG calculation
        dcg = 0
        ideal_dcg = 0
        for i, doc in enumerate(retrieved_meetings):
            rel = 1 if doc in ground_truth_set else 0
            dcg += rel / math.log2(i + 2)  # i starts from 0, log starts from 2
        for i in range(min(len(ground_truth_meetings), len(retrieved_meetings))):
            ideal_dcg += 1 / math.log2(i + 2)
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0

        # MAP calculation (Average Precision for this query)
        num_relevant = 0
        sum_precision = 0
        for i, doc in enumerate(retrieved_meetings):
            if doc in ground_truth_set:
                num_relevant += 1
                sum_precision += num_relevant / (i + 1)
        ap = sum_precision / len(ground_truth_set) if ground_truth_set else 0
        #return {"Precision": precision, "Recall": recall, "F1": f1, "NDCG": ndcg, "AP": ap}
        return {"Precision": precision, "Recall": recall, "NDCG": ndcg, "AP": ap}

    def run_evaluation(
        self, 
        method, 
        model_name,
        model=None,
        domain="story",
        n_values=[1, 3, 6], 
        labels=["MIN", "MEAN", "MAX"]
    ):
        if method == "llm_embedding":
            # Try to load precomputed embeddings from file
            if os.path.exists(self.embedding_file):
                with open(self.embedding_file, "r") as f:
                    self.embeddings = json.load(f)
                print("Loaded precomputed embeddings from file.")
            else:
                print("No precomputed embeddings found. Computing now...")
                # Dictionary to store pre-computed embeddings
                self.embeddings = self.precompute_embeddings(
                    model_name, model, domain
                )  
                # Save the computed embeddings to a file
                with open(self.embedding_file, "w") as f:
                    json.dump(self.embeddings, f)
                print(f"Saved precomputed embeddings to {self.embedding_file}.")

        # Initialize dictionary to store sum and count for each metric and label
        for n, label in zip(n_values, labels):
            performance_metrics = defaultdict(
                lambda: defaultdict(lambda: {"sum": 0.0, "count": 0})
            )
            # Initialize list to hold results for this method and top_k
            rankings = {}
            for query_id, query_data in tqdm(self.queries_dict.items()):
                query = query_data["query"]
                ground_truth_meetings = query_data["gold_documents"]
                if method == "llm_embedding":
                    retrieved_meetings = self.llm_embedding(
                        model_name, model, domain, query, n, rankings
                    )
                elif method == "bm25":
                    retrieved_meetings = self.bm25(query, n, rankings)
                # Evaluate performance
                metrics = self.evaluate_performance(retrieved_meetings, ground_truth_meetings)
                # Accumulate for average metrics
                for metric_name, metric_value in metrics.items():
                    performance_metrics[label][metric_name]["sum"] += metric_value
                    performance_metrics[label][metric_name]["count"] += 1

            if len(rankings) > 0:
                with open("rankings.json", "w") as f:
                    json.dump(rankings, f)

        for label, metrics in performance_metrics.items():
            print(f"Performance Metrics for {label}({n}):")
            for metric_name, values in metrics.items():
                avg_value = values["sum"] / values["count"]
                print(f"{metric_name}: {avg_value:.4f}")
            print()

# ========= Two-round helpers (non-intrusive) =========

def verifier_oracle(r1_ranking, gold_documents, k):
    """Select up to K docs from R1 that are actually gold (upper bound)."""
    gold = set(gold_documents)
    return [d for d in r1_ranking if d in gold][:k]

def build_augmented_query(base_query, selected_ids, id2text, max_tokens_per_doc=256):
    """Concatenate query + short snippets from the selected docs in the specified format."""
    def head_tokens(text, n): 
        toks = text.split()
        return " ".join(toks[:n])
    
    # Print base query info
    print(f"\n[Augmented Query Construction]")
    print(f"  Base query tokens: {len(base_query.split())}")
    print(f"  Documents to include: {len(selected_ids)}")
    
    # Start with "Question: [query]"
    augmented = f"Question: {base_query.strip()}"
    
    # Collect document snippets
    doc_snippets = []
    total_original_tokens = 0
    total_kept_tokens = 0
    
    for did in selected_ids:
        if did in id2text:
            full_text = id2text[did]
            original_tokens = len(full_text.split())
            snippet = head_tokens(full_text, max_tokens_per_doc)
            snippet_tokens = len(snippet.split())
            
            total_original_tokens += original_tokens
            total_kept_tokens += snippet_tokens
            
            # Print truncation info per document
            if original_tokens > max_tokens_per_doc:
                print(f"  Doc {did}: TRUNCATED from {original_tokens} to {snippet_tokens} tokens")
            else:
                print(f"  Doc {did}: {snippet_tokens} tokens (no truncation needed)")
            
            # Format each doc with its ID as a prefix
            doc_snippets.append(f"{did} {snippet}")
    
    # Add documents section if we have any
    if doc_snippets:
        documents_text = "\n".join(doc_snippets)
        augmented += f"\n\nDocuments: {documents_text}"
    
    # Print summary statistics
    total_augmented_tokens = len(augmented.split())
    print(f"\n  Summary:")
    print(f"    Total document tokens before truncation: {total_original_tokens}")
    print(f"    Total document tokens after truncation: {total_kept_tokens}")
    print(f"    Tokens discarded: {total_original_tokens - total_kept_tokens}")
    print(f"    Final augmented query length: {total_augmented_tokens} tokens")
    
    # Warning if exceeds contriever limit
    if total_augmented_tokens > 512:
        print(f"  ⚠️  WARNING: Augmented query has {total_augmented_tokens} tokens, exceeds contriever's 512 limit!")
        print(f"      Will be handled by chunking/averaging in get_embedding()")
    
    return augmented

def dedup_union(list_a, list_b):
    seen, out = set(), []
    for x in list_a + list_b:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def load_or_compute_embeddings(ir, model_name, model, domain, embeddings_json_path):
    """Return {doc_id: embedding(list[float])}; compute & cache if missing."""
    path = embeddings_json_path if embeddings_json_path.endswith(".json") else embeddings_json_path + ".json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    emb = ir.precompute_embeddings(model_name, model, domain)
    with open(path, "w") as f:
        json.dump(emb, f)
    return emb

def llm_embedding_with_given_embeddings(ir, model_name, model, domain, query, n, embeddings_dict):
    """Rank docs using *provided* embeddings dict (keeps your original .embeddings untouched)."""
    q = np.array(get_embedding(model_name, model, domain, query, is_query=True), dtype=float).reshape(1, -1)
    D = np.array([embeddings_dict[mid] for mid in ir.meeting_ids], dtype=float)
    sims = cosine_similarity(q, D)[0]
    idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:n]
    return [ir.meeting_ids[i] for i in idxs]

def run_two_round_pipeline(
    ir,
    domain,
    split,
    base_model_name, base_model, base_embeddings_path,   # Round 1
    tuned_model_name, tuned_model, tuned_embeddings_path, # Round 2
    n_eval=8,                     
    K_list=(1,2,3),
    final_cap=None,               
    max_tokens_per_doc=256,
    verifier_mode="oracle"
):
    """
    Two-round retrieval where both rounds retrieve exactly n_eval docs,
    and metrics are computed on top-n_eval (mirrors vanilla).
    Outputs are written under:
      outputs/two_round/{domain}/r1-{base_model_name}__r2-{tuned_model_name}/K{K}/...
    """
    # Prepare per-model embedding caches
    emb_base  = load_or_compute_embeddings(ir, base_model_name,  base_model,  domain, base_embeddings_path)
    emb_tuned = load_or_compute_embeddings(ir, tuned_model_name, tuned_model, domain, tuned_embeddings_path)

    # Output root includes model names so runs are self-describing
    combo_dir = f"r1-{base_model_name}__r2-{tuned_model_name}"
    out_root = os.path.join("outputs", "two_round", domain, combo_dir)
    os.makedirs(out_root, exist_ok=True)

    for K in K_list:
        results = []
        agg = {"R1": defaultdict(float), "R2": defaultdict(float), "Final": defaultdict(float)}
        n_q = 0
        total_docs_used = 0

        for qid, qdata in ir.queries_dict.items():
            query = qdata["query"]
            gold  = qdata["gold_documents"]

            r1_eval = llm_embedding_with_given_embeddings(
                ir, base_model_name, base_model, domain, query, n_eval, emb_base
            )

            if verifier_mode == "oracle":
                r1_sel = verifier_oracle(r1_eval, gold, K)
            else:
                raise NotImplementedError("Only 'oracle' verifier implemented.")

            total_docs_used += len(r1_sel)

            aug_query = build_augmented_query(query, r1_sel, ir.meeting_texts, max_tokens_per_doc=max_tokens_per_doc)

            r2_eval = llm_embedding_with_given_embeddings(
                ir, tuned_model_name, tuned_model, domain, aug_query, n_eval, emb_tuned
            )

            final_full = dedup_union(r1_eval, r2_eval)
            if final_cap is not None:
                final_full = final_full[:final_cap]
            final_eval = final_full[:n_eval]

            m_r1    = ir.evaluate_performance(r1_eval,    gold)
            m_r2    = ir.evaluate_performance(r2_eval,    gold)
            m_final = ir.evaluate_performance(final_eval, gold)
            for k,v in m_r1.items():    agg["R1"][k]    += v
            for k,v in m_r2.items():    agg["R2"][k]    += v
            for k,v in m_final.items(): agg["Final"][k] += v
            n_q += 1

            results.append({
                "qid": qid,
                "query": query,
                "gold_documents": gold,
                "R1_eval_topk": r1_eval,       
                "R1_selected_K": r1_sel,
                "augmented_query": aug_query,
                "R2_eval_topk": r2_eval,       
                "final_ranking_full": final_full, 
                "final_eval_topk": final_eval    
            })

        avg_docs_used = total_docs_used / n_q if n_q > 0 else 0

        k_dir = os.path.join(out_root, f"K{K}")
        os.makedirs(k_dir, exist_ok=True)

        out_json = os.path.join(k_dir, f"{split}.json")
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)

        def _avg(d): return {k: (v / n_q if n_q else 0.0) for k,v in d.items()}
        avg_r1, avg_r2, avg_fin = _avg(agg["R1"]), _avg(agg["R2"]), _avg(agg["Final"])

        out_txt = os.path.join(k_dir, "metrics.txt")
        with open(out_txt, "w") as f:
            f.write(f"K={K} (domain={domain}, split={split}, r1={base_model_name}, r2={tuned_model_name}, n_eval={n_eval})\n")
            f.write(f"Average docs used for augmentation: {avg_docs_used:.2f} / {K}\n")
            f.write("R1 -> "    + ", ".join(f"{k}: {100 * v:.4f}" for k,v in avg_r1.items())  + "\n")
            f.write("R2 -> "    + ", ".join(f"{k}: {100 * v:.6f}" for k,v in avg_r2.items())  + "\n")
            f.write("Final -> " + ", ".join(f"{k}: {100 * v:.4f}" for k,v in avg_fin.items()) + "\n")

        print(f"[two-round] Wrote {out_json} and {out_txt}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval")
    parser.add_argument("--queries_path", type=str, required=True, help="Path to JSON file of queries")
    parser.add_argument("--documents_path", type=str, required=True, help="Path to directory containing text files of documents")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to JSON embeddings file")
    parser.add_argument("--method", required=True, choices=["llm_embedding", "bm25"], help="Retrieval method to run")
    parser.add_argument("--model", type=str, required=True, help="Specify the model to use for llm-embedding method")
    parser.add_argument("--n_values", nargs="+", type=int, help="Space-separated list of n-values to use")
    parser.add_argument("--labels", nargs="+", type=str, help="Space-separated list of labels to use")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--domain", required=True, choices=["story", "meeting"])
    parser.add_argument("--split", type=str, default="test")
    # ---------- two-round mode ----------
    parser.add_argument("--two_round", action="store_true",
                        help="If set, also run the two-round pipeline (vanilla run remains unchanged).")
    parser.add_argument("--r1_model", type=str, default="contriever",
                        help="Round-1 model name (default: contriever).")
    parser.add_argument("--r1_embeddings_path", type=str, default=None,
                        help="Path for Round-1 embeddings JSON (default: embeddings/{domain}/contriever_base.json).")
    parser.add_argument("--r1_depth", type=int, default=20, help="Top-N to retrieve in Round 1 (default: 20).")
    parser.add_argument("--r2_depth", type=int, default=20, help="Top-N to retrieve in Round 2 (default: 20).")
    parser.add_argument("--k_list", nargs="+", type=int, default=[1,2,3],
                        help="K values for R1->R2 augmentation (default: 1 2 3).")
    parser.add_argument("--final_cap", type=int, default=None,
                        help="Optional cap for final union list (None keeps all).")
    parser.add_argument("--snippet_tokens", type=int, default=256,
                        help="Max tokens per selected doc to stuff into augmented query (default: 256).")
    parser.add_argument("--verifier", choices=["oracle"], default="oracle",
                        help="Verifier type; 'oracle' only (keeps deps unchanged).")
    args = parser.parse_args()
    # Process arguments
    queries_path = os.path.abspath(args.queries_path)
    documents_path = os.path.abspath(args.documents_path)
    embeddings_path = args.embeddings_path
    domain = args.domain


    n = 8 if domain == "story" else 3
    if not embeddings_path.endswith(".json"):
        embeddings_path = f"{args.embeddings_path}.json"
    elif args.model not in supported_models:
        print("--model must be one of", supported_models)
        exit(1)
    model_name = args.model
    model = None
    if model_name in model_to_path:
        if model_name == "tuned-contriever":
            import torch
    
            model = SentenceTransformer("facebook/contriever-msmarco")
            
            checkpoint_path = f"{model_to_path[model_name]}/checkpoint.pth"
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'model' in checkpoint:
                full_model_state = checkpoint['model']
            else:
                full_model_state = checkpoint
 
            encoder_state = {}
            for key, value in full_model_state.items():
                if key.startswith("encoder."):
                    clean_key = key.replace("encoder.", "", 1)
                    encoder_state[clean_key] = value
            
            model[0].auto_model.load_state_dict(encoder_state, strict=False)
            model.max_seq_length = 512
            print(f"Loaded fine-tuned contriever from {checkpoint_path}")

        else:
            model = SentenceTransformer(
                model_to_path[model_name], 
                trust_remote_code=True
            )
            if model_name == "contriever":
                model.max_seq_length = 512
            else:
                model.max_seq_length = 4000
                model.tokenizer.padding_side="right"
    elif model_name == "gritlm":
        model = GritLM("GritLM/GritLM-7B", torch_dtype="auto")
    elif model_name == "promptriever":
        model = Promptriever("samaya-ai/promptriever-llama2-7b-v1")

    # Retrieval
    ir = InformationRetrieval(
        queries_path, 
        documents_path,
        embeddings_path
    )
    ir.run_evaluation(
        args.method,
        model_name, 
        model, 
        domain,
        [n], 
        ["MEAN"],
    )
    if not os.path.exists(f"{domain}/{args.model}") or args.overwrite:
        print("Creating", f"{domain}/{args.model}")
        with open(queries_path) as f:
            queries = json.load(f)
        with open("rankings.json") as f:
            rankings = json.load(f)
        data = []
        for qid in queries:
            entry = {}
            ans_docs = []
            query = queries[qid]["query"]
            entry["Query"] = query
            if domain == "story":
                for j in range(4):
                    entry[f"Summary_{j + 1}"] = queries[qid]["answer"][j]
            elif domain == "meeting":
                entry["Summary"] = queries[qid]["answer"]
            for id in rankings[query][:n]:
                with open(f"../../data/{domain}/documents/{id}.txt") as f:
                    ans_docs.append(f.read())
            entry["Ranking"] = rankings[query][:n]
            entry["Article"] = " <doc-sep> ".join(ans_docs)
            data.append(entry)
        os.makedirs(f"{domain}/{model_name}")
        with open(f"{domain}/{model_name}/{args.split}.json", "w") as f:
            json.dump(data, f, indent=4)

        # ===== Two-round (optional). Vanilla stays unchanged if --two_round is not provided. =====
    if args.two_round:
        # Round-1 model loader (default: base contriever). Keep simple to avoid changing your existing loaders.
        if args.r1_model == "contriever":
            r1_model = SentenceTransformer("facebook/contriever-msmarco")
            r1_model.max_seq_length = 512
        else:
            # Fallback: if they choose a different name that exists in your table
            if args.r1_model in model_to_path:
                r1_model = SentenceTransformer(model_to_path[args.r1_model], trust_remote_code=True)
                if args.r1_model == "contriever":
                    r1_model.max_seq_length = 512
                else:
                    r1_model.max_seq_length = 4000
                    r1_model.tokenizer.padding_side = "right"
            else:
                raise ValueError(f"Unsupported r1_model: {args.r1_model}")

        # Paths for embedding caches (keep separate per model)
        r1_emb_path = args.r1_embeddings_path or f"embeddings/{domain}/contriever_base.json"
        r2_emb_path = embeddings_path  # your tuned model embeddings path already computed/loaded above
        n_eval = 8 if domain == "story" else 3

        run_two_round_pipeline(
            ir=ir,
            domain=domain,
            split=args.split,
            base_model_name=args.r1_model,
            base_model=r1_model,
            base_embeddings_path=r1_emb_path,
            tuned_model_name=model_name,     # whatever you passed to vanilla (e.g., tuned-contriever)
            tuned_model=model,
            tuned_embeddings_path=r2_emb_path,
            n_eval=n_eval,
            K_list=tuple(args.k_list),
            final_cap=args.final_cap,
            max_tokens_per_doc=args.snippet_tokens,
            verifier_mode=args.verifier
        )