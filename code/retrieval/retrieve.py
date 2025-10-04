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
    "tuned-contriever": "/scratch/dq2024/models/diverse_retriever/checkpoints/qampari_base_finetuned_only_random/checkpoint/best_model"
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
    "bm25", "contriever", "tuned-contriever"
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
                if model_name == "contriever":
                    # Contriever doesn't use instruction prefixes
                    return model.encode(
                        [text], 
                        batch_size=1, 
                        normalize_embeddings=True
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
                if model_name == "contriever":
                    # Contriever doesn't use instruction prefixes
                    return model.encode(
                        [text], 
                        batch_size=1, 
                        normalize_embeddings=True
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
        elif model_name == "contriever":
            chunk_embeddings = []
            for chunk in chunks:
                chunk_embeddings.append(
                    model.encode([chunk], batch_size=1, normalize_embeddings=True)[0]
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