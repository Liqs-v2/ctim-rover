import enum
import json
import os
import re
from collections import deque
from functools import partial
from typing import List

import torch
from datasets import load_dataset, Dataset
from pymilvus import MilvusClient, DataType
from transformers import AutoTokenizer, T5EncoderModel

from app.data_structures import MessageThread


class TrajectoryType(enum.Enum):
    WRITE_PATCH = 'debug_agent_write_patch'
    LOCATE_BUG = 'conversation_round'

def fetch_embedding_model_and_tokenizer():
    model = T5EncoderModel.from_pretrained("Salesforce/codet5-base")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    return model, tokenizer


def fetch_swe_bench_verified() -> Dataset:
    return load_dataset("princeton-nlp/SWE-bench_Verified", split='test')

def fetch_swe_bench_lite() -> Dataset:
    return load_dataset("princeton-nlp/SWE-bench_Lite", split='test')

def _count_tokens(batch):
    return {'tokens_in_problem_statement': [len(sample) for sample in batch['input_ids']]}


def split_text_by_paragraphs(text):
    """
    Split text into paragraphs based on 2 consecutive newlines in an OS-agnostic way. The resulting list
    of paragraphs does not include the newlines (ie captured groups).
    """
    return re.split(r'(?:\r?\n){2}|(?:\r){2}', text)


def chunk_paragraphs(paragraphs, tokenizer, chunk_size):
    """
    Chunk paragraphs into segments of approximately chunk_size tokens.
    Combines short paragraphs and splits long paragraphs intelligently.
    """
    chunks = []
    current_chunk = deque([], maxlen=chunk_size + 2)  # Use chunk_size+2, because we will need the special tokens too
    current_length = 0

    for paragraph in paragraphs:
        paragraph_tokens = tokenizer.encode(paragraph, add_special_tokens=False)
        paragraph_length = len(paragraph_tokens)

        if current_length + paragraph_length <= chunk_size:
            # We can fit this paragraph into the current chunk -> Add paragraph to current chunk
            current_chunk.extend(paragraph_tokens)
            current_length += paragraph_length
        else:
            # We cannot fit this paragraph into the current chunk, finalize the current chunk before moving on
            if current_chunk:
                current_chunk.insert(0, tokenizer.bos_token_id)
                current_chunk.append(tokenizer.eos_token_id)
                chunks.append(current_chunk)

            if paragraph_length <= chunk_size:
                # The paragraph fits into an entire chunk. Initialize the new chunk to this paragraph.
                current_chunk = deque(paragraph_tokens, maxlen=chunk_size + 2)
                current_length = len(paragraph_tokens)
            else:
                # The paragraph is too large for an entire chunk, process explicitly until the remaining tokens fit into
                # a single chunk
                for i in range(0, paragraph_length, chunk_size):
                    if paragraph_length - i > chunk_size:
                        # If we are still over the chunk size, keep generating full chunks
                        current_chunk = deque(paragraph_tokens[i:i + chunk_size], maxlen=chunk_size + 2)
                        current_chunk.insert(0, tokenizer.bos_token_id)
                        current_chunk.append(tokenizer.eos_token_id)
                        chunks.append(current_chunk)
                    else:
                        # If we are below the chunk size, we simply initialize the current chunk and its length
                        # with the remainder.
                        current_length = (paragraph_length - i) % chunk_size
                        current_chunk = deque(paragraph_tokens[i + chunk_size:], maxlen=chunk_size + 2)

    if current_chunk:
        current_chunk.insert(0, tokenizer.bos_token_id)
        current_chunk.append(tokenizer.eos_token_id)
        chunks.append(current_chunk)

    return chunks


def process_chunks(chunks, tokenizer, model):
    """
    Process each chunk through the model to obtain embeddings.
    """
    embeddings = []

    # Note that the chunks are dequeues of already tokenized paragraphs with bos and eos token ids pre and appended
    # This allows for very low level control when chunking. The drawback is that batching becomes more difficult,
    # since I cant use the tokenizer to generate paddings and attention_masks for me. However, since I am chunking
    # within problem_statement and want to build one embedding per problem_statement I need to keep track of
    # which problem_statement a chunk came from. The easiest way to do this is to simply process sequentially.
    # Considering my dataset is only 500 samples, this should be fine.
    for chunk in chunks:
        with torch.no_grad():
            outputs = model(torch.tensor([list(chunk)]))
            # Mean pooling for embeddings
            chunk_embedding = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(chunk_embedding)
    return embeddings


def aggregate_embeddings(embeddings):
    """
    Aggregate the embeddings obtained for each chunk.
    """
    return torch.mean(torch.stack(embeddings), dim=0)


def generate_embedding_for_sample(sample, model, tokenizer):
    paragraphs = split_text_by_paragraphs(sample['problem_statement'])
    chunks = chunk_paragraphs(paragraphs, tokenizer, chunk_size=model.config.n_positions - 2)
    embeddings = process_chunks(chunks, tokenizer, model)
    embedding = aggregate_embeddings(embeddings)
    return {
        'embedding': embedding.flatten()
    }


def main():
    # 1. Load SWE-Bench from HuggingFace
    swe_bench_verified = fetch_swe_bench_verified()
    swe_bench_lite = fetch_swe_bench_lite()

    swe_bench_lite, swe_bench_verified = select(
        columns=['instance_id', 'problem_statement'],
        from_benchmarks=[swe_bench_lite, swe_bench_verified])

    # 3. Setup embedding model (CodeT5-base) from HF
    code_t5, code_t5_tokenizer = fetch_embedding_model_and_tokenizer()
    code_t5.eval()

    index_dimensions = code_t5.config.d_model

    # Set up vector database and create collections from schema
    client = MilvusClient('data/task_embeddings.db')
    schema = MilvusClient.create_schema(
        auto_id=False,
    )

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, description="primary id")
    schema.add_field(field_name="instance_id", datatype=DataType.VARCHAR, max_length=512,
                     description="instance id name")
    schema.add_field(field_name="patch_generation_trajectory", datatype=DataType.VARCHAR, max_length=65535,
                     description="write_patch agent trajectory")
    schema.add_field(field_name="locate_bug_trajectory", datatype=DataType.VARCHAR, max_length=65535,
                     description="locate bug agent trajectory")
    schema.add_field(field_name="problem_statement", datatype=DataType.VARCHAR, max_length=65535,
                     description="github issue")
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768,
                     description="problem statement embedding vector")

    # Create the collection
    client.create_collection(
        collection_name="swe_bench_lite",
        dimensions=index_dimensions,
        schema=schema
    )
    client.create_collection(
        collection_name="swe_bench_verified",
        dimension=index_dimensions,
        schema=schema
    )

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE",
        index_type="FLAT",
        index_name="vector_index",
    )

    # Create index for first collection
    client.create_index(
        collection_name="swe_bench_lite",
        index_params=index_params,
        sync=True
    )

    # Create index for second collection
    client.create_index(
        collection_name="swe_bench_verified",
        index_params=index_params,
        sync=True
    )


    # Reading Note: Count tokens in problem statements to check context limit violations of embedder context limit
    # swe_bench_verified = swe_bench_verified.map(lambda batch: code_t5_tokenizer(batch['problem_statement']),
    #                                             batched=True, batch_size=16, add_special_tokens=False)
    #
    # swe_bench_verified = swe_bench_verified.map(_count_tokens, batched=True, batch_size=16)

    # 4. Generate embeddings for problem statements (use HF datasets instead of df here).
    generate_embedding_for_sample_fn = partial(
        generate_embedding_for_sample,
        tokenizer=code_t5_tokenizer,
        model=code_t5)

    swe_bench_verified = setup_benchmark(generate_embedding_for_sample_fn, swe_bench_verified, swe_bench_verified.num_rows)

    # # 5.    i. Extract old trajectories of ACR in LLM format
    # #       ii. Populate df with trajectories
    resolved_instance_ids = None
    excluded_repos = ['astropy', 'xarray', 'pylint']
    with open('results/acr-val-only/new_eval_results/report.json') as f:
        resolved_instance_ids = json.load(f)
        resolved_instance_ids = resolved_instance_ids['resolved']
        resolved_instance_ids = [repo for repo in resolved_instance_ids if not \
            any(excluded_repo in repo for excluded_repo in excluded_repos)]

    # Process only resolved trajectories
    swe_bench_lite = swe_bench_lite.filter(lambda example: example['instance_id'] in resolved_instance_ids)
    swe_bench_lite = setup_benchmark(generate_embedding_for_sample_fn, swe_bench_lite, len(resolved_instance_ids))

    if not resolved_instance_ids:
        raise RuntimeError('Could not load report results, unable to initialize vector db with old trajectories from SWE-Bench Lite.')

    for dirpath, _, filenames in os.walk('results/acr-val-only/applicable_patch'):
        if  any(excluded_repo in dirpath for excluded_repo in excluded_repos) or not filenames:
            continue

        instance_id = re.match('^([^_]*__[^_]*)', dirpath.split('/')[-1]).group(0)

        if not instance_id in resolved_instance_ids:
            continue

        locate_bug_trajectory = extract_swe_bench_lite_trajectory_for(TrajectoryType.LOCATE_BUG, filenames, dirpath)
        swe_bench_lite = swe_bench_lite.map(lambda instance: {'locate_bug_trajectory': locate_bug_trajectory} \
            if instance['instance_id'] == instance_id else instance)

        patch_generation_trajectory = extract_swe_bench_lite_trajectory_for(TrajectoryType.WRITE_PATCH, filenames, dirpath)
        swe_bench_lite = swe_bench_lite.map(lambda instance: {'patch_generation_trajectory': patch_generation_trajectory} \
            if instance['instance_id'] == instance_id else instance)

    # Populate vector db with data
    client.insert(collection_name='swe_bench_verified', data=[dict(row) for row in swe_bench_verified])
    client.insert(collection_name='swe_bench_lite', data=[dict(row) for row in swe_bench_lite])

    # SQL style filtering
    # Reading Note: Seems like it only returns limit // 2 samples, so just double it to 1000 to consider all

    # Reading Note: Read index in again
    client.get(collection_name='swe_bench_verified', ids=[0,499])
    client.get(collection_name='swe_bench_lite', ids=[0, 52])

    # client.query(collection_name='task_embeddings', limit=10, filter='instance_id LIKE "django%"')
    # Alternatively use get for direct access via index


def select(columns: List[str], from_benchmarks: List):
    def _filter_columns(benchmark):
        return benchmark.remove_columns(
            [column for column in benchmark.column_names if column not in columns]
        )

    return tuple(map(_filter_columns, from_benchmarks))

def setup_benchmark(mapper, benchmark: Dataset, num_rows: int):
    benchmark = benchmark.map(mapper, batched=False)
    benchmark = benchmark.rename_column('embedding', 'vector')
    benchmark = benchmark.add_column('id', [i for i in range(num_rows)])
    benchmark = benchmark.add_column('patch_generation_trajectory', [''] * num_rows)
    benchmark = benchmark.add_column('locate_bug_trajectory', [''] * num_rows)
    return benchmark

def extract_swe_bench_lite_trajectory_for(trajectory_type: TrajectoryType, directory_files: List[str], current_dir_path: str):
    max_index = max(
        [int(match.group(1)) for file in directory_files
         if (match := re.search(rf'{trajectory_type.value}_(\d).json', file))]
    )
    msg_thread = MessageThread.load_from_file(os.path.join(current_dir_path, f'{trajectory_type.value}_{max_index}.json'))
    agent_trajectory = json.dumps(msg_thread.to_msg())

    return agent_trajectory

if __name__ == "__main__":
    main()
