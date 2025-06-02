# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.utility_functions import run_for_wormhole_b0
from models.experimental.sentence_bert.tests.sentence_bert_e2e_performant import SentenceBERTrace2CQ
from models.experimental.sentence_bert.reference.sentence_bert import BertModel, custom_extended_mask
from models.experimental.sentence_bert.ttnn.ttnn_sentence_bert_model import TtnnSentenceBertModel
from models.experimental.sentence_bert.ttnn.common import custom_preprocessor, preprocess_inputs
from ttnn.model_preprocessing import preprocess_model_parameters
import transformers
import torch
import ttnn
import pytest
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from loguru import logger
from tqdm import tqdm
from itertools import islice, cycle


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def compute_ttnn_embeddings(sentence_bert_trace_2cq, sentences, model_name, device, batch_size=8):
    transformers_model = transformers.AutoModel.from_pretrained(model_name).eval()
    config = transformers.BertConfig.from_pretrained(model_name)
    logger.info("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    all_embeddings = []
    all_sentences = []
    num_batches = (len(sentences) + batch_size - 1) // batch_size
    ttnn_module = None
    for i in tqdm(range(0, len(sentences), batch_size), desc="Batches"):
        batch_sentences = sentences[i : i + batch_size]
        logger.info(f"Encoding batch {i//batch_size + 1} with {len(batch_sentences)} sentences...")
        # If batch is smaller than batch_size, repeat to fill batch_size
        orig_batch_size = len(batch_sentences)
        if orig_batch_size < batch_size:
            # Repeat sentences as needed to fill the batch, even if original batch is very small
            batch_sentences = list(islice(cycle(batch_sentences), batch_size))
        encoded_input = tokenizer(
            batch_sentences, padding="max_length", max_length=384, truncation=True, return_tensors="pt"
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        token_type_ids = encoded_input["token_type_ids"]
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)
        if ttnn_module is None:
            logger.info("Loading TTNN model...")
            reference_module = BertModel(config).to(torch.bfloat16)
            reference_module.load_state_dict(transformers_model.state_dict())
            parameters = preprocess_model_parameters(
                initialize_model=lambda: reference_module,
                custom_preprocessor=custom_preprocessor,
                device=device,
            )
            ttnn_module = TtnnSentenceBertModel(parameters=parameters, config=config)
        ttnn_input_ids, ttnn_token_type_ids, ttnn_position_ids, ttnn_attention_mask = preprocess_inputs(
            input_ids, token_type_ids, position_ids, extended_mask, device
        )
        ttnn_input_ids = ttnn.from_device(ttnn_input_ids)
        logger.info("Running inference on TTNN model for current batch...")
        output = sentence_bert_trace_2cq.execute_sentence_bert_trace_2cqs_inference(ttnn_input_ids)
        output_torch = ttnn.to_torch(output).squeeze(dim=1)
        embeddings = mean_pooling(output_torch, attention_mask)
        # ttnn_out = ttnn_module(
        #     ttnn_input_ids, ttnn_attention_mask, ttnn_token_type_ids, ttnn_position_ids, device=device
        # )
        # ttnn_out = ttnn.to_torch(ttnn_out[0]).squeeze(1)
        # embeddings = mean_pooling(ttnn_out, attention_mask)
        # Always slice to the original batch size (before padding)embeddings = embeddings[:orig_batch_size]
        all_embeddings.append(embeddings)
        all_sentences.extend(sentences[i : i + orig_batch_size])
    all_embeddings = torch.cat(all_embeddings, dim=0)
    logger.info("All embeddings computed.")
    return all_embeddings, all_sentences


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("batch_size, sequence_length", [(8, 384)])
def test_semantic_search_with_ttnn(device, use_program_cache, batch_size, sequence_length):
    model_name = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    kb_file = "knowledge_base.txt"
    logger.info(f"Loading knowledge base from {kb_file}...")
    kb_sentences = []
    filepath = os.path.join(os.path.dirname(__file__), kb_file)
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Remove inline comments (e.g., anything after # or //)
            for comment_token in ["#", "//"]:
                if comment_token in line:
                    line = line.split(comment_token, 1)[0].strip()
            # Skip empty lines
            if line:
                kb_sentences.append(line)
    logger.info(f"Loaded {len(kb_sentences)} sentences from knowledge base.")
    sentence_bert_trace_2cq = SentenceBERTrace2CQ()
    sentence_bert_trace_2cq.initialize_sentence_bert_trace_2cqs_inference(
        device, sequence_length=sequence_length, device_batch_size=batch_size, weight_dtype=ttnn.bfloat8_b
    )
    kb_embeddings, kb_sentences = compute_ttnn_embeddings(sentence_bert_trace_2cq, kb_sentences, model_name, device)
    logger.info(
        f"kb load {kb_embeddings.size(),kb_sentences}",
    )
    # Example query (in Turkish): "Siparişim ne zaman teslim edilir?"
    # English translation: "When will my order be delivered?"
    # This matches the knowledge base entry about order delivery times. i.e Siparişim ne zaman kargoya verilecek?
    run = 0
    while 1:
        logger.info(f"run no--{run}")
        run += 1
        logger.info("Ready for semantic search. Please enter your query:")
        query = input("Query: ").strip()
        query_embeddings, _ = compute_ttnn_embeddings(sentence_bert_trace_2cq, [query], model_name, device)
        logger.info(f"calculated shapes are {query_embeddings.shape,kb_embeddings.shape}")
        logger.info("Computing cosine similarities...")
        similarities = cosine_similarity(query_embeddings.detach().cpu().numpy(), kb_embeddings.detach().cpu().numpy())[
            0
        ]
        top_idx = np.argmax(similarities)
        logger.info(f"similarities and top_idx {similarities.shape},{top_idx}")
        logger.info(f"Best match: {kb_sentences[top_idx]}")
        logger.info(f"Similarity score: {similarities[top_idx]:.4f}")
        print(f"\tQuery: {query}")
        print(f"\tBest match: {kb_sentences[top_idx]}")
        print(f"\tSimilarity score: {similarities[top_idx]:.4f}")
    sentence_bert_trace_2cq.release_sentence_bert_trace_2cqs_inference()
