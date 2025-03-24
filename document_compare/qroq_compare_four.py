import json
import re
import numpy as np
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os
import torch.nn.functional as F
from bert_score import score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, pipeline, RobertaForSequenceClassification
from . import utility

DEBERTA_MODEL = "microsoft/deberta-xlarge-mnli"
T5_MODEL = "prithivida/grammar_error_correcter_v1"
# T5_MODEL = "hassaanik/grammar-correction-model"
# T5_MODEL = "vennify/t5-base-grammar-correction"
# T5_MODEL = "facebook/bart-large-cnn"
# T5_MODEL = "google-t5/t5-small"
# T5_MODEL = "grammarly/coedit-large"
ROBERTA_MODEL = "textattack/roberta-base-CoLA"

# Define paths for saving models locally
DEBERTA_MODEL_PATH = "./local_models/deberta-xlarge-mnli"
T5_MODEL_PATH = "./local_models/grammar_error_correcter_v1"
# T5_MODEL_PATH = "./local_models/grammar-correction-model"
# T5_MODEL_PATH = "./local_models/t5-base-grammar-correction"
# T5_MODEL_PATH = "./local_models/bart-large-cnn"
# T5_MODEL_PATH = "./local_models/t5-small"
# T5_MODEL_PATH = "./local_models/coedit-large"
ROBERTA_MODEL_PATH = "./local_models/roberta-base-CoLA"

# # Load the tokenizer and model
# model_name = "microsoft/deberta-xlarge-mnli"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
#
# t5_tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
# t5_model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")

# DeBERTa for classification
if os.path.exists(DEBERTA_MODEL_PATH):
    print("Loading DeBERTa model from local storage...")
    tokenizer1 = AutoTokenizer.from_pretrained(DEBERTA_MODEL_PATH)
    model1 = AutoModelForSequenceClassification.from_pretrained(DEBERTA_MODEL_PATH)
else:
    print("Downloading DeBERTa model from Hugging Face...")
    tokenizer1 = AutoTokenizer.from_pretrained(DEBERTA_MODEL)
    model1 = AutoModelForSequenceClassification.from_pretrained(DEBERTA_MODEL)

    # Save model locally
    os.makedirs(DEBERTA_MODEL_PATH, exist_ok=True)
    tokenizer1.save_pretrained(DEBERTA_MODEL_PATH)
    model1.save_pretrained(DEBERTA_MODEL_PATH)

# T5 for correction
if os.path.exists(T5_MODEL_PATH):
    print("Loading T5 grammar correction model from local storage...")
    tokenizer2 = AutoTokenizer.from_pretrained(T5_MODEL_PATH)
    model2 = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_PATH)
    # model2 = AutoModelForSequenceClassification.from_pretrained(T5_MODEL_PATH)
else:
    print("Downloading T5 grammar correction model from Hugging Face...")
    tokenizer2 = AutoTokenizer.from_pretrained(T5_MODEL)
    model2 = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL)
    # model2 = AutoModelForSequenceClassification.from_pretrained(T5_MODEL)

    # Save model locally
    os.makedirs(T5_MODEL_PATH, exist_ok=True)
    tokenizer2.save_pretrained(T5_MODEL_PATH)
    model2.save_pretrained(T5_MODEL_PATH)

# Roberta for grammar detection
if os.path.exists(ROBERTA_MODEL_PATH):
    print("Loading ROBERTA grammar detection from local storage...")
    tokenizer3 = AutoTokenizer.from_pretrained(ROBERTA_MODEL_PATH)
    model3 = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_PATH)
else:
    print("Downloading ROBERTA grammar detection model from Hugging Face...")
    tokenizer3 = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
    model3 = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL)

    # Save model locally
    os.makedirs(ROBERTA_MODEL_PATH, exist_ok=True)
    tokenizer3.save_pretrained(ROBERTA_MODEL_PATH)
    model3.save_pretrained(ROBERTA_MODEL_PATH)


class TextDifference(BaseModel):
    change_type: str = Field(..., description="Type of change: 'matched', 'added', 'removed', or 'replaced'.")
    original_section_id: Optional[str] = Field(None, description="Section ID from the original text.")
    modified_section_id: Optional[str] = Field(None, description="Section ID from the modified text.")


def compare_with_groq1(file1_path, file2_path):
    def get_verify_matches(semantic_matches, original_text, modified_text, groq_llm, batch_size=3):
        """
        Verify semantic matches using LLM and return standardized output format.
        """
        comparison_prompt = ChatPromptTemplate.from_template(
            """
            Compare these potentially matching sections and verify if they are truly matches.
            For each comparison, determine if the sections are:
            - matched (same content or only minor formatting differences)
            - removed (content exists in original but not in modified)
            - added (content exists in modified but not in original)

            Original sections:
            {original_sections}

            Modified sections:
            {modified_sections}

            Return ONLY a JSON array of objects with this exact structure:
            [
                {{"change_type": "matched", "original_section_id": "org_id", "modified_section_id": "mod_id"}},
                {{"change_type": "removed", "original_section_id": "org_id", "modified_section_id": null}},
                {{"change_type": "added", "original_section_id": null, "modified_section_id": "mod_id"}}
            ]
            """
        )

        verified_matches = []

        for i in range(0, len(semantic_matches), batch_size):
            batch = semantic_matches[i:i + batch_size]

            # Prepare sections for comparison
            original_sections = {}
            modified_sections = {}

            for match in batch:
                if match["original_section_id"]:
                    for item in original_text:
                        if match["original_section_id"] in item:
                            original_sections[match["original_section_id"]] = item[match["original_section_id"]]
                            break

                if match["modified_section_id"]:
                    for item in modified_text:
                        if match["modified_section_id"] in item:
                            modified_sections[match["modified_section_id"]] = item[match["modified_section_id"]]
                            break

            # Get LLM verification
            input_text = comparison_prompt.format(
                original_sections=json.dumps(original_sections, indent=2),
                modified_sections=json.dumps(modified_sections, indent=2)
            )

            response = groq_llm.invoke(input_text)
            batch_results = extract_valid_json(response.content)

            if batch_results:
                # Validate and standardize each result
                for result in batch_results:
                    standardized_result = {
                        "change_type": result["change_type"],
                        "original_section_id": result["original_section_id"] if result.get("original_section_id") else None,
                        "modified_section_id": result["modified_section_id"] if result.get("modified_section_id") else None
                    }
                    verified_matches.append(standardized_result)

        return verified_matches

    def find_matching_sections1(original_texts, modified_texts):
        """Finds matching sections using embedding similarity."""
        # Convert input lists of dicts to separate dicts
        original_dict = {}
        modified_dict = {}

        for item in original_texts:
            original_dict.update(item)
        for item in modified_texts:
            modified_dict.update(item)

        transformer_model = SentenceTransformer("all-MiniLM-L6-v2")

        original_keys, original_values = list(original_dict.keys()), list(original_dict.values())
        modified_keys, modified_values = list(modified_dict.keys()), list(modified_dict.values())

        original_embeddings = transformer_model.encode(original_values)
        modified_embeddings = transformer_model.encode(modified_values)

        matches = []
        used_modified_indices = set()

        # Find matches using cosine similarity
        for i, org_embedding in enumerate(original_embeddings):
            similarities = np.dot(modified_embeddings, org_embedding) / (
                    np.linalg.norm(modified_embeddings, axis=1) * np.linalg.norm(org_embedding)
            )

            # Sort similarities to find best matches
            sorted_indices = np.argsort(similarities)[::-1]

            # Find the best unused match
            for idx in sorted_indices:
                if idx not in used_modified_indices and similarities[idx] > 0.9:
                    matches.append({
                        "change_type": "matched",
                        "original_section_id": original_keys[i],
                        "modified_section_id": modified_keys[idx]
                    })
                    used_modified_indices.add(idx)
                    break

        # Find removed sections (in original but not matched)
        matched_original = {match["original_section_id"] for match in matches}
        for i, key in enumerate(original_keys):
            if key not in matched_original:
                matches.append({
                    "change_type": "removed",
                    "original_section_id": key,
                    "modified_section_id": None
                })

        # Find added sections (in modified but not matched)
        matched_modified = {match["modified_section_id"] for match in matches}
        for i, key in enumerate(modified_keys):
            if key not in matched_modified:
                matches.append({
                    "change_type": "added",
                    "original_section_id": None,
                    "modified_section_id": key
                })

        return matches

    def find_matching_sections2(original_texts, modified_texts, similarity_threshold=0.85):
        """
        Finds matching sections using embedding similarity with improved accuracy.
        """
        # Convert input lists of dicts to separate dicts
        original_dict = {}
        modified_dict = {}

        for item in original_texts:
            original_dict.update(item)
        for item in modified_texts:
            modified_dict.update(item)

        transformer_model = SentenceTransformer("all-MiniLM-L6-v2")

        original_keys, original_values = list(original_dict.keys()), list(original_dict.values())
        modified_keys, modified_values = list(modified_dict.keys()), list(modified_dict.values())

        matches = []
        used_modified_indices = set()

        # First, check for exact content matches
        for i, original_value in enumerate(original_values):
            for j, modified_value in enumerate(modified_values):
                if j not in used_modified_indices and original_value == modified_value:
                    matches.append({
                        "change_type": "matched",
                        "original_section_id": original_keys[i],
                        "modified_section_id": modified_keys[j]
                    })
                    used_modified_indices.add(j)
                    break

        # Then check for semantic similarity for remaining sections
        original_embeddings = transformer_model.encode(original_values)
        modified_embeddings = transformer_model.encode(modified_values)

        for i, org_embedding in enumerate(original_embeddings):
            if any(match["original_section_id"] == original_keys[i] for match in matches):
                continue

            similarities = np.dot(modified_embeddings, org_embedding) / (
                    np.linalg.norm(modified_embeddings, axis=1) * np.linalg.norm(org_embedding)
            )

            # Find best match above threshold
            max_sim_idx = np.argmax(similarities)
            if similarities[max_sim_idx] > similarity_threshold and max_sim_idx not in used_modified_indices:
                matches.append({
                    "change_type": "matched",
                    "original_section_id": original_keys[i],
                    "modified_section_id": modified_keys[max_sim_idx]
                })
                used_modified_indices.add(max_sim_idx)

        # Add removed sections
        matched_original = {match["original_section_id"] for match in matches}
        for i, key in enumerate(original_keys):
            if key not in matched_original:
                matches.append({
                    "change_type": "removed",
                    "original_section_id": key,
                    "modified_section_id": None
                })

        # Add added sections
        matched_modified = {match["modified_section_id"] for match in matches}
        for i, key in enumerate(modified_keys):
            if key not in matched_modified:
                matches.append({
                    "change_type": "added",
                    "original_section_id": None,
                    "modified_section_id": key
                })

        return matches

    def find_matching_sections3(original_texts, modified_texts, similarity_threshold=0.85):
        """Finds matching sections using both content equality and embedding similarity."""
        # Convert input lists of dicts to separate dicts
        original_dict = {}
        modified_dict = {}

        for item in original_texts:
            original_dict.update(item)
        for item in modified_texts:
            modified_dict.update(item)

        transformer_model = SentenceTransformer("all-MiniLM-L6-v2")

        original_keys = list(original_dict.keys())
        modified_keys = list(modified_dict.keys())
        original_values = list(original_dict.values())
        modified_values = list(modified_dict.values())

        matches = []
        used_modified_indices = set()
        used_original_indices = set()

        # First pass: Find exact content matches regardless of section ID
        for i, orig_content in enumerate(original_values):
            if i in used_original_indices:
                continue

            for j, mod_content in enumerate(modified_values):
                if j in used_modified_indices:
                    continue

                # Check for exact content match first
                if orig_content.strip() == mod_content.strip():
                    matches.append({
                        "change_type": "matched",
                        "original_section_id": original_keys[i],
                        "modified_section_id": modified_keys[j]
                    })
                    used_original_indices.add(i)
                    used_modified_indices.add(j)
                    break

        # Second pass: Use embedding similarity for remaining sections
        if len(used_original_indices) < len(original_values):
            # Get embeddings only for unmatched sections
            remaining_original_indices = [i for i in range(len(original_values)) if i not in used_original_indices]
            remaining_modified_indices = [i for i in range(len(modified_values)) if i not in used_modified_indices]

            remaining_original_values = [original_values[i] for i in remaining_original_indices]
            remaining_modified_values = [modified_values[i] for i in remaining_modified_indices]

            original_embeddings = transformer_model.encode(remaining_original_values)
            modified_embeddings = transformer_model.encode(remaining_modified_values)

            for idx1, i in enumerate(remaining_original_indices):
                best_similarity = -1
                best_match_idx = -1

                for idx2, j in enumerate(remaining_modified_indices):
                    if j not in used_modified_indices:
                        similarity = np.dot(original_embeddings[idx1], modified_embeddings[idx2]) / (
                                np.linalg.norm(original_embeddings[idx1]) * np.linalg.norm(modified_embeddings[idx2])
                        )

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_idx = j

                if best_similarity > similarity_threshold:
                    matches.append({
                        "change_type": "matched",
                        "original_section_id": original_keys[i],
                        "modified_section_id": modified_keys[best_match_idx]
                    })
                    used_original_indices.add(i)
                    used_modified_indices.add(best_match_idx)

        # Add removed sections
        for i in range(len(original_values)):
            if i not in used_original_indices:
                matches.append({
                    "change_type": "removed",
                    "original_section_id": original_keys[i],
                    "modified_section_id": None
                })

        # Add added sections
        for i in range(len(modified_values)):
            if i not in used_modified_indices:
                matches.append({
                    "change_type": "added",
                    "original_section_id": None,
                    "modified_section_id": modified_keys[i]
                })

        return matches

    def find_matching_sections4(original_texts, modified_texts, similarity_threshold=0.85):
        """Finds matching sections using both content equality and embedding similarity."""
        # Convert input lists of dicts to separate dicts
        original_dict = {}
        modified_dict = {}

        for item in original_texts:
            original_dict.update(item)
        for item in modified_texts:
            modified_dict.update(item)

        transformer_model = SentenceTransformer("all-MiniLM-L6-v2")

        original_keys = list(original_dict.keys())
        modified_keys = list(modified_dict.keys())
        original_values = list(original_dict.values())
        modified_values = list(modified_dict.values())

        matches = []
        used_modified_indices = set()
        used_original_indices = set()

        # First pass: Find exact content matches regardless of section ID
        for i, orig_content in enumerate(original_values):
            if i in used_original_indices:
                continue

            for j, mod_content in enumerate(modified_values):
                if j in used_modified_indices:
                    continue

                # Check for exact content match first
                if orig_content.strip() == mod_content.strip():
                    matches.append({
                        "change_type": "matched",
                        "original_section_id": original_keys[i],
                        "modified_section_id": modified_keys[j]
                    })
                    used_original_indices.add(i)
                    used_modified_indices.add(j)
                    break

        # Second pass: Use embedding similarity for remaining sections
        if len(used_original_indices) < len(original_values):
            # Get embeddings only for unmatched sections
            remaining_original_indices = [i for i in range(len(original_values)) if i not in used_original_indices]
            remaining_modified_indices = [i for i in range(len(modified_values)) if i not in used_modified_indices]

            remaining_original_values = [original_values[i] for i in remaining_original_indices]
            remaining_modified_values = [modified_values[i] for i in remaining_modified_indices]

            original_embeddings = transformer_model.encode(remaining_original_values)
            modified_embeddings = transformer_model.encode(remaining_modified_values)

            for idx1, i in enumerate(remaining_original_indices):
                best_similarity = -1
                best_match_idx = -1

                for idx2, j in enumerate(remaining_modified_indices):
                    if j not in used_modified_indices:
                        similarity = np.dot(original_embeddings[idx1], modified_embeddings[idx2]) / (
                                np.linalg.norm(original_embeddings[idx1]) * np.linalg.norm(modified_embeddings[idx2])
                        )

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_idx = j

                if best_similarity > similarity_threshold:
                    matches.append({
                        "change_type": "matched",
                        "original_section_id": original_keys[i],
                        "modified_section_id": modified_keys[best_match_idx]
                    })
                    used_original_indices.add(i)
                    used_modified_indices.add(best_match_idx)

        # Add removed sections
        for i in range(len(original_values)):
            if i not in used_original_indices:
                matches.append({
                    "change_type": "removed",
                    "original_section_id": original_keys[i],
                    "modified_section_id": None
                })

        # Add added sections
        for i in range(len(modified_values)):
            if i not in used_modified_indices:
                matches.append({
                    "change_type": "added",
                    "original_section_id": None,
                    "modified_section_id": modified_keys[i]
                })

        return matches

    def find_matching_sections(original_texts, modified_texts, similarity_threshold=0.85):
        """Finds matching sections using both content equality and embedding similarity."""
        # Convert input lists of dicts to separate dicts
        original_dict = {}
        modified_dict = {}

        for item in original_texts:
            original_dict.update(item)
        for item in modified_texts:
            modified_dict.update(item)

        transformer_model = SentenceTransformer("all-MiniLM-L6-v2")

        original_keys = list(original_dict.keys())
        modified_keys = list(modified_dict.keys())
        original_values = list(original_dict.values())
        modified_values = list(modified_dict.values())

        matches = []
        used_modified_indices = set()
        used_original_indices = set()

        # First pass: Find exact content matches regardless of section ID
        for i, orig_content in enumerate(original_values):
            if i in used_original_indices:
                continue

            for j, mod_content in enumerate(modified_values):
                if j in used_modified_indices:
                    continue

                # Check for exact content match first
                if orig_content.strip() == mod_content.strip():
                    matches.append({
                        "change_type": "matched",
                        "original_section_id": original_keys[i],
                        "modified_section_id": modified_keys[j]
                    })
                    used_original_indices.add(i)
                    used_modified_indices.add(j)
                    break

        # Second pass: Use embedding similarity for remaining sections
        if len(used_original_indices) < len(original_values):
            # Get embeddings only for unmatched sections
            remaining_original_indices = [i for i in range(len(original_values)) if i not in used_original_indices]
            remaining_modified_indices = [i for i in range(len(modified_values)) if i not in used_modified_indices]

            remaining_original_values = [original_values[i] for i in remaining_original_indices]
            remaining_modified_values = [modified_values[i] for i in remaining_modified_indices]

            original_embeddings = transformer_model.encode(remaining_original_values)
            modified_embeddings = transformer_model.encode(remaining_modified_values)

            for idx1, i in enumerate(remaining_original_indices):
                best_similarity = -1
                best_match_idx = -1

                for idx2, j in enumerate(remaining_modified_indices):
                    if j not in used_modified_indices:
                        similarity = np.dot(original_embeddings[idx1], modified_embeddings[idx2]) / (
                                np.linalg.norm(original_embeddings[idx1]) * np.linalg.norm(modified_embeddings[idx2])
                        )

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_idx = j

                if best_similarity > similarity_threshold:
                    matches.append({
                        "change_type": "matched",
                        "original_section_id": original_keys[i],
                        "modified_section_id": modified_keys[best_match_idx]
                    })
                    used_original_indices.add(i)
                    used_modified_indices.add(best_match_idx)

        # Add removed sections
        for i in range(len(original_values)):
            if i not in used_original_indices:
                matches.append({
                    "change_type": "removed",
                    "original_section_id": original_keys[i],
                    "modified_section_id": None
                })

        # Add added sections
        for i in range(len(modified_values)):
            if i not in used_modified_indices:
                matches.append({
                    "change_type": "added",
                    "original_section_id": None,
                    "modified_section_id": modified_keys[i]
                })

        # Now, manually ensure identical content sections are matched, even with different IDs
        for i, orig_content in enumerate(original_values):
            for j, mod_content in enumerate(modified_values):
                if orig_content.strip() == mod_content.strip() and original_keys[i] != modified_keys[j]:
                    # They have identical content, even though IDs differ
                    matches.append({
                        "change_type": "matched",
                        "original_section_id": original_keys[i],
                        "modified_section_id": modified_keys[j]
                    })

        return matches

    def semantic_similarity_search1(original_text, modified_text, threshold=0.85):
        """
        Perform similarity search between original and modified sections using embeddings.

        Parameters:
            original_text (list): List of dictionaries with original sections.
            modified_text (list): List of dictionaries with modified sections.
            threshold (float): Similarity threshold to consider a match.

        Returns:
            list: List of matched sections with change type information.
        """
        # Load a pre-trained Sentence-BERT model for embedding
        # model = SentenceTransformer("all-MiniLM-L6-v2")
        model = SentenceTransformer("all-mpnet-base-v2")

        # Extract text from original and modified sections
        original_sections = [list(section.values())[0] for section in original_text]
        modified_sections = [list(section.values())[0] for section in modified_text]

        # Generate embeddings for original and modified sections
        original_embeddings = model.encode(original_sections)
        modified_embeddings = model.encode(modified_sections)

        # Initialize the list for storing matches
        matches = []
        used_original_indices = set()
        used_modified_indices = set()

        # Perform similarity search (cosine similarity)
        for i, original_embed in enumerate(original_embeddings):
            best_match_score = -1
            best_match_idx = -1

            for j, modified_embed in enumerate(modified_embeddings):
                if j in used_modified_indices:
                    continue

                # Compute cosine similarity between original and modified section
                sim_score = cosine_similarity([original_embed], [modified_embed])[0][0]

                if sim_score > best_match_score:
                    best_match_score = sim_score
                    best_match_idx = j

            # If the best similarity score exceeds the threshold, it's a match
            if best_match_score > threshold:
                matches.append({
                    "change_type": "matched",
                    "original_section_id": f"org_section_{i}",
                    "modified_section_id": f"mod_section_{best_match_idx}",
                    "similarity_score": best_match_score
                })
                used_original_indices.add(i)
                used_modified_indices.add(best_match_idx)

        # Add remaining unmatched sections as 'removed' or 'added'
        for i in range(len(original_sections)):
            if i not in used_original_indices:
                matches.append({
                    "change_type": "removed",
                    "original_section_id": f"org_section_{i}",
                    "modified_section_id": None
                })

        for j in range(len(modified_sections)):
            if j not in used_modified_indices:
                matches.append({
                    "change_type": "added",
                    "original_section_id": None,
                    "modified_section_id": f"mod_section_{j}"
                })

        return matches

    def semantic_similarity_search2(original_text, modified_text, threshold=0.85):
        from sentence_transformers import SentenceTransformer, util

        # Load a better model for similarity search
        model = SentenceTransformer("all-mpnet-base-v2")  # More accurate than MiniLM

        # Extract section IDs and text content
        original_sections = {list(section.keys())[0]: list(section.values())[0] for section in original_text}
        modified_sections = {list(section.keys())[0]: list(section.values())[0] for section in modified_text}

        # Compute embeddings for all sections
        original_embeddings = model.encode(list(original_sections.values()), convert_to_tensor=True)
        modified_embeddings = model.encode(list(modified_sections.values()), convert_to_tensor=True)

        # Compute cosine similarity
        similarity_scores = util.cos_sim(original_embeddings, modified_embeddings)

        # Define threshold for considering two sections as "matched"
        SIMILARITY_THRESHOLD = 0.9  # Adjust based on performance

        # Store verified matches
        verified_matches = []
        matched_original = set()
        matched_modified = set()

        # Identify best matches
        for i, org_id in enumerate(original_sections.keys()):
            for j, mod_id in enumerate(modified_sections.keys()):
                if similarity_scores[i][j] > SIMILARITY_THRESHOLD:
                    verified_matches.append({
                        "change_type": "matched",
                        "original_section_id": org_id,
                        "modified_section_id": mod_id
                    })
                    matched_original.add(org_id)
                    matched_modified.add(mod_id)

        # Mark remaining sections as added or removed
        for org_id in original_sections.keys():
            if org_id not in matched_original:
                verified_matches.append({
                    "change_type": "removed",
                    "original_section_id": org_id,
                    "modified_section_id": None
                })

        for mod_id in modified_sections.keys():
            if mod_id not in matched_modified:
                verified_matches.append({
                    "change_type": "added",
                    "original_section_id": None,
                    "modified_section_id": mod_id
                })

        return verified_matches

    def semantic_similarity_search(original_text, modified_text, threshold=0.85):
        from sentence_transformers import SentenceTransformer, util

        # Load a better model for similarity search
        model = SentenceTransformer("all-mpnet-base-v2")

        # Extract section IDs and text content
        original_sections = {list(section.keys())[0]: list(section.values())[0] for section in original_text}
        modified_sections = {list(section.keys())[0]: list(section.values())[0] for section in modified_text}

        # Compute embeddings for all sections
        original_embeddings = model.encode(list(original_sections.values()), convert_to_tensor=True)
        modified_embeddings = model.encode(list(modified_sections.values()), convert_to_tensor=True)

        # Compute cosine similarity
        similarity_scores = util.cos_sim(original_embeddings, modified_embeddings)

        # Define threshold for considering two sections as "matched"
        SIMILARITY_THRESHOLD = 0.9

        # Store verified matches
        verified_matches = []
        matched_original = set()
        matched_modified = set()

        # Identify best matches
        for i, org_id in enumerate(original_sections.keys()):
            best_match = None
            best_score = SIMILARITY_THRESHOLD  # Only consider matches above threshold

            for j, mod_id in enumerate(modified_sections.keys()):
                if similarity_scores[i][j] > best_score:
                    best_score = similarity_scores[i][j]
                    best_match = mod_id

            if best_match:
                verified_matches.append({
                    "change_type": "matched",
                    "original_section_id": org_id,
                    "modified_section_id": best_match
                })
                matched_original.add(org_id)
                matched_modified.add(best_match)

        # Mark remaining sections as added or removed
        for org_id in original_sections.keys():
            if org_id not in matched_original:
                verified_matches.append({
                    "change_type": "removed",
                    "original_section_id": org_id,
                    "modified_section_id": None
                })

        for mod_id in modified_sections.keys():
            if mod_id not in matched_modified:
                verified_matches.append({
                    "change_type": "added",
                    "original_section_id": None,
                    "modified_section_id": mod_id
                })

        return verified_matches

    def extract_valid_json(text):
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None

    try:
        # Your existing file loading code here...
        original_text = [
            {
                "org_section_0": "Business Requirements Document"
            },
            {
                "org_section_1": "I. Introduction This Business Requirements Document (BRD) serves as a foundational reference for all stakeholders, ensuring alignment in project goals and guiding the development of solutions to meet business requirements. The goal is to improve customer management processes, enhance data tracking, and streamline workflows across departments."
            },
            {
                "org_section_3": "II. Project Overview The CRM System Upgrade aims to modernize the organization\u2019s customer management processes by integrating new features and improving system performance. The project will address business needs such as enhanced customer data tracking, improved reporting capabilities, and more efficient workflows for the sales and customer support teams. This initiative will help boost customer satisfaction, drive sales, and optimize internal operations. Prepared by: [YOUR NAME] Email: [YOUR EMAIL] Company Name: [YOUR COMPANY NAME] Company Address: [YOUR COMPANY ADDRESS]"
            },
            {
                "org_section_10": "III. Business Objectives The primary objectives of the CRM System Upgrade are as follows: Objective 1: Improve customer data tracking and reporting to enhance sales strategies and decision-making. Objective 2: Automate sales processes to increase efficiency and reduce manual errors. Objective 3: Enhance customer support capabilities to ensure faster resolution of queries and issues. These objectives will be achieved by implementing a state-of-the-art CRM system that integrates seamlessly with existing tools and supports new features such as predictive analytic and automated reminders."
            },
            {
                "org_section_16": "IV. Scope of the Project The project will include the following activities: Activity 1: CRM system upgrade and configuration to meet business needs. Activity 2: Data migration from the legacy CRM to the new platform, ensuring no loss of customer information. Activity 3: User training and adoption to ensure smooth transition to the new system. Exclusions: Development of any features not related to CRM functionality (e.g., non-sales tools). Integration with systems not already in use (e.g., new marketing platforms outside of the CRM). The project will be executed within the established timeline and constraints."
            },
            {
                "org_section_25": "V. Stakeholder Requirements"
            },
            {
                "org_section_26": "Stakeholder Requirement/Need Priority Deadline Sales Manager Ability to track customer interactions and sales forecasts High January 15, 2050 Customer Support Improved ticket resolution capabilities Medium February 22, 2050 Marketing Director Enhanced lead tracking and campaign performance Low March 10, 2050"
            },
            {
                "org_section_27": "VI. Functional Requirements The following functional requirements have been identified for the project: Requirement 1: The CRM system must integrate with existing email and marketing platforms to track leads and customer interactions. Requirement 2: The system must allow for custom reporting based on customer demographics, sales activity, and support history. Requirement 3: The CRM should include automated task reminders for sales follow-ups, customer service queries, and appointments."
            },
            {
                "org_section_32": "VII. Project Timeline"
            },
            {
                "org_section_33": "Task/Phase Start Date End Date Responsible Party Initial Planning January 5, 2050 January 15, 2050 Project Manager Requirement Gathering January 16, 2050 February 15, 2050 Business Analyst Design Phase February 16, 2050 March 5, 2050 System Architect Implementation March 6, 2050 May 30, 2050 Development Team Go Live June 1, 2050 June 1, 2050 Project Manager"
            },
            {
                "org_section_34": "VIII. Budget and Resources"
            },
            {
                "org_section_35": "Category Estimated Cost Notes Personnel Costs $200,000 Developers, project manager, BA Technology/Tools $150,000 CRM software, integration tools Miscellaneous $50,000 Training materials, contingency"
            },
            {
                "org_section_36": "The resources required include CRM software licenses, data migration tools, development environments, and a dedicated team of 10 people for the duration of the project."
            },
            {
                "org_section_39": "IX. Conclusion The successful implementation of the CRM System Upgrade will provide significant benefits to the organization, including improved sales efficiency, enhanced customer tracking, and a better user experience for both customers and internal teams. By aligning all stakeholders on the requirements and timelines outlined in this document, the project will proceed smoothly, ensuring that all objectives are met on time and within budget."
            }
        ]
        modified_text = [
            {
                "mod_section_0": "Business Requirements Document"
            },
            {
                "mod_section_1": "I. Introduction This Business Requirements Document (BRD) serves as a foundational reference for all stakeholders, ensuring alignment in project goals and guiding the development of solutions to meet business requirements. The goal is to improve customer management processes, enhance data tracking, and streamline workflows across departments."
            },
            {
                "mod_section_3": "II. Project Overview The CRM System Upgrade aims to modernize the organization\u2019s customer management processes by integrating new features and improving system performance. The project will address business needs such as enhanced customer data tracking, improved reporting capabilities, and more efficient workflows for the sales and customer support teams. This initiative will help boost customer satisfaction, drive sales, and optimize internal operations. Prepared by: [YOUR NAME] Email: [YOUR EMAIL] Company Name: [YOUR COMPANY NAME] Company Address: [YOUR COMPANY ADDRESS]"
            },
            {
                "mod_section_10": "IV. Scope of the Project The project will include the following activities: Activity 1: CRM system upgrade and configuration to meet business needs. Activity 2: Data migration from the legacy CRM to the new platform, ensuring no loss of customer information. Activity 3: User training and adoption to ensure smooth transition to the new system. Exclusions: Development of any features not related to CRM functionality (e.g., non-sales tools). Integration with systems not already in use (e.g., new marketing platforms outside of the CRM). The project will be executed within the established timeline and constraints."
            },
            {
                "mod_section_20": "III. Business Objectives The primary objectives of the CRM System Upgrade are as follows: Objective 1: Improve customer data tracking and reporting to enhance sales strategies and decision-making. Objective 2: Automate sales processes to increase efficiency and reduce manual errors. Objective 3: Enhance customer support capabilities to ensure faster resolution of queries and issues. These objectives will be achieved by implementing a state-of-the-art CRM system that integrates seamlessly with existing tools and supports new features such as predictive analytic and automated reminders."
            },
            {
                "mod_section_26": "V. Stakeholder Requirements"
            },
            {
                "mod_section_27": "Stakeholder Requirement/Need Priority Deadline Sales Manager Ability to track customer interactions and sales forecasts High January 15, 2050 Customer Support Improved ticket resolution capabilities Medium February 22, 2050 Marketing Director Enhanced lead tracking and campaign performance Low March 10, 2050"
            },
            {
                "mod_section_28": "VI. Functional Requirements The following functional requirements have been identified for the project: Requirement 1: The CRM system must integrate with existing email and marketing platforms to track leads and customer interactions. Requirement 2: The system must allow for custom reporting based on customer demographics, sales activity, and support history. Requirement 3: The CRM should include automated task reminders for sales follow-ups, customer service queries, and appointments."
            },
            {
                "mod_section_33": "VII. Project Timeline"
            },
            {
                "mod_section_34": "Task/Phase Start Date End Date Responsible Party Initial Planning January 5, 2050 January 15, 2050 Project Manager Requirement Gathering January 16, 2050 February 15, 2050 Business Analyst Design Phase February 16, 2050 March 5, 2050 System Architect Implementation March 6, 2050 May 30, 2050 Development Team Go Live June 1, 2050 June 1, 2050 Project Manager"
            },
            {
                "mod_section_35": "VIII. Budget and Resources"
            },
            {
                "mod_section_36": "Category Estimated Cost Notes Personnel Costs $200,000 Developers, project manager, BA Technology/Tools $150,000 CRM software, integration tools"
            },
            {
                "mod_section_40": "IX. Risk Assessment"
            },
            {
                "mod_section_41": "Risk Factor Impact Mitigation Strategy Data Loss High Implement rigorous backup and testing procedures User Resistance Medium Conduct thorough training and provide user support Integration Issues High Perform extensive compatibility testing Security Breaches High Apply encryption, access control, and security audits Budget Overruns Medium Monitor project spending and reallocate resources as needed"
            },
            {
                "mod_section_43": "IX. Conclusion The successful implementation of the CRM System Upgrade will provide significant benefits to the organization, including improved sales efficiency, enhanced customer tracking, and a better user experience for both customers and internal teams. By aligning all stakeholders on the requirements and timelines outlined in this document, the project will proceed smoothly, ensuring that all objectives are met on time and within budget."
            }
        ]

        # First, use semantic matching to find similar sections
        # semantic_matches = find_matching_sections(original_text, modified_text)
        semantic_matches = semantic_similarity_search(original_text, modified_text)

        # Use LLM for detailed comparison of matched sections
        groq_api_key = "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"
        groq_llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key, temperature=0.0)

        comparison_prompt = ChatPromptTemplate.from_template(
            """
            Compare these potentially matching sections and verify if they are truly matches,
            considering both semantic meaning and exact content.

            Original sections:
            {original_sections}

            Modified sections:
            {modified_sections}

            Return a JSON array of verified matches, with any content differences noted.
            """
        )

        """
        # Process matches in batches to avoid token limits
        verified_matches = []
        batch_size = 3  # Adjust based on token limits
        for i in range(0, len(semantic_matches), batch_size):
            batch = semantic_matches[i:i + batch_size]

            # Prepare sections for comparison
            original_sections = {match["original_section_id"]: next((item[match["original_section_id"]]
                                                                     for item in original_text
                                                                     if match["original_section_id"] in item), None)
                                 for match in batch if match["original_section_id"]}

            modified_sections = {match["modified_section_id"]: next((item[match["modified_section_id"]]
                                                                     for item in modified_text
                                                                     if match["modified_section_id"] in item), None)
                                 for match in batch if match["modified_section_id"]}
            print('original_sections---', original_sections)
            print('modified_sections---', modified_sections)
            # Get LLM verification
            input_text = comparison_prompt.format(
                original_sections=json.dumps(original_sections, indent=2),
                modified_sections=json.dumps(modified_sections, indent=2)
            )

            response = groq_llm.invoke(input_text)
            verified_batch = extract_valid_json(response.content)

            if verified_batch:
                verified_matches.extend(verified_batch)
        print('verified_matches---', verified_matches)
        """

        # Verify matches and get standardized output
        verified_matches = get_verify_matches(
            semantic_matches=semantic_matches,
            original_text=original_text,
            modified_text=modified_text,
            groq_llm=groq_llm
        )
        print('verified_matches---', verified_matches)
        return verified_matches

    except Exception as e:
        import traceback; traceback.print_exc()
        print("Error:", str(e))
        return {"error": str(e)}

# def compare_with_groq(original_text, modified_text, threshold=0.90, replace_threshold=0.80):
    """
    Compare sections from original_text with modified_text, classify them as matched, replaced, removed, or added.
    """

    # model = SentenceTransformer("all-mpnet-base-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Extract section IDs and content
    original_sections = {list(section.keys())[0]: list(section.values())[0] for section in original_text}
    modified_sections = {list(section.keys())[0]: list(section.values())[0] for section in modified_text}

    # Compute embeddings
    original_embeddings = model.encode(list(original_sections.values()), convert_to_tensor=True)
    modified_embeddings = model.encode(list(modified_sections.values()), convert_to_tensor=True)

    # Move tensors to CPU before conversion
    original_embeddings = original_embeddings.cpu().numpy()
    modified_embeddings = modified_embeddings.cpu().numpy()

    # Compute similarity scores
    similarity_scores = cosine_similarity(original_embeddings, modified_embeddings)

    # Store results
    verified_matches = []
    matched_original = set()
    matched_modified = set()

    # Match sections based on similarity
    for i, org_id in enumerate(original_sections.keys()):
        best_match = None
        best_score = replace_threshold  # Minimum threshold for replacement

        for j, mod_id in enumerate(modified_sections.keys()):
            score = similarity_scores[i][j]

            if score > best_score:
                best_score = score
                best_match = mod_id

        if best_match:
            if best_score >= threshold:
                change_type = "matched"
            elif replace_threshold <= best_score < threshold:
                change_type = "replaced"
            else:
                continue  # Not confident in a match

            verified_matches.append({
                "change_type": change_type,
                "original_section_id": org_id,
                "modified_section_id": best_match
            })

            matched_original.add(org_id)
            matched_modified.add(best_match)

    # Mark removed sections
    for org_id in original_sections.keys():
        if org_id not in matched_original:
            verified_matches.append({
                "change_type": "removed",
                "original_section_id": org_id,
                "modified_section_id": None
            })

    # Mark added sections
    for mod_id in modified_sections.keys():
        if mod_id not in matched_modified:
            verified_matches.append({
                "change_type": "added",
                "original_section_id": None,
                "modified_section_id": mod_id
            })
    # print("verified_matches--->", verified_matches)
    return verified_matches

def compare_with_groq(original_text, modified_text, threshold=0.999999, replace_threshold=0.999998):
    """
    Compare sections from original_text with modified_text, classify them as matched, replaced, removed, or added.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Extract section IDs and content
    original_sections = {list(section.keys())[0]: list(section.values())[0] for section in original_text}
    modified_sections = {list(section.keys())[0]: list(section.values())[0] for section in modified_text}

    # Compute embeddings
    original_embeddings = model.encode(list(original_sections.values()), convert_to_tensor=True).cpu().numpy()
    modified_embeddings = model.encode(list(modified_sections.values()), convert_to_tensor=True).cpu().numpy()

    # Compute similarity scores
    similarity_scores = cosine_similarity(original_embeddings, modified_embeddings)

    # Store results
    verified_matches = []
    matched_original = set()
    matched_modified = set()

    # Match sections based on highest similarity
    for i, org_id in enumerate(original_sections.keys()):
        best_match = None
        best_score = replace_threshold  # Minimum threshold for replacement

        for j, mod_id in enumerate(modified_sections.keys()):
            if mod_id in matched_modified:
                continue  # Skip already matched modified sections

            score = similarity_scores[i][j]
            if score > best_score:
                best_score = score
                best_match = mod_id

        if best_match:
            change_type = "matched" if best_score >= threshold else "replaced"
            verified_matches.append({
                "change_type": change_type,
                "original_section_id": org_id,
                "modified_section_id": best_match
            })
            matched_original.add(org_id)
            matched_modified.add(best_match)

    # Mark removed sections
    for org_id in original_sections.keys():
        if org_id not in matched_original:
            verified_matches.append({
                "change_type": "removed",
                "original_section_id": org_id,
                "modified_section_id": None
            })

    # Mark added sections
    for mod_id in modified_sections.keys():
        if mod_id not in matched_modified:
            verified_matches.append({
                "change_type": "added",
                "original_section_id": None,
                "modified_section_id": mod_id
            })

    return verified_matches

def check_similarity(sentence1, sentence2):

    def length_percentage_score(num1, num2):
        return (1 - abs(num1 - num2) / max(num1, num2)) * 100

    def get_corrected_sentence(sentence):
        if " | " in sentence:
            return sentence

        # Format the input for grammatical correction
        input_text = f"{sentence}"
        input_ids = tokenizer2.encode(input_text, return_tensors="pt", truncation=False)

        # Generate corrected sentence
        # output_ids = model.generate(input_ids, max_length=256)
        # output_ids = model2.generate(input_ids, max_length=input_ids.shape[1] + 50)
        output_ids = model2.generate(
            input_ids,
            max_length=input_ids.shape[1] + 50,
            no_repeat_ngram_size=2,  # Prevents repeated n-grams
            num_beams=5,  # Improves sentence quality
            early_stopping=True  # Stops when output is good enough
        )
        # corrected_sentence = tokenizer2.decode(output_ids[0], skip_special_tokens=True)
        corrected_sentence = tokenizer2.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # print("output_ids--->", output_ids)
        print("corrected_sentence--->", corrected_sentence)
        return corrected_sentence
        # return corrected_sentence.replace("Correct the grammar in English: ", "")

    def check_and_correct_grammar(sentence):
        """Checks grammar mistakes and corrects if necessary."""
        classifier = pipeline("text-classification", model="textattack/roberta-base-CoLA")
        result = classifier(sentence)
        # print("result--", result)
        if result[0]['label'] == 'LABEL_1':  # If errors are found, correct them
            return get_corrected_sentence(sentence)
        return sentence  # Return the original if no mistakes

    if '<tr>' in sentence1 and '<tr>' not in sentence2:
        return {"is_match": False, "score": 0.0}

    if '<tr>' not in sentence1 and '<tr>' in sentence2:
        return {"is_match": False, "score": 0.0}

    # plain_sentence1 = sentence1
    # plain_sentence2 = sentence2

    plain_sentence1 = utility.extract_plain_text_from_html(sentence1)
    plain_sentence2 = utility.extract_plain_text_from_html(sentence2)

    # plain_sentence1 = check_and_correct_grammar(plain_sentence1)
    # plain_sentence2 = check_and_correct_grammar(plain_sentence2)

    # plain_sentence1 = get_corrected_sentence(plain_sentence1)
    # plain_sentence2 = get_corrected_sentence(plain_sentence2)

    # Tokenize the sentence pair
    inputs = tokenizer1(
        plain_sentence1,
        plain_sentence2,
        return_tensors="pt",
        padding=True,
        truncation=False
    )

    # Get model prediction
    with torch.no_grad():
        outputs = model1(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)

    # The model outputs probabilities for [contradiction, neutral, entailment]
    contradiction_score = predictions[:, 0].item()
    neutral_score = predictions[:, 1].item()
    entailment_score = predictions[:, 2].item()

    length_percentage = length_percentage_score(len(plain_sentence1), len(plain_sentence2))

    # print(f"Sentence 1: {sentence1}")
    # print(f"Sentence 2: {sentence2}")
    # print(f"Contradiction: {contradiction_score:.4f}")
    # print(f"Neutral: {neutral_score:.4f}")
    # print(f"Entailment: {entailment_score:.4f}")
    # print(f"Sentence 1: {plain_sentence1}")
    # print(f"Sentence 2: {plain_sentence2}")

    # Higher entailment score indicates higher similarity
    if entailment_score > neutral_score and entailment_score > contradiction_score:
        print("Result: The sentences are similar (entailment)")
    elif contradiction_score > neutral_score and contradiction_score > entailment_score:
        print("Result: The sentences are contradictory")
    else:
        print("Result: The sentences are neutral (neither similar nor contradictory)")

    # print("1scores--->", entailment_score, contradiction_score, length_percentage) if "(BRD)" in plain_sentence1 and "(BRD)" in plain_sentence2 else None
    # print("2scores--->", entailment_score, contradiction_score, length_percentage) if "The CRM System Upgrade aims" in plain_sentence1 and "The CRM System Upgrade aims" in plain_sentence2 else None
    # print("3scores--->", entailment_score, contradiction_score, length_percentage) if "Enhanced lead tracking and campaign" in plain_sentence1 and "Enhanced lead tracking and campaign" in plain_sentence2 else None
    if entailment_score > neutral_score and entailment_score > contradiction_score:
        # print("contradiction_score, length_percentage, entailment_score--->", contradiction_score, length_percentage, entailment_score) if 'Enhanced lead tracking and campaign performance' in sentence1 else None
        # avg = (percentage_similarity + contradiction_score + entailment_score) / 3
        score_dict = {'entailment':entailment_score, 'contradiction': contradiction_score, 'length': length_percentage}
        # print("Length: ", avg) if 'Enhanced lead tracking and campaign performance' in sentence1 else None
        return {"is_match": True, "score": score_dict}
    else:
        return {"is_match": False, "score": 0.0}

def is_similar_sections3(sentence1, sentence2, threshold=0.999998):
    # Tokenize the sentences
    # inputs = tokenizer([sentence1, sentence2], padding=True, truncation=True, return_tensors='pt')
    inputs = tokenizer(
        [sentence1, sentence2],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    # Get model predictions
    with torch.no_grad():
        logits = model(**inputs).logits

    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # The probability of the 'entailment' class (index 2)
    similarity_score = probs[0, 2].item()

    # return similarity_score
    print(similarity_score >= threshold, similarity_score)
    return {"is_match": similarity_score >= threshold, "score": similarity_score}

def is_similar_sections2(original, modified, threshold=0.950000):
    """Check similarity between original and modified text using BERTScore."""

    def normalize_text(text):
        """Normalize text by removing extra spaces, lowercasing, and handling minor variations."""
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
        return text

    if not original or not modified:  # Handle empty inputs
        return {"is_match": False, "score": 0.0}

    # Normalize text
    original_normalized = original # normalize_text(original)
    modified_normalized = modified # normalize_text(modified)

    # Compute BERTScore (using F1 score as it captures precision and recall better)
    P, R, F1 = score([original_normalized], [modified_normalized], lang="en", model_type="bert-base-uncased")
    bertscore_f1 = F1.item()

    # Final result
    return {"is_match": bertscore_f1 >= threshold, "score": bertscore_f1}

def is_similar_sections1(original, modified, threshold=0.900000):
    """Check similarity between original and modified text while handling minor spelling variations."""

    def normalize_text(text):
        """Normalize text by removing extra spaces, lowercasing, and handling minor variations."""
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
        return text

    if not original or not modified:  # Handle empty inputs
        return {"is_match": False, "score": 0.0}

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Normalize text
    original_normalized = normalize_text(original)
    modified_normalized = normalize_text(modified)

    # Check Levenshtein-based fuzzy similarity
    fuzzy_score = fuzz.ratio(original_normalized, modified_normalized) / 100.0  # Convert to 0-1 scale

    # Generate embeddings
    original_embedding = model.encode(original_normalized, convert_to_tensor=True)
    modified_embedding = model.encode(modified_normalized, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(original_embedding, modified_embedding, dim=0).item()

    # Final similarity score (weighted combination)
    final_score = (cosine_sim + fuzzy_score) / 2  # Averaging both similarity measures

    return {"is_match": final_score >= threshold, "score": final_score}

if __name__ == "__main__":
    file1_path = "./uploads/brd_one.docx"
    file2_path = "./uploads/brd_two.docx"
    # original_text = [
    #     {
    #         "org_section_0": "Business Requirements Document"
    #     },
    #     {
    #         "org_section_1": "I. Introduction This Business Requirements Document (BRD) serves as a foundational reference for all stakeholders, ensuring alignment in project goals and guiding the development of solutions to meet business requirements. The goal is to improve customer management processes, enhance data tracking, and streamline workflows across departments."
    #     },
    #     {
    #         "org_section_3": "II. Project Overview The CRM System Upgrade aims to modernize the organization\u2019s customer management processes by integrating new features and improving system performance. The project will address business needs such as enhanced customer data tracking, improved reporting capabilities, and more efficient workflows for the sales and customer support teams. This initiative will help boost customer satisfaction, drive sales, and optimize internal operations. Prepared by: [YOUR NAME] Email: [YOUR EMAIL] Company Name: [YOUR COMPANY NAME] Company Address: [YOUR COMPANY ADDRESS]"
    #     },
    #     {
    #         "org_section_10": "III. Business Objectives The primary objectives of the CRM System Upgrade are as follows: Objective 1: Improve customer data tracking and reporting to enhance sales strategies and decision-making. Objective 2: Automate sales processes to increase efficiency and reduce manual errors. Objective 3: Enhance customer support capabilities to ensure faster resolution of queries and issues. These objectives will be achieved by implementing a state-of-the-art CRM system that integrates seamlessly with existing tools and supports new features such as predictive analytic and automated reminders."
    #     },
    #     {
    #         "org_section_16": "IV. Scope of the Project The project will include the following activities: Activity 1: CRM system upgrade and configuration to meet business needs. Activity 2: Data migration from the legacy CRM to the new platform, ensuring no loss of customer information. Activity 3: User training and adoption to ensure smooth transition to the new system. Exclusions: Development of any features not related to CRM functionality (e.g., non-sales tools). Integration with systems not already in use (e.g., new marketing platforms outside of the CRM). The project will be executed within the established timeline and constraints."
    #     },
    #     {
    #         "org_section_25": "V. Stakeholder Requirements"
    #     },
    #     {
    #         "org_section_26": "Stakeholder Requirement/Need Priority Deadline Sales Manager Ability to track customer interactions and sales forecasts High January 15, 2050 Customer Support Improved ticket resolution capabilities Medium February 22, 2050 Marketing Director Enhanced lead tracking and campaign performance Low March 10, 2050"
    #     },
    #     {
    #         "org_section_27": "VI. Functional Requirements The following functional requirements have been identified for the project: Requirement 1: The CRM system must integrate with existing email and marketing platforms to track leads and customer interactions. Requirement 2: The system must allow for custom reporting based on customer demographics, sales activity, and support history. Requirement 3: The CRM should include automated task reminders for sales follow-ups, customer service queries, and appointments."
    #     },
    #     {
    #         "org_section_32": "VII. Project Timeline"
    #     },
    #     {
    #         "org_section_33": "Task/Phase Start Date End Date Responsible Party Initial Planning January 5, 2050 January 15, 2050 Project Manager Requirement Gathering January 16, 2050 February 15, 2050 Business Analyst Design Phase February 16, 2050 March 5, 2050 System Architect Implementation March 6, 2050 May 30, 2050 Development Team Go Live June 1, 2050 June 1, 2050 Project Manager"
    #     },
    #     {
    #         "org_section_34": "VIII. Budget and Resources"
    #     },
    #     {
    #         "org_section_35": "Category Estimated Cost Notes Personnel Costs $200,000 Developers, project manager, BA Technology/Tools $150,000 CRM software, integration tools Miscellaneous $50,000 Training materials, contingency"
    #     },
    #     {
    #         "org_section_36": "The resources required include CRM software licenses, data migration tools, development environments, and a dedicated team of 10 people for the duration of the project."
    #     },
    #     {
    #         "org_section_39": "IX. Conclusion The successful implementation of the CRM System Upgrade will provide significant benefits to the organization, including improved sales efficiency, enhanced customer tracking, and a better user experience for both customers and internal teams. By aligning all stakeholders on the requirements and timelines outlined in this document, the project will proceed smoothly, ensuring that all objectives are met on time and within budget."
    #     }
    # ]
    # modified_text = [
    #     {
    #         "mod_section_0": "Business Requirements Document"
    #     },
    #     {
    #         "mod_section_1": "I. Introduction This Business Requirements Document (BRD) serves as a foundational reference for all stakeholders, ensuring alignment in project goals and guiding the development of solutions to meet business requirements. The goal is to improve customer management processes, enhance data tracking, and streamline workflows across departments."
    #     },
    #     {
    #         "mod_section_3": "II. Project Overview The CRM System Upgrade aims to modernize the organization\u2019s customer management processes by integrating new features and improving system performance. The project will address business needs such as enhanced customer data tracking, improved reporting capabilities, and more efficient workflows for the sales and customer support teams. This initiative will help boost customer satisfaction, drive sales, and optimize internal operations. Prepared by: [YOUR NAME] Email: [YOUR EMAIL] Company Name: [YOUR COMPANY NAME] Company Address: [YOUR COMPANY ADDRESS]"
    #     },
    #     {
    #         "mod_section_10": "IV. Scope of the Project The project will include the following activities: Activity 1: CRM system upgrade and configuration to meet business needs. Activity 2: Data migration from the legacy CRM to the new platform, ensuring no loss of customer information. Activity 3: User training and adoption to ensure smooth transition to the new system. Exclusions: Development of any features not related to CRM functionality (e.g., non-sales tools). Integration with systems not already in use (e.g., new marketing platforms outside of the CRM). The project will be executed within the established timeline and constraints."
    #     },
    #     {
    #         "mod_section_20": "III. Business Objectives The primary objectives of the CRM System Upgrade are as follows: Objective 1: Improve customer data tracking and reporting to enhance sales strategies and decision-making. Objective 2: Automate sales processes to increase efficiency and reduce manual errors. Objective 3: Enhance customer support capabilities to ensure faster resolution of queries and issues. These objectives will be achieved by implementing a state-of-the-art CRM system that integrates seamlessly with existing tools and supports new features such as predictive analytic and automated reminders."
    #     },
    #     {
    #         "mod_section_26": "V. Stakeholder Requirements"
    #     },
    #     {
    #         "mod_section_27": "Stakeholder Requirement/Need Priority Deadline Sales Manager Ability to track customer interactions and sales forecasts High January 15, 2050 Customer Support Improved ticket resolution capabilities Medium February 22, 2050 Marketing Director Enhanced lead tracking and campaign performance Low March 10, 2050"
    #     },
    #     {
    #         "mod_section_28": "VI. Functional Requirements The following functional requirements have been identified for the project: Requirement 1: The CRM system must integrate with existing email and marketing platforms to track leads and customer interactions. Requirement 2: The system must allow for custom reporting based on customer demographics, sales activity, and support history. Requirement 3: The CRM should include automated task reminders for sales follow-ups, customer service queries, and appointments."
    #     },
    #     {
    #         "mod_section_33": "VII. Project Timeline"
    #     },
    #     {
    #         "mod_section_34": "Task/Phase Start Date End Date Responsible Party Initial Planning January 5, 2050 January 15, 2050 Project Manager Requirement Gathering January 16, 2050 February 15, 2050 Business Analyst Design Phase February 16, 2050 March 5, 2050 System Architect Implementation March 6, 2050 May 30, 2050 Development Team Go Live June 1, 2050 June 1, 2050 Project Manager"
    #     },
    #     {
    #         "mod_section_35": "VIII. Budget and Resources"
    #     },
    #     {
    #         "mod_section_36": "Category Estimated Cost Notes Personnel Costs $200,000 Developers, project manager, BA Technology/Tools $150,000 CRM software, integration tools"
    #     },
    #     {
    #         "mod_section_40": "IX. Risk Assessment"
    #     },
    #     {
    #         "mod_section_41": "Risk Factor Impact Mitigation Strategy Data Loss High Implement rigorous backup and testing procedures User Resistance Medium Conduct thorough training and provide user support Integration Issues High Perform extensive compatibility testing Security Breaches High Apply encryption, access control, and security audits Budget Overruns Medium Monitor project spending and reallocate resources as needed"
    #     },
    #     {
    #         "mod_section_43": "IX. Conclusion The successful implementation of the CRM System Upgrade will provide significant benefits to the organization, including improved sales efficiency, enhanced customer tracking, and a better user experience for both customers and internal teams. By aligning all stakeholders on the requirements and timelines outlined in this document, the project will proceed smoothly, ensuring that all objectives are met on time and within budget."
    #     }
    # ]
    # comparison_result = compare_with_groq(original_text, modified_text)
    # print(json.dumps(comparison_result, indent=2))

#     sentence5 = "John mango eat "
#     sentence6 = "John do not eat a mango"
#
#     check_similarity(sentence5, sentence6)
#
#     sentence5 = """Aseem Infrastructure Finance Limited, a Company within the meaning of the Companies Act, 2013 having Corporate Identification Number U65990MH2019PLC325794 and its registered office at 4th Floor, UTI Tower, GN Block, South Block, BKC, Bandra (East), Mumbai 400 051 (hereinafter referred to as “AIFL”) which expression shall, unless it be repugnant to the meaning or context thereof, be deemed to mean and include its , successors and assigns of the SECOND PART.
# The Service Provider and AIFL are hereinafter collectively referred to as “Parties” and individually as “Party”.
# WHEREAS:"""
#     sentence6 = """Aseem Infrastructure Finance Limited, a Company within the meaning of the Companies Act, 2013 having Corporate Identification Number U65990DL2019PLC437821 and its registered office at Hindustan Times House, 3rd Floor, 18-20, Kasturba Gandhi Marg, Connaught Place, New Delhi – 110001 (hereinafter referred to as “AIFL”) which expression shall, unless it be repugnant to the meaning or context thereof, be deemed to mean and include its successors and assigns of the SECOND PART.
# The Service Provider and AIFL are hereinafter collectively referred to as “Parties” and individually as “Party”.
# The Service Provider and AIFL are hereinafter collectively referred to as “Parties” and individually as “Party”."""
#
#     check_similarity(sentence5, sentence6)
#
#     sentence5 = "Clover Infotech Private Limited, a company incorporated under the provisions of Companies Act, 1956, with corporate identity number U72200PN2000PTC014922 , having its registered office at Clover Centrum."
#     sentence6 = "Clover Infotech Private Limited, a company incorporated under the provisions of Companies Act, 1956, with corporate identity number U72200PN2000PTC014922 , having its registered office at Clover Centrum,"
#
#     check_similarity(sentence5, sentence6)