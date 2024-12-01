# Import necessary libraries
import gradio as gr
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein, JaroWinkler
from sentence_transformers import SentenceTransformer, util
from typing import List
import zipfile
import os
import io

def calculate_similarity(code1, code2, Ws, Wl, Wj, model_name):
    model = SentenceTransformer(model_name)
    embedding1 = model.encode(code1)
    embedding2 = model.encode(code2)
    sim_similarity = util.cos_sim(embedding1, embedding2).item()
    lev_ratio = Levenshtein.normalized_similarity(code1, code2)
    jaro_winkler_ratio = JaroWinkler.normalized_similarity(code1, code2)
    overall_similarity = Ws * sim_similarity + Wl * lev_ratio + Wj * jaro_winkler_ratio

    return "The similarity score between the two codes is: %.2f" % overall_similarity

# Define the function to process the uploaded file and return a DataFrame
def extract_and_read_compressed_file(file_path):
    file_names = []
    codes = []

    # Handle .zip files
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as z:
            file_names = z.namelist()
            codes = [z.read(file).decode('utf-8', errors='ignore') for file in file_names]

    else:
        raise ValueError("Unsupported file type. Only .zip is supported.")

    return file_names, codes

def filter_and_return_top(df, similarity_threshold,returned_results):
    filtered_df = df[df['similarity_score'] > similarity_threshold]
    return filtered_df.head(returned_results)

# Perform paraphrase mining with the specified weights
def perform_paraphrase_mining(model, codes_list, weight_semantic, weight_levenshtein, weight_jaro_winkler):
    return paraphrase_mining_with_combined_score(
        model,
        codes_list,
        weight_semantic=weight_semantic,
        weight_levenshtein=weight_levenshtein,
        weight_jaro_winkler=weight_jaro_winkler
    )
    
def paraphrase_mining_with_combined_score(
    model,
    sentences: List[str],
    show_progress_bar: bool = False,
    weight_semantic: float = 1.0,
    weight_levenshtein: float = 0.0,
    weight_jaro_winkler: float = 0.0
):
    embeddings = model.encode(
        sentences, show_progress_bar=show_progress_bar, convert_to_tensor=True)
    paraphrases = util.paraphrase_mining_embeddings(embeddings, score_function=util.cos_sim)

    results = []
    for score, i, j in paraphrases:
        lev_ratio = Levenshtein.normalized_similarity(sentences[i], sentences[j])
        jaro_winkler_ratio = JaroWinkler.normalized_similarity(sentences[i], sentences[j])

        combined_score = (weight_semantic * score) + \
                         (weight_levenshtein * lev_ratio) + \
                         (weight_jaro_winkler * jaro_winkler_ratio)

        results.append([combined_score, i, j])

    results = sorted(results, key=lambda x: x[0], reverse=True)
    return results

def get_sim_list(zipped_file,Ws, Wl, Wj, model_name,threshold,number_results):
    file_names, codes = extract_and_read_compressed_file(zipped_file)
    model = SentenceTransformer(model_name)
    code_pairs = perform_paraphrase_mining(model, codes,Ws, Wl, Wj)
    pairs_results = []

    for score, i, j in code_pairs:
      pairs_results.append({
        'file_name_1': file_names[i],
        'file_name_2': file_names[j],
        'similarity_score': score
    })

    similarity_df = pd.concat([pd.DataFrame(pairs_results)], ignore_index=True)
    similarity_df = similarity_df.sort_values(by='similarity_score', ascending=False)
    result = filter_and_return_top(similarity_df,threshold,number_results).round(2)

    return result

# Define the Gradio app
with gr.Blocks(theme=gr.themes.Glass()) as demo:
    # Tab for similarity calculation
    with gr.Tab("Code Pair Similarity"):
        # Input components
        code1 = gr.Textbox(label="Code 1")
        code2 = gr.Textbox(label="Code 2")

        # Accordion for weights and models
        with gr.Accordion("Weights and Models", open=False):
            Ws = gr.Slider(0, 1, value=0.7, label="Semantic Search Weight", step=0.1)
            Wl = gr.Slider(0, 1, value=0.3, label="Levenshiern Distance Weight", step=0.1)
            Wj = gr.Slider(0, 1, value=0.0, label="Jaro Winkler Weight", step=0.1)
            model_dropdown = gr.Dropdown(
            [("codebert", "microsoft/codebert-base"),
             ("graphcodebert", "microsoft/graphcodebert-base"),
             ("UnixCoder", "microsoft/unixcoder-base-unimodal"),
             ("CodeBERTa", "huggingface/CodeBERTa-small-v1"),
             ("CodeT5 small", "Salesforce/codet5-small"),
             ("PLBART", "uclanlp/plbart-java-cs"),],
            label="Select Model",
            value= "uclanlp/plbart-java-cs"
            )

        # Output component
        output = gr.Textbox(label="Similarity Score")

        def update_weights(Ws, Wl, Wj):
            total = Ws + Wl + Wj
            if total != 1:
                Wj = 1 - (Ws + Wl)
            return Ws, Wl, Wj

        # Update weights when any slider changes
        Ws.change(update_weights, [Ws, Wl, Wj], [Ws, Wl, Wj])
        Wl.change(update_weights, [Ws, Wl, Wj], [Ws, Wl, Wj])
        Wj.change(update_weights, [Ws, Wl, Wj], [Ws, Wl, Wj])

        # Button to trigger the similarity calculation
        calculate_btn = gr.Button("Calculate Similarity")
        calculate_btn.click(calculate_similarity, inputs=[code1, code2, Ws, Wl, Wj, model_dropdown], outputs=output)

    # Tab for file upload and DataFrame output
    with gr.Tab("Code Collection Pair Similarity"):
        # File uploader component
        file_uploader = gr.File(label="Upload a Zip file",file_types=[".zip"])

        with gr.Accordion("Weights and Models", open=False):
            Ws = gr.Slider(0, 1, value=0.7, label="Semantic Search Weight", step=0.1)
            Wl = gr.Slider(0, 1, value=0.3, label="Levenshiern Distance Weight", step=0.1)
            Wj = gr.Slider(0, 1, value=0.0, label="Jaro Winkler Weight", step=0.1)
            model_dropdown = gr.Dropdown(
            [("codebert", "microsoft/codebert-base"),
             ("graphcodebert", "microsoft/graphcodebert-base"),
             ("UnixCoder", "microsoft/unixcoder-base-unimodal"),
             ("CodeBERTa", "huggingface/CodeBERTa-small-v1"),
             ("CodeT5 small", "Salesforce/codet5-small"),
             ("PLBART", "uclanlp/plbart-java-cs"),],
            label="Select Model",
            value= "uclanlp/plbart-java-cs"
            )
            threshold = gr.Slider(0, 1, value=0, label="Threshold", step=0.01)
            number_results = gr.Slider(1, 1000, value=10, label="Number of Returned pairs", step=1)

        # Output component for the DataFrame
        df_output = gr.Dataframe(label="Uploaded Data")

        # Button to trigger the file processing
        process_btn = gr.Button("Process File")
        process_btn.click(get_sim_list, inputs=[file_uploader, Ws, Wl, Wj, model_dropdown,threshold,number_results], outputs=df_output)

# Launch the Gradio app with live=True
demo.launch(show_error=True,debug=True)
