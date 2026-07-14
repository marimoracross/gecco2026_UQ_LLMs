import os
import torch
import matplotlib.pyplot as plt
from datasets import DatasetDict
import random
import pandas as pd
import numpy as np
import math
#import seaborn as sns
from IPython.display import display, Markdown


from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score

from huggingface_hub import notebook_login

# Transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
#from transformers import FalconForCausalLM
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset, load_from_disk
from datasets import Dataset
import evaluate
from evaluate import load
from huggingface_hub import login
from peft import PeftModel

# Tensorization
import tensorly as tl
from tensorly.decomposition import tucker


from hdbscan import HDBSCAN, all_points_membership_vectors

# scipy
from scipy.stats import entropy
from scipy.stats import wilcoxon, ttest_rel
from scipy.spatial.distance import mahalanobis


TOKENIZERS_PARALLELISM=False

#------------------------------------------------------------------------------------------------------
def read_data_from_dir(folder_path, column_list, delimiter):
    """
    Read csv files in a directory with the same structure.    
    param: folder_path: Specify the directory containing the CSV files
    """
    # Initialize an empty list to hold the DataFrames
    dataframes = []

    # Loop through the files in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path, encoding='utf-8', delimiter=delimiter) [column_list]
                # Append the DataFrame to the list
                dataframes.append(df)
            except pd.errors.ParserError as e:
                print(f"Error parsing {filename}: {e}")

    # Concatenate all the DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df

# --------------------------------------------------------------------------------------------------------
def cls_pooling(model_output, layer):
    """
    CLS Pooling - Extract the representation of the CLS token (or first token in the sequence).
    :param model_output: Model output containing hidden states.
    :param layer: Layer index to extract features from.
    :return: Tensor with CLS token representation.
    """
    token_embeddings = model_output.hidden_states[layer]  # Hidden states for the given layer
    cls_token_representation = token_embeddings[:, 0, :]  # Extract the first token (CLS token equivalent)
    return cls_token_representation

#----------------------------------------------------------------------------------------------------------
def mean_pooling(model_output, attention_mask, layer):
    """
    Mean Pooling - Take attention mask into account for correct averaging
    :param model_output:  of type transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions
       model_output.hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True
       is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor
       (one for the output of the embeddings, if the model has an embedding layer, +
       one for the output of each layer, Falcon has 32 layers) of shape (batch_size, sequence_length, hidden_size).
       Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
       hidden_size (int, optional, defaults to 4544) — Dimension of the hidden representations.
       Documentation: https://huggingface.co/docs/transformers/main/model_doc/falcon
    :param attention_mask: list of text valid token positions. Example: attention_mask': tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0]
    :param layer: layer number
         num_hidden_layers (int, optional, defaults to 32) — Number of hidden layers in the Transformer decoder.
         ### Outermost LAYER = 32 (final)
         When passing output_hidden_states=True you may expect the outputs.hidden_states[-1] to match outputs.
         last_hidden_states exactly. However, this is not always the case. Some models apply normalization
         or subsequent process to the last hidden state when it’s returned.
        https://huggingface.co/docs/transformers/main_classes/output
    """
    token_embeddings = model_output.hidden_states[layer]  # contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
#----------------------------------------------------------------------------------------------------------
def last_token_embedding(model_output, attention_mask, layer: int):
    """
    Extract the embedding of the last *non-padding* token for each sequence in the batch
    from a given hidden-state layer.

    Args:
        model_output: transformers output with .hidden_states (tuple of tensors)
            Each tensor: [batch_size, seq_len, hidden_size]
        attention_mask: Long/Bool tensor [batch_size, seq_len] with 1 for valid tokens, 0 for padding
        layer: which hidden-states layer to use (e.g., -1 for final)

    Returns:
        Tensor [batch_size, hidden_size] with the last valid token embedding per sequence.
    """
    token_embeddings = model_output.hidden_states[layer]  # [B, L, H]

    # index of last valid token: (num_valid_tokens - 1)
    lengths = attention_mask.long().sum(dim=1)            # [B]
    last_idx = (lengths - 1).clamp(min=0)                 # [B]

    # gather the last token embedding for each sequence
    batch_idx = torch.arange(token_embeddings.size(0), device=token_embeddings.device)
    return token_embeddings[batch_idx, last_idx]          # [B, H]
#----------------------------------------------------------------------------------------------------------

def generate_embeddings_simple_batch(texts, model, tokenizer, layer, embedding_length):
    """
    Generate embeddings using the batch tokenizer capacity  
    texts: list[str]
    returns: np.ndarray shape (N, H)
    """
    tokenizer.pad_token = tokenizer.eos_token

    encoded_input = tokenizer(
        texts,
        padding=True,
        max_length=embedding_length,   # embedding length, not generation length
        truncation=True,
        return_tensors="pt"
    )

    model_device = next(model.parameters()).device
    encoded_input = {k: v.to(model_device) for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input, output_hidden_states=True, return_dict=True)

    sent_emb = mean_pooling(model_output, encoded_input["attention_mask"], layer)  # (N, H)

    return sent_emb.detach().cpu().numpy()


#----------------------------------------------------------------------------------------------------------
def sentence_embedding(sentences, model, tokenizer, device, layer, answer_length):
    tokenizer.pad_token = tokenizer.eos_token

    encoded_input = tokenizer(
        sentences,
        padding=True,
        max_length=answer_length,
        truncation=True,
        return_tensors="pt"
    )

    # use the model's device (works with device_map="auto" + 8-bit)
    model_device = next(model.parameters()).device
    encoded_input = {k: v.to(model_device) for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input, output_hidden_states=True, return_dict=True)

    the_sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"], layer)

    # free GPU tensors
    encoded_input = {k: v.cpu() for k, v in encoded_input.items()}

    return the_sentence_embeddings
    
#-------------------------------------------------------------------------------------------
def sentence_embedding_last_token(sentences, model, tokenizer, device, layer, answer_length):
    tokenizer.pad_token = tokenizer.eos_token

    encoded_input = tokenizer(
        sentences,
        padding=True,
        max_length=answer_length,
        truncation=True,
        return_tensors="pt"
    )

    # use the model's device (works with device_map="auto" + 8-bit)
    model_device = next(model.parameters()).device
    encoded_input = {k: v.to(model_device) for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input, output_hidden_states=True, return_dict=True)

    the_sentence_embeddings = last_token_embedding(model_output, encoded_input["attention_mask"], layer)

    # free GPU tensors
    encoded_input = {k: v.cpu() for k, v in encoded_input.items()}

    return the_sentence_embeddings
    
#----------------------------------------------------------------------------------------------------------
def sentence_embedding_lastToken(sentences, model, tokenizer, device, layer, answer_length):
    tokenizer.pad_token = tokenizer.eos_token

    encoded_input = tokenizer(
        sentences,
        padding=True,
        max_length=answer_length,
        truncation=True,
        return_tensors="pt"
    )

    # use the model's device (works with device_map="auto" + 8-bit)
    model_device = next(model.parameters()).device
    encoded_input = {k: v.to(model_device) for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input, output_hidden_states=True, return_dict=True)

    the_sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"], layer)

    # free GPU tensors ASAP
    encoded_input = {k: v.cpu() for k, v in encoded_input.items()}

    return the_sentence_embeddings

#----------------------------------------------------------------------------------------------------------
def sentence_embeddingBACK(sentences, model, tokenizer, device, layer, answer_length):
    """
    Compute sentence embeddings for a batch of sentences.
    :param sentences: a batch of sentences
    :param model:
    :param tokenizer:
    :param device:
    :param layer:
    :return: sentences_embeddings: a batch of sentence embeddings
    """

    # Tokenize sentence
    tokenizer.pad_token = tokenizer.eos_token

    encoded_input = tokenizer(sentences, padding=True, max_length=answer_length, truncation=True, return_tensors='pt').to(device)

    # Compute token embeddings
    #   - First, you pass your input through the transformer model,
    #   - Then you have to apply the right pooling-operation on-top of the contextualized word embeddings.
    """
    Documentation:
    ### Outermost LAYER = 32 (final)
    When passing output_hidden_states=True you may expect the outputs.hidden_states[-1] to match outputs.last_hidden_states exactly. 
    However, this is not always the case. Some models apply normalization or subsequent process to the last hidden state when it’s returned.
    https://huggingface.co/docs/transformers/main_classes/output
    """
    with torch.no_grad():
        model_output = model(**encoded_input, output_hidden_states=True, return_dict=True)

    # Perform pooling. In this case, max pooling.
    the_sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'], layer)
    #the_sentence_embeddings = cls_pooling(model_output, layer)
    encoded_input.to("cpu")

    return the_sentence_embeddings


#----------------------------------------------------------------------------------------------------------
def generate_embeddings(data_df, model, tokenizer, device, num_samples, layer, answer_length, q_id_field, answer_id_field,  test = True):
    """
    Using data_df generates embeddings for all records.
    :param data_df: df with the dataset
    :param model: model to used for generating embeddings.
    :param tokenizer:
    :param device:
    :param num_samples: number of samples to generate. No in used.
    :param layer: Layer to extract from the model. 
    :param answer_length: 
    :param q_id_field: 
    :param answer_id_field: 
    :param test (default = True) for training samples it is not needed to compute METEOR, BERTScore, 
             TS and perplexity because they are used to train KDE or define clusters.

    """

    # to store results
    embedding_data = []
    q_ids = []
    answers = []
    meteors = []
    bertscores=[]
    perplexities = []
    tscalings = []
    mcds = []

    
    # process all elements in groups  
    for i in range(0, len(data_df)):
        #print(data_df.iloc[i][answer_id_field])
        sentence_embeddings = sentence_embedding(data_df.iloc[i][answer_id_field], model, tokenizer, device, layer, answer_length)
        sentence_embeddings = sentence_embeddings.cpu().numpy()[0]

        # append results
        embedding_data.append(sentence_embeddings)
        q_ids.append(data_df.iloc[i][q_id_field])
        answers.append(data_df.iloc[i][answer_id_field])
        if test: 
            meteors.append(data_df.iloc[i]['meteor_evals'])
            bertscores.append(data_df.iloc[i]['bertscore_f1'])
            perplexities.append(data_df.iloc[i]['perplexities'])
            tscalings.append(data_df.iloc[i]['TS09_uq_values'])
            mcds.append(data_df.iloc[i]['mcd_uq_value'])


    return q_ids, answers, embedding_data, meteors, bertscores, perplexities, tscalings, mcds

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
def generate_embeddings_simple(data_df, model, tokenizer, device, layer, answer_length, answer_id_field):
    """
    Using data_df generates embeddings for all records.
    :param data_df: df with the dataset
    :param model: model to used for generating embeddings.
    :param tokenizer:
    :param device:
    :param layer: Layer to extract from the model. 
    :param answer_length: 
    :param answer_id_field: 

    """

    # to store results
    embedding_data = []
    
    # process all elements in groups  
    for i in range(0, len(data_df)):
        sentence_embeddings = sentence_embedding(data_df.iloc[i][answer_id_field], model, tokenizer, device, layer, answer_length)
        sentence_embeddings = sentence_embeddings.cpu().numpy()[0]

        # append results
        embedding_data.append(sentence_embeddings)

    return embedding_data
    
#----------------------------------------------------------------------------------------------------------
def generate_embeddings_simple_last_token(data_df, model, tokenizer, device, layer, answer_length, answer_id_field):
    """
    Using data_df generates embeddings for all records.
    :param data_df: df with the dataset
    :param model: model to used for generating embeddings.
    :param tokenizer:
    :param device:
    :param layer: Layer to extract from the model. 
    :param answer_length: 
    :param answer_id_field: 

    """

    # to store results
    embedding_data = []
    
    # process all elements in groups  
    for i in range(0, len(data_df)):
        sentence_embeddings = sentence_embedding_last_token(data_df.iloc[i][answer_id_field], model, tokenizer, device, layer, answer_length)
        sentence_embeddings = sentence_embeddings.cpu().numpy()[0]

        # append results
        embedding_data.append(sentence_embeddings)

    return embedding_data    
    
#----------------------------------------------------------------------------------------------------------
def generate_embeddings_ds(data_df, model, tokenizer, device, num_samples, layer, answer_length, q_id_field, answer_id_field,  test = True):
    """
    Using data_df generates embeddings for all records. Returns one additional column, the MCD of the full text (mcd02).
    :param data_df: df with the dataset
    :param model: model to used for generating embeddings.
    :param tokenizer:
    :param device:
    :param num_samples: number of samples to generate. No in used.
    :param layer: Layer to extract from the model. 
    :param answer_length: 
    :param q_id_field: 
    :param answer_id_field: 
    :param test (default = True) for training samples it is not needed to compute METEOR, BERTScore, TS 
             and perplexity because they are used to train KDE or define clusters.
    """

    # to store results
    embedding_data = []
    q_ids = []
    answers = []
    meteors = []
    bertscores=[]
    perplexities = []
    tscalings = []
    mcds = []
    mcd02 = []

    
    # process all elements in groups  
    for i in range(0, len(data_df)):
        #print(data_df.iloc[i][answer_id_field])
        sentence_embeddings = sentence_embedding(data_df.iloc[i][answer_id_field], model, tokenizer, device, layer, answer_length)
        sentence_embeddings = sentence_embeddings.cpu().numpy()[0]

        # append results
        embedding_data.append(sentence_embeddings)
        q_ids.append(data_df.iloc[i][q_id_field])
        answers.append(data_df.iloc[i][answer_id_field])
        if test: 
            meteors.append(data_df.iloc[i]['meteor_evals'])
            bertscores.append(data_df.iloc[i]['bertscore_f1'])
            perplexities.append(data_df.iloc[i]['perplexities'])
            tscalings.append(data_df.iloc[i]['TS09_uq_values'])
            mcds.append(data_df.iloc[i]['mcd_uq_value'])
            mcd02.append(data_df.iloc[i]['mcd_uq_value_paragraph'])


    return q_ids, answers, embedding_data, meteors, bertscores, perplexities, tscalings, mcds, mcd02    
    
# ----------------------------------------------------------------------------------------------------------------------
def generate_prediction_model02(sample_question, tokenizer, model, answer_length, topk, temperature, topp, device ):
    """
    It receives a model (ex deepseek) a dataframe with a sample record and the parameters topk and penalty_alpha
    and returns an answer string.

    Sampling introduces natural variation, which is the key to observing dispersion in the inclusion space.
    Varied generations allow for more expressive distributions, ideal for estimating density or detecting dense/rare regions.
    Variation in outputs allows us to observe how uncertainty correlates with response quality.
    Sampling generates distinct outputs that you can use to apply Monte Carlo Dropout-like dispersion analysis without having to modify weights.
    params:
       tokenizer_name: the model tokenizer name
       model: the model to test.
    return:
       predictions
    """
    # data = data.to_dict()

    # print('data[question]', data['question'] )

    # sample_question = data['question'].strip()
    # print("=============================== La pregunta")
    # print(sample_question)

    # Tokenize text
    #inputs = tokenizer(sample_question, max_length=answer_length, truncation=True, return_tensors="pt")
    inputs = tokenizer(sample_question, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attn_masks = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids.to(device),
            attention_mask=attn_masks.to(device),
            max_new_tokens=answer_length,
            do_sample=True,
            temperature=temperature,
            top_k=topk,
            top_p=topp,
            num_return_sequences=1
        )

    model_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # print("************** Model answer : ", model_answer)

    return model_answer


#---------------------------------------------------------------------------------------------------
def load_model_and_tokenizer(model_name, adapter_model, tokenizer_name, answer_length, device="cuda:0"):
    """ 8-bit only (bitsandbytes 0.42.0), avoid accelerate dispatch_model calling model.to() on 
    quantized models. Loads on CPU, then moves to one GPU. """
    import inspect
    print("USING load_model_and_tokenizer from:", inspect.getsourcefile(load_model_and_tokenizer))
    print("load_model_and_tokenizer line:", inspect.getsourcelines(load_model_and_tokenizer)[1])

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    if device is not None:
        model = model.to(device)

    # Adapter loading
    if adapter_model:
        loaded = False
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_model)
            loaded = True
        except Exception:
            pass

        if not loaded:
           try:
              model.load_adapter(adapter_model)
              loaded = True
           except Exception as e:
              raise RuntimeError(
                    f"Could not load adapter from {adapter_model}. "
                    f"Tried PEFT and model.load_adapter. Last error: {e}"
              )

    tok_src = tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model.eval()
    torch.set_grad_enabled(False)

    return model, tokenizer



#------------------------------------------------------------------------------------------------
def load_model_and_tokenizerBACK(model_name, adapter_model, tokenizer_name, answer_length, device):
    """Load pretrained model, the model adapter and tokenizer. It uses the BitsAndBytes library to load
    the model in 8-bit precision, reducing memory usage.
    Params:
        model_name (str): The name or path of the pre-trained model to be loaded.
        tokenizer_name (str): The name or path of the pre-trained tokenizer associated with the model.
        answer_length (int): The maximum length for tokenizing inputs.
        device (str): The device on which to load the model, typically 'cpu', 'cuda', or another device identifier.
    Returns:
        model (transformers.AutoModelForCausalLM): The pre-trained model loaded with 8-bit quantization.
        tokenizer (transformers.AutoTokenizer): The tokenizer corresponding to the pre-trained model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    # load the adapter
    model.load_adapter(adapter_model)

    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, max_length=answer_length, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Ensure gradients are not calculated during inference
    model.eval()
    torch.set_grad_enabled(False)

    return model, tokenizer

 #------------------------------------------------------------------------------------
def load_model_and_tokenizer_model02(model_name, tokenizer_name, answer_length, device):
    """Load pretrained model, the model adapter and tokenizer. It uses the BitsAndBytes library to load
    the model in 8-bit precision, reducing memory usage. 
    Params:
        model_name (str): The name or path of the pre-trained model to be loaded.
        tokenizer_name (str): The name or path of the pre-trained tokenizer associated with the model.
        answer_length (int): The maximum length for tokenizing inputs.
        device (str): The device on which to load the model, typically 'cpu', 'cuda', or another device identifier.
    Returns:
        model (transformers.AutoModelForCausalLM): The pre-trained model loaded with 8-bit quantization.
        tokenizer (transformers.AutoTokenizer): The tokenizer corresponding to the pre-trained model.
    """

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )

    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, max_length=answer_length, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model.eval()
    torch.set_grad_enabled(False)
    
    # to avoid the warning: Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    # After loading tokenizer and model
    if tokenizer.pad_token_id is None:
       # Prefer EOS as PAD if available
       if tokenizer.eos_token_id is not None:
          tokenizer.pad_token_id = tokenizer.eos_token_id
       else:
          # last resort: use 0 or bos
          tokenizer.pad_token_id = tokenizer.bos_token_id or 0

    if model.config.pad_token_id is None:
       model.config.pad_token_id = tokenizer.pad_token_id

    if model.config.eos_token_id is None and tokenizer.eos_token_id is not None:
       model.config.eos_token_id = tokenizer.eos_token_id


    return model, tokenizer


#-----------------------------------------------------------------------------------------
def compute_kde_sklearn(data):
    """generate a KDE models. It estimates the underlying probability density function (PDF).
    It uses cross-validation to estimate the performance of the model.
    It then selects the hyperparameters that result in the highest cross-validated log-likelihood."""

    # use grid search cross-validation to optimize the bandwidth
    params = { 'kernel': ['gaussian'],
        "bandwidth": np.logspace(-1, 1.5, 30)}
 
    grid = GridSearchCV(estimator=KernelDensity(), param_grid=params, n_jobs=-1, verbose=0)
    grid.fit(data)

    # use the best estimator to compute the kernel density estimate
    kde = grid.best_estimator_
    bandwith = grid.best_estimator_.bandwidth
    return kde

#--------------------------------------------------------------------------------------------
def merge_small_clusters(embeddings, labels, min_samples):
    """
    Merges clusters with fewer than `min_samples` points with the nearest clusters.
    Parameters:
	- embeddings: np.array of form (n_samples, n_features)
	- labels: Array with cluster labels assigned by KMeans
	- min_samples: Minimum number of points per cluster
    Returns:
	- new_labels: New version of labels after the merge.
    """
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)

    # Exclude -1 when searching for small clusters
    small_clusters = unique[(counts < min_samples) & (unique != -1)]
    if len(small_clusters) == 0:
        return labels

    # Identify large (valid) clusters excluding -1 and small ones
    valid_clusters = [c for c in unique if c != -1 and counts[unique.tolist().index(c)] >= min_samples]
    cluster_centers = {c: np.mean(embeddings[labels == c], axis=0) for c in valid_clusters}

    new_labels = labels.copy()

    for cluster in small_clusters:
        cluster_points = embeddings[labels == cluster]

        if len(cluster_points) >= min_samples:
            continue

        small_cluster_center = np.mean(cluster_points, axis=0)
        if np.isnan(small_cluster_center).any():
            print(f"Cluster {cluster} tiene NaN en su centroide y será ignorado.")
            continue

        # Encontrar el cluster grande más cercano
        distances = {c: np.linalg.norm(small_cluster_center - center) for c, center in cluster_centers.items()}
        if distances:
            closest_cluster = min(distances, key=distances.get)
            new_labels[labels == cluster] = closest_cluster
        else:
            print(f"No hay clusters válidos para fusionar el cluster {cluster}.")

    return new_labels


#---------------------------------------------------------------------------------------------------------------------        
def calculate_kde_for_clusters(embeddings, n_clusters, min_samples_in_kmeans):
    """
    # Apply KMeans: for K-Means compute KDE for each cluster 
    
    Parámetros:
    - embeddings: np.array con los datos a clusterizar.
    - n_clusters: Número de clusters iniciales para K-Means.
    - min_samples_in_kmeans: Número mínimo de muestras por cluster para ser válido.
    
    Retorna:
    - kde_models: Lista de modelos KDE ajustados a cada cluster.
    - uncertainties: Lista de incertidumbres (densidad negativa promedio) por cluster.
    - kmeans: Modelo KMeans ajustado.
    - cluster_labels: Etiquetas de clusters después de fusionar clusters pequeños.
    """

    # Aplicar K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # merge small clusters
    cluster_labels = merge_small_clusters(embeddings, cluster_labels, min_samples_in_kmeans) 
    
    kde_models = []
    uncertainties = []

    # Iterar sobre los valores únicos de cluster_labels en lugar de un rango fijo
    for cluster_id in np.unique(cluster_labels):
        #Filter embeddings that belong to the current cluster
        cluster_data = embeddings[cluster_labels == cluster_id]
        
        # Evitar clusters vacíos (puede ocurrir después de fusionar clusters)
        if len(cluster_data) == 0:
            kde_models.append(None)
            uncertainties.append(np.inf)
            continue

        # Compute KDE for current cluster
        kde = compute_kde_sklearn(cluster_data)

        # Compute the average uncertainty inside each cluster (negative log-density)
        log_density = kde.score_samples(cluster_data)
        avg_uncertainty = -np.mean(log_density)  
        
        # Save the KDE model and uncertainty 
        kde_models.append(kde)
        uncertainties.append(avg_uncertainty)
        
    return kde_models, uncertainties, kmeans, cluster_labels
    
#---------------------------------------------------------------------------------------------------------------------        
def calculate_kmeans(embeddings, n_clusters, min_samples_in_kmeans):
    """
    # Apply KMeans to compute cluster 
    
    Parámetros:
    - embeddings: np.array with data for clustering.
    - n_clusters: numbers of clusters for K-MEANS.
    - min_samples_in_kmeans: Minimum number of samples per cluster to be valid.
    
    Return:
    - kmeans: KMeans model.
    - cluster_labels: Cluster labels after merging small clusters.
    """

    # apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # merge small clusters
    cluster_labels = merge_small_clusters(embeddings, cluster_labels, min_samples_in_kmeans) 
    
        
    return kmeans, cluster_labels    
 
#------------------------------------------------------------------------------------------------------
def prepare_MCD_TS_data (test_meteors,test_bertsf1,test_perpl,test_ts,test_mcd,
                      d01_meteors,d01_bertsf1,d01_perpl,d01_ts,d01_mcd,
                      d02_meteors,d02_bertsf1,d02_perpl,d02_ts,d02_mcd) :
    # ================== Create DataFrames for Visualization
    test_df = pd.DataFrame({
                'meteor_evals': test_meteors,
                'bertscore_f1': test_bertsf1,
                'perplexities': test_perpl,
                'TS': test_ts,
                'MCD': test_mcd
    })

    d01_df = pd.DataFrame({
                'meteor_evals': d01_meteors,
                'bertscore_f1': d01_bertsf1,
                'perplexities': d01_perpl,
                'TS': d01_ts,
                'MCD': d01_mcd
    })

    d02_df = pd.DataFrame({
                'meteor_evals': d02_meteors,
                'bertscore_f1': d02_bertsf1,
                'perplexities': d02_perpl,
                'TS': d02_ts,
                'MCD': d02_mcd })

    return (d01_df, d02_df, test_df)

#------------------------------------------------------------------------------------------------------
def find_nearest_cluster_and_avg_knn_distance(kmeans_model, new_cluster_labels, data, new_data, k=5):
    """
    Find the nearest cluster to each point in new_data and compute the average distance to the K nearest neighbors within that cluster.

    Parameters:
        - kmeans_model (KMeans): Trained KMeans model.
        - new_cluster_labels (numpy.ndarray): Cluster labels corrected after merging small clusters.
        - data (numpy.ndarray): Training data used for clustering.
        - new_data (numpy.ndarray): Set of points for which the nearest cluster is sought.
        - k (int): Number of nearest neighbors to consider.
    Returns
        results: A list containing the average distance to the K nearest neighbors within the nearest cluster for each point.
    """
    results = []
    max_dist = 0

    # obtain the centroids of the merged clusters
    unique_labels = np.unique(new_cluster_labels)
    cluster_centers = np.array([np.mean(data[new_cluster_labels == c], axis=0) for c in unique_labels])

    for i, test_point in enumerate(new_data):
        # Find the cluster closest to the test point
        distances_to_centroids = np.linalg.norm(cluster_centers - test_point, axis=1)  # Distancia Euclidiana
        nearest_cluster = unique_labels[np.argmin(distances_to_centroids)]

        # Filter the points that belong to the nearest cluster
        cluster_points = data[new_cluster_labels == nearest_cluster]

        if len(cluster_points) < k:
            print(f"Cluster {nearest_cluster} tiene menos de {k} puntos, omitiendo cálculo de KNN para el punto {i}.")
            results.append(max_dist)
            continue

        # Calculate average distance to the K nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k).fit(cluster_points)
        distances, _ = nbrs.kneighbors([test_point])
        avg_distance = np.mean(distances)

        # Maintain the greatest possible distance
        max_dist = max(max_dist, avg_distance)

        results.append(avg_distance)

    return results
    
# -------------------------------------------------------------------------------------
def calculate_uncertainty_for_new_samples(new_embeddings, kmeans, kmeans_new_labels, kde_models, clusters_uq, min_cluster_size):
    """
    Calculate the uncertainty of new samples using K-Means with KDE.
    Parameters:
	- new_embeddings: np.array with the new embeddings to be evaluated.
	- kmeans: Trained KMeans model.
	- kmeans_new_labels: Corrected cluster labels after merging small clusters.
	- kde_models: List of KDE models fitted to each cluster.
	- clusters_uq: List of precomputed uncertainties per cluster.
	- cluster_weight: Weight for combining the cluster and sample uncertainties.
	- min_cluster_size: Minimum number of points a cluster must have to use KDE.

   Returns:
	- List of individual uncertainties.
	- List of combined uncertainties.    """
    uncertainties = []
  
    for embedding in new_embeddings:
        embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)

        # predict the asigned cluster 
        cluster_id = kmeans.predict(embedding)[0]
        cluster_size = np.sum(kmeans_new_labels == cluster_id)

        # If the cluster is small or does not have a valid KDE model, look for another nearby one.
        if (cluster_size < min_cluster_size or 
            cluster_id >= len(kde_models) or 
            kde_models[cluster_id] is None):
            
            # Distance to all centroids
            distances = np.linalg.norm(kmeans.cluster_centers_ - embedding, axis=1)

            # Order clusters by proximity
            sorted_indices = np.argsort(distances)

            reassigned = False
            for alt_id in sorted_indices:
                alt_size = np.sum(kmeans_new_labels == alt_id)
                if (alt_size >= min_cluster_size and 
                    alt_id < len(kde_models) and 
                    kde_models[alt_id] is not None):

                    cluster_id = alt_id
                    reassigned = True
                    break

            if not reassigned:
                uncertainties.append(np.inf)
                combined_uncertainties.append(np.inf)
                continue

        # compute density using KDE
        kde = kde_models[cluster_id]
        log_density = kde.score_samples(embedding)[0]
        uncertainty = -log_density  # Negative because low density implies more uncertainty

        uncertainties.append(uncertainty)

    return uncertainties


#---------------------------------------------------------------------------------
def standard_scaler_normalization(train_sentence_embeddings, eval_embedding_data):
    """
    Applies StandardScaler to all embeddings (train and eval).
    """
    all_embeddings = np.concatenate((train_sentence_embeddings, eval_embedding_data), axis=0)

    scaler = StandardScaler()
    scaler.fit(all_embeddings)

   
    train_sentence_embeddings = scaler.transform(train_sentence_embeddings)
    eval_embeddings = scaler.transform(eval_embedding_data)

    return train_sentence_embeddings, eval_embeddings

#---------------------------------------------------------------------------------
def standard_scaler_normalization_scaler(train_sentence_embeddings, eval_embedding_data):
    """
    Applies StandardScaler to all embeddings (train and eval).
    """
    all_embeddings = np.concatenate((train_sentence_embeddings, eval_embedding_data), axis=0)

    scaler = StandardScaler()
    scaler.fit(all_embeddings)

   
    train_sentence_embeddings = scaler.transform(train_sentence_embeddings)
    eval_embeddings = scaler.transform(eval_embedding_data)

    return train_sentence_embeddings, eval_embeddings, scaler


#--------------------------------------------------------------------------------------------
def reduce_data_dimensionality(train_sentence_embeddings, eval_embeddings, embedding_dimension, algorithm_for_reduction):
    """
    Dimensionality reduction using different algorithms like ISOMAP, PCA, etc. Default PCA.
    :param sentence_embeddings: a sentence o paragraph embeddings.
    :param algorithm_for_reduction: the name of the algorithm to use for dimensionality reduction.
    :return:
        data_reduce
    """
    #
    match algorithm_for_reduction:
        case 'TSNE':

            tsne = TSNE(n_components=embedding_dimension, random_state=42)

            embedding_train = tsne.fit(np.array(train_sentence_embeddings))

            # dimensionality reduction of train data
            train_data_reduced = embedding_train.transform(np.array(train_sentence_embeddings))

            # dimensionality reduction of eval data
            eval_data_reduced = embedding_train.transform(np.array(eval_embeddings))


        case 'ISOMAP':

            dim_reduction_instance = Isomap(n_components=embedding_dimension)

            dim_reduction_instance.fit(train_sentence_embeddings)

            # dimensionality reduction of train data
            train_data_reduced = dim_reduction_instance.transform(train_sentence_embeddings)

            # dimensionality reduction of eval data
            #test_data_reduced = dim_reduction_instance.transform(eval_embeddings)
            eval_data_reduced = dim_reduction_instance.transform(eval_embeddings)



        case 'SVD':
            #trainTensor_sentence_embeddings = torch.from_numpy(train_sentence_embeddings)       
            #evalTensor_embeddings = torch.from_numpy(eval_embeddings)
            
            #U_train, S_train, V_train = torch.svd(trainTensor_sentence_embeddings)

            # Truncate V to create the projection matrix
            #V_train_k = V_train[:, :embedding_dimension]

            # Apply the projection matrix to the train embeddings 
            #train_data_reduced = trainTensor_sentence_embeddings @ V_train_k

            # Apply the projection matrix to the eval embeddings 
            #eval_data_reduced = evalTensor_embeddings @ V_train_k

            # from tensor to numpy 
            #train_data_reduced = train_data_reduced.detach().numpy()
            #eval_data_reduced =  eval_data_reduced.detach().numpy()
            
            Xtr = np.asarray(train_sentence_embeddings, dtype=np.float64)
            Xev = np.asarray(eval_embeddings, dtype=np.float64)

            # projection matrix W has shape (D, d)
            U, S, Vt = randomized_svd(Xtr, n_components=int(embedding_dimension), random_state=0)
            W = Vt.T

            train_data_reduced = Xtr @ W
            eval_data_reduced  = Xev @ W

        case "TUCKER":
            # Apply Tucker decomposition to the training embeddings
            core_train, factors_train = tucker(train_sentence_embeddings, rank=[train_sentence_embeddings.shape[0], 
                                        embedding_dimension]) 

            # Apply the projection matrix to reduce dimensionality of test and distant data
            train_data_reduced = train_sentence_embeddings @ factors_train[1]  # For completeness
            eval_data_reduced = eval_embeddings @ factors_train[1]


        case _:

            # dim_reduction_instance = Isomap(n_components=embedding_dimension)
            dim_reduction_instance = PCA(n_components=embedding_dimension,     
                              svd_solver="randomized",   # fast for large dense matrices
                              random_state=0,
                              whiten=False,              # keep geometry for KNN distances
                              )

            dim_reduction_instance.fit(train_sentence_embeddings)

            # dimensionality reduction of train data
            train_data_reduced = dim_reduction_instance.transform(train_sentence_embeddings)

            # dimensionality reduction of test data
            eval_data_reduced = dim_reduction_instance.transform(eval_embeddings)

  
    return train_data_reduced, eval_data_reduced
    
    

def compute_pca (train_sentence_embeddings, eval_embeddings, embedding_dimension):
    """
    Dimensionality reduction using different algorithms like ISOMAP, PCA, etc. Default PCA.
    """
    dim_reduction_instance = PCA(n_components=embedding_dimension,     
                              svd_solver="randomized",   # fast for large dense matrices
                              random_state=0,
                              whiten=False,              # keep geometry for KNN distances
                              )

    dim_reduction_instance.fit(train_sentence_embeddings)

    # dimensionality reduction of train data
    train_data_reduced = dim_reduction_instance.transform(train_sentence_embeddings)

    # dimensionality reduction of test data
    eval_data_reduced = dim_reduction_instance.transform(eval_embeddings)
    
    return train_data_reduced, eval_data_reduced, dim_reduction_instance

#--------------------------------------------------------------------------------------------
def load_data_random(root_folder_path,
                          num_samples,num_samples_testing,num_samples_distant,layer_embedding, random_state=1):

    # train data
    train_folder = 'Train/'
    base_df = read_data_from_dir(root_folder_path  + train_folder ,   
                             ['train_questions_id', 'train_model_answers', 'train_best_answers', 'train_questions'], '|')    
    # random records
    train_samples_df = base_df.sample(n=num_samples, random_state=random_state)  
    #print("Selected train samples ", len(train_samples_df))

    # test data
    test_folder = 'Test/'   
    test_df = read_data_from_dir(root_folder_path  + test_folder + layer_embedding ,   
                 ['question', 'test_model_answers', 'test_best_answers', 
                 'meteor_evals', 'bertscore_f1', 'perplexities', 'TS09_uq_values', 'mcd_uq_value'], '|')    
    # random records
    test_samples_df = test_df.sample(n=num_samples_testing, random_state=random_state)  

    # distant 01 data
    distant01_folder = 'Distant01/'   
    dist01_df = read_data_from_dir(root_folder_path  + distant01_folder + layer_embedding ,   
                 ['question', 'distant_model_answers', 'distant_true_answers', 
                 'meteor_evals', 'bertscore_f1', 'perplexities',  'TS09_uq_values', 'mcd_uq_value'], '|')    
    # random recordsx
    #print('============== dist01_samples ================= ')    
    dist01_samples_df = dist01_df.sample(n=num_samples_distant, random_state=random_state)  
        
    # distant 02 data
    distant02_folder = 'Distant02/'   
    dist02_df = read_data_from_dir(root_folder_path  + distant02_folder + layer_embedding ,   
                 ['question', 'distant_model_answers', 'distant_true_answers', 
                 'meteor_evals', 'bertscore_f1', 'perplexities', 'TS09_uq_values', 'mcd_uq_value'], '|')    
    # random records
    #print('============== dist02_samples ================= ')
    dist02_samples_df = dist02_df.sample(n=num_samples_distant, random_state=random_state)  
    return train_samples_df, test_samples_df, dist01_samples_df, dist02_samples_df
    
#------------------------------------------------------------------------------------------------------
def load_experiment_results(file_path):
    """
    Loads an experiment results CSV file into a DataFrame.

    Parameters:
    - file_path: str, path to the CSV file.

    Returns:
    - df: Pandas DataFrame containing the experiment results.
    """
    df = pd.read_csv(file_path)
    return df
    

#--------------------------------------------------------------------------------------------------------------------------

def compute_kl_divergence(p, q):
    """
    Computes the Kullback-Leibler (KL) Divergence between two probability distributions.
    
    Parameters:
    - p: First probability distribution (numpy array or list, should sum to 1)
    - q: Second probability distribution (numpy array or list, should sum to 1)
    
    Returns:
    - KL Divergence value:  a real number
    """
    p = np.array(p)
    q = np.array(q)
    
    # Ensure the distributions sum to 1 (normalize if needed)
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    
    return entropy(p, q)
    
#--------------------------------------------------------------------------------------------------------------------------
def compute_r2_score(y_true, y_pred):
    """
    Computes the R² (coefficient of determination) between true and predicted values.
    
    Parameters:
    - y_true: Array of true values
    - y_pred: Array of predicted values
    
    Returns:
    - R² score:  a real number
    """
    return r2_score(y_true, y_pred)

#--------------------------------------------------------------------------------------------------------------------------
# CLUSTERING AND KDE OPERATIONS

#------------------------------------------------------------------------------------------------------
def compute_kmeans_kde (train_data_reduced, test_data_reduced, distant_data_reduced,  distant02_data_reduced,
                        test_meteors, test_bertsf1,test_perpl,
                        d01_meteors, d01_bertsf1, d01_perpl, 
                        d02_meteors, d02_bertsf1, d02_perpl, min_samples_in_kmeans, k_clusters):
    """
    Compute kmeans and apply KDE for each cluster. Additionally calculate uncertainty for datasets (test, distant and distant02).
    
    """
    
    # ==== Clustering ====
    # convert train_data_reduced to float32 (for clustering)
    train_data_reduced = np.array(train_data_reduced, dtype=np.float32)
    test_data_reduced = np.array(test_data_reduced, dtype=np.float32)
    
    # create clusters
    kde_models, clusters_uq, kmeans, k_means_labels = calculate_kde_for_clusters(train_data_reduced, k_clusters, min_samples_in_kmeans)

    # merge small clusters and updates the list of labels.
    labels = kmeans.labels_
    k_means_labels = merge_small_clusters(train_data_reduced, labels, min_samples_in_kmeans)  
    
    #==== Compute uncertainty ====
    test_uq = calculate_uncertainty_for_new_samples(test_data_reduced, kmeans, k_means_labels, kde_models, 
                                                                           clusters_uq, min_samples_in_kmeans)

    d01_uq = calculate_uncertainty_for_new_samples(distant_data_reduced, kmeans, k_means_labels, kde_models, 
                                                                         clusters_uq, min_samples_in_kmeans)

    d02_uq = calculate_uncertainty_for_new_samples(distant02_data_reduced, kmeans, k_means_labels, kde_models, 
                                                                         clusters_uq, min_samples_in_kmeans)    
    # dataframes for visualizacion
    test_df = pd.DataFrame({
                'KDE_uq_values': test_uq, 
                'meteor_evals': test_meteors, 
                'bertscore_f1': test_bertsf1, 
                'perplexities': test_perpl})
    
    d01_df =  pd.DataFrame({
                'KDE_uq_values': d01_uq, 
                'meteor_evals': d01_meteors,
                'bertscore_f1': d01_bertsf1,
                'perplexities': d01_perpl})

    d02_df =  pd.DataFrame({
                'KDE_uq_values': d02_uq, 
                'meteor_evals': d02_meteors,
                'bertscore_f1': d02_bertsf1,
                'perplexities': d02_perpl})    

    return (d01_df, d02_df, test_df)


#------------------------------------------------------------------------------------------------------
def compute_kmeans_KNN (train_data_reduced, test_data_reduced, distant_data_reduced,  distant02_data_reduced,
                                        test_meteors,test_bertsf1,test_perpl,
                                        d01_meteors,d01_bertsf1,d01_perpl, d02_meteors,d02_bertsf1,d02_perpl, 
                                        min_samples_in_kmeans, k_clusters): 
    """
    Fit KMeans on reduced training features and compute a KNN based uncertainty score
    for test data and the two distant datasets. Return three pandas DataFrames with
    uncertainty and evaluation metrics.

    The uncertainty score is the mean distance to the k nearest neighbors inside the
    nearest KMeans cluster of each point. 
    
    Args:
        train_data_reduced (np.ndarray or pd.DataFrame):
            Training features after dimensionality reduction, shape [n_train, d].
        test_data_reduced (np.ndarray or pd.DataFrame):
            Test features after dimensionality reduction, shape [n_test, d].
        distant_data_reduced (np.ndarray or pd.DataFrame):
            Distant01 features after dimensionality reduction, shape [n_d01, d].
        distant02_data_reduced (np.ndarray or pd.DataFrame):
            Distant02 features after dimensionality reduction, shape [n_d02, d].

        test_meteors (Sequence[float]): METEOR scores for test items, length n_test.
        test_bertsf1 (Sequence[float]): BERTScore F1 for test items, length n_test.
        test_perpl   (Sequence[float]): Perplexities for test items, length n_test.

        d01_meteors  (Sequence[float]): METEOR scores for Distant01, length n_d01.
        d01_bertsf1  (Sequence[float]): BERTScore F1 for Distant01, length n_d01.
        d01_perpl    (Sequence[float]): Perplexities for Distant01, length n_d01.

        d02_meteors  (Sequence[float]): METEOR scores for Distant02, length n_d02.
        d02_bertsf1  (Sequence[float]): BERTScore F1 for Distant02, length n_d02.
        d02_perpl    (Sequence[float]): Perplexities for Distant02, length n_d02.

    """
    
    # create clusters and copute KDE 
    kmeans, k_means_labels = calculate_kmeans(train_data_reduced, k_clusters, min_samples_in_kmeans)
 
    #==== Compute uncertainty ====
    test_uq = find_nearest_cluster_and_avg_knn_distance(kmeans, k_means_labels, train_data_reduced, test_data_reduced, 
                                                        min_samples_in_kmeans)
    
    d01_uq = find_nearest_cluster_and_avg_knn_distance(kmeans, k_means_labels, train_data_reduced,distant_data_reduced, min_samples_in_kmeans)
    
    d02_uq = find_nearest_cluster_and_avg_knn_distance(kmeans,k_means_labels, train_data_reduced, distant02_data_reduced, min_samples_in_kmeans)
    
    # data for visualizacion
    test_df = pd.DataFrame({
                'KDE_uq_values': test_uq, 
                'meteor_evals': test_meteors, 
                'bertscore_f1': test_bertsf1, 
                'perplexities': test_perpl})
    
    d01_df =  pd.DataFrame({
                'KDE_uq_values': d01_uq, 
                'meteor_evals': d01_meteors,
                'bertscore_f1': d01_bertsf1,
                'perplexities': d01_perpl})

    d02_df =  pd.DataFrame({
                'KDE_uq_values': d02_uq, 
                'meteor_evals': d02_meteors,
                'bertscore_f1': d02_bertsf1,
                'perplexities': d02_perpl})    

    return (d01_df, d02_df, test_df)
    
#--------------------------------------------------------------------------------------------------------
def merge_small_clusters_hdbscan(hdb_model, train_data, min_samples=5):
    """
    Merges small clusters with the nearest cluster using Euclidean distances between centroids.

    Parameters:
        - hdb_model: Trained HDBSCAN model.
        - train_data: np.array with training data.
        - min_samples: Minimum number of points a cluster must have to avoid being merged.

    Returns:
        - new_labels: Cluster labels after merging.    
    """
    cluster_labels = hdb_model.labels_
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)

    # Separate small and large clusters
    small_clusters = unique_labels[counts < min_samples]
    large_clusters = unique_labels[counts >= min_samples]

    if len(small_clusters) == 0:
        return cluster_labels  # There are no small clusters to merge

    # Calculate centroids of large clusters
    cluster_centers = {label: np.mean(train_data[cluster_labels == label], axis=0) for label in large_clusters}

    # Create new merged tags
    new_labels = np.copy(cluster_labels)

    for small_cluster in small_clusters:
        small_cluster_points = train_data[cluster_labels == small_cluster]
        if len(small_cluster_points) == 0:
            continue

        # Calculate centroid of small cluster
        small_cluster_center = np.mean(small_cluster_points, axis=0)

        # Find the nearest large cluster
        distances = {label: np.linalg.norm(small_cluster_center - center) for label, center in cluster_centers.items()}
        closest_cluster = min(distances, key=distances.get)

        # Assign the points from the small cluster to the nearest cluster
        new_labels[cluster_labels == small_cluster] = closest_cluster

    return new_labels
 
 
#------------------------------------------------------------------------------------------------------    
def calculate_uncertainty_for_new_samples_hdb(new_samples, new_hdb_labels, train_data, n_neighbors=5):
    """
    Calculates the uncertainty of new samples based on fused HDBSCAN and KNN.

    Parameters:
        - new_samples: np.array with the new samples to be evaluated.
        - new_hdb_labels: Cluster labels corrected after merging small clusters.
        - train_data: np.array with the training data used in HDBSCAN.
        - n_neighbors: Number of nearest neighbors to consider.

    Returns:
        - List of uncertainty values ​​for each sample in `new_samples`.
    """
    # group points by cluster (excluding noise: label = -1)
    clusters = {label: train_data[new_hdb_labels == label] for label in np.unique(new_hdb_labels) if label != -1}

    uncertainties = []

    for sample in new_samples:
        min_uncertainty = float('inf')
        found_valid_cluster = False

        for label, points in clusters.items():
            if len(points) < n_neighbors:
                continue  

            # calculate the average distance to the K nearest neighbors
            knn = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
            distances, _ = knn.kneighbors([sample])
            avg_distance = np.mean(distances)

            # select the cluster with the least uncertainty
            min_uncertainty = min(min_uncertainty, avg_distance)
            found_valid_cluster = True

        if not found_valid_cluster:
            # fallback: global KNN distance to all train points (keeps scale stable)
            k_eff = min(n_neighbors, len(train_data))
            if k_eff >= 1:
                knn_all = NearestNeighbors(n_neighbors=k_eff).fit(train_data)
                distances, _ = knn_all.kneighbors([sample])
                min_uncertainty = float(np.mean(distances))
            else:
                min_uncertainty = 0.0

        uncertainties.append(min_uncertainty)

    return uncertainties

#------------------------------------------------------------------------------------------------------
def compute_kmeans_uq(
    train_X: np.ndarray,
    eval_X: np.ndarray,
    k: int,
    n_init: int,
    max_iter: int,
    init: str,
    knn_min_samples: int,
    random_state: int = 0,
):
    """
    Fit K-Means on train_X and compute a per sample uncertainty score for eval_X.

    Uncertainty signal
    - Base signal is the distance from each eval point to its assigned centroid.
    - Optional density proxy uses the mean distance to the K nearest neighbors
      within the assigned cluster (computed against train points in that cluster).
      
    If a cluster has too few train points for the requested knn_min_samples,
      we reduce the neighbor count: nn_k = min(knn_min_samples, cluster_size).
    - If cluster_size <= 1 (or nn_k <= 1), we fall back to centroid distance
      for those eval points.
    """
    # Ensure stable dtype/layout for downstream numeric libs.
    # This prevents errors like:
    #   ValueError("Buffer dtype mismatch, expected 'double' but got 'float'")    
    train_X = np.ascontiguousarray(train_X, dtype=np.float64)
    eval_X = np.ascontiguousarray(eval_X, dtype=np.float64)

    km = KMeans(
        n_clusters=int(k),
        init=init,
        n_init=int(n_init),
        max_iter=int(max_iter),
        random_state=int(random_state),
    )
    km.fit(train_X)

    eval_labels = km.predict(eval_X)

    # For each eval point, compute the Euclidean distance to the centroid of its assigned cluster.
    # This gives a smooth uncertainty proxy. Farther from centroid => more "uncertain".
    centers = km.cluster_centers_
    diffs = eval_X - centers[eval_labels]
    centroid_dist = np.linalg.norm(diffs, axis=1).astype(np.float64)

    uq = centroid_dist.copy()

    # If enabled, compute mean distance to K nearest train neighbors within the assigned cluster.
    # Smaller mean distance => denser neighborhood => typically lower uncertainty.
    # Larger mean distance => sparser neighborhood => typically higher uncertainty.
    #
    # Only run KNN if knn_min_samples >= 2 because k=1 is not meaningful for a mean distance.
    if knn_min_samples is not None and int(knn_min_samples) >= 2:
        k_req = int(knn_min_samples)

        # Allocate array for the KNN uncertainty, fill with NaN to detect missing values.
        uq_knn = np.empty(len(eval_X), dtype=np.float64)
        uq_knn.fill(np.nan)

        # Loop over clusters and compute KNN distances for eval points belonging to each cluster.
        for c in range(int(k)):
        
            # Eval indices assigned to cluster c.
            idx_eval = np.where(eval_labels == c)[0]
            if len(idx_eval) == 0:
                # No eval points in this cluster, nothing to compute.
                continue
                
            # Train indices that belong to cluster c (based on fitted labels).
            idx_train = np.where(km.labels_ == c)[0]
            n_train_c = len(idx_train)

            # Not enough points to compute a meaningful KNN distance: fall back.
            if n_train_c <= 1:
                uq_knn[idx_eval] = centroid_dist[idx_eval]
                continue

            # Reduce neighbor count if cluster is smaller than requested K.
            nn_k = min(k_req, n_train_c)

            # If nn_k ends up <= 1, fall back to centroid distance.
            if nn_k <= 1:
                uq_knn[idx_eval] = centroid_dist[idx_eval]
                continue

            # compute mean KNN distance inside the cluster
            nn = NearestNeighbors(n_neighbors=nn_k, metric="euclidean")
            nn.fit(train_X[idx_train])

            # Distances from each eval point in this cluster to its nn_k nearest neighbors in train.
            dists, _ = nn.kneighbors(eval_X[idx_eval], return_distance=True)
            uq_knn[idx_eval] = dists.mean(axis=1)

        # Any remaining NaNs fallback to centroid distance.
        missing = np.isnan(uq_knn)
        if np.any(missing):
            uq_knn[missing] = centroid_dist[missing]

        # Combine centroid distance and KNN density proxy.
        uq = 0.4 * centroid_dist + 0.6 * uq_knn

    # Replace NaN/inf with 0 to keep downstream correlation computations stable.
    uq = np.nan_to_num(uq, nan=0.0, posinf=0.0, neginf=0.0)
    return uq

#------------------------------------------------------------------------------------------------------
def compute_hdbscan_KNN_BACK (train_data_reduced, eval_data_reduced, 
                         min_cluster_size, min_samples, cluster_selection_epsilon, knn_min_samples ):     
    """
    Compute a KNN-based uncertainty score for test data and the two distant datasets.
    The uncertainty score is the distance to nearest cluster.

   """
    
    #==== Clustering ====
    # convert train_data_reduced to float32 (for clustering)
    train_data_reduced = np.array(train_data_reduced, dtype=np.float32)
    test_data_reduced = np.array(eval_data_reduced, dtype=np.float32)
    
    hdb = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric="euclidean",
        )
    cluster_labels = hdb.fit_predict(train_data_reduced)

    # clusters without noise
    #clusters = {label: train_data_reduced[cluster_labels == label] for label in set(cluster_labels) if label != -1}

    new_hdb_labels = merge_small_clusters_hdbscan(hdb, train_data_reduced, min_cluster_size)
  
    #==== Compute uncertainty ====
    eval_uq = calculate_uncertainty_for_new_samples_hdb(eval_data_reduced, new_hdb_labels, train_data_reduced, knn_min_samples)
   
    return eval_uq
  
# --------------------------------------------------------------------------------------------------
def compute_hdbscan_KNN(
    train_data_reduced,
    eval_data_reduced,
    min_cluster_size,
    min_samples,
    cluster_selection_epsilon,
    n_neighbors):
    """
    compute a KNN-based uncertainty score for test data and the two distant datasets.

    The uncertainty score is the mean distance to the k nearest neighbors within the
    assigned cluster for each sample.

   """


    train = np.asarray(train_data_reduced, dtype=np.float32)
    eval_ = np.asarray(eval_data_reduced, dtype=np.float32)

    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric="euclidean",
    )
    _ = hdb.fit_predict(train)

    # Must return labels aligned with `train`
    train_labels = merge_small_clusters_hdbscan(hdb, train, min_cluster_size)

    uq = calculate_uncertainty_for_new_samples_hdb_assigned(
        new_samples=eval_,
        train_labels=train_labels,
        train_data=train,
        n_neighbors=n_neighbors,
    )
    return uq

#--------------------------------------------------------------------------
def calculate_uncertainty_for_new_samples_hdb_assigned(
    new_samples,
    train_labels,
    train_data,
    n_neighbors=5,
):
    train_labels = np.asarray(train_labels)

    # Build cluster point sets excluding noise
    cluster_points = {
        label: train_data[train_labels == label]
        for label in np.unique(train_labels)
        if label != -1
    }

    if len(cluster_points) == 0:
        # No clusters at all, fallback to global KNN for everything
        k_eff = min(n_neighbors, len(train_data))
        if k_eff < 1:
            return [0.0] * len(new_samples)
        knn_all = NearestNeighbors(n_neighbors=k_eff).fit(train_data)
        dists, _ = knn_all.kneighbors(new_samples)
        return dists.mean(axis=1).astype(float).tolist()

    # Simple deterministic assignment: nearest cluster centroid
    centroids = {label: pts.mean(axis=0) for label, pts in cluster_points.items()}
    labels_list = list(centroids.keys())
    centroid_matrix = np.stack([centroids[l] for l in labels_list], axis=0)

    centroid_nn = NearestNeighbors(n_neighbors=1).fit(centroid_matrix)
    _, idx = centroid_nn.kneighbors(new_samples)
    assigned_labels = [labels_list[i[0]] for i in idx]

    uncertainties = []
    for sample, label in zip(new_samples, assigned_labels):
        pts = cluster_points[label]

        if len(pts) >= n_neighbors:
            knn = NearestNeighbors(n_neighbors=n_neighbors).fit(pts)
            dists, _ = knn.kneighbors([sample])
            uncertainties.append(float(dists.mean()))
        else:
            # fallback global KNN
            k_eff = min(n_neighbors, len(train_data))
            knn_all = NearestNeighbors(n_neighbors=k_eff).fit(train_data)
            dists, _ = knn_all.kneighbors([sample])
            uncertainties.append(float(dists.mean()))

    return uncertainties

#----------------------------------------------------------------------------------------------
def compute_knn_uncertainty(train_embeddings, test_embeddings, k=5):
    """
    Computes an uncertainty score for each vector in test_embeddings based on the average distance to its k-nearest
    neighbors in train_embeddings.
    
    Parameters:
    - train_embeddings (numpy.ndarray): Array of embeddings used as the reference set for KNN.
    - test_embeddings (numpy.ndarray): Array of embeddings for which the uncertainty scores are computed.
    - k (int): Number of nearest neighbors to consider.
    
    Returns:
    - List of uncertainty scores, one for each vector in test_embeddings.
    """
    # Fit KNN on train_embeddings
    nbrs = NearestNeighbors(n_neighbors=k).fit(train_embeddings)
    
    # Compute distances to the k-nearest neighbors for each test_embedding
    distances, _ = nbrs.kneighbors(test_embeddings)
    
    # Calculate the average distance for each test point as the uncertainty score
    uncertainty_scores = np.mean(distances, axis=1)
    
    return uncertainty_scores

#-------------------------------------------------------------------------------------------------------
def compute_KNN (train_data_reduced, test_data_reduced, distant_data_reduced,  distant02_data_reduced,
                                        test_meteors,test_bertsf1,test_perpl,
                                        d01_meteors,d01_bertsf1,d01_perpl, d02_meteors,d02_bertsf1,d02_perpl, min_samples_in_clusters):   
    """
    Compute a KNN-based uncertainty score for test data and the two distant datasets.
    Returns three pandas DataFrames containing uncertainty and evaluation metrics.

    The uncertainty score is the mean distance to the k nearest neighbors within the
    assigned cluster for each sample.
    """
    
    #================= Compute uncertainty
    test_uq = compute_knn_uncertainty(train_data_reduced, test_data_reduced, min_samples_in_clusters)

    d01_uq = compute_knn_uncertainty(train_data_reduced,distant_data_reduced, min_samples_in_clusters)

    d02_uq = compute_knn_uncertainty(train_data_reduced, distant02_data_reduced, min_samples_in_clusters)
    
    # data for visualizacion
    test_df = pd.DataFrame({
                'KDE_uq_values': test_uq, 
                'meteor_evals': test_meteors, 
                'bertscore_f1': test_bertsf1, 
                'perplexities': test_perpl})
    
    d01_df =  pd.DataFrame({
                'KDE_uq_values': d01_uq, 
                'meteor_evals': d01_meteors,
                'bertscore_f1': d01_bertsf1,
                'perplexities': d01_perpl})

    d02_df =  pd.DataFrame({
                'KDE_uq_values': d02_uq, 
                'meteor_evals': d02_meteors,
                'bertscore_f1': d02_bertsf1,
                'perplexities': d02_perpl
    })
    
    return (d01_df, d02_df, test_df)
    
    
#-------------------------------------------------------------------------------------------------------
def compute_KDE (train_data_reduced, test_data_reduced, distant_data_reduced,  distant02_data_reduced,
                                        test_meteors,test_bertsf1,test_perpl,
                                        d01_meteors,d01_bertsf1,d01_perpl, d02_meteors,d02_bertsf1,d02_perpl):  
    """
    Fit a Kernel Density Estimation model on reduced train embeddings, select bandwidth
    via cross validation, and compute uncertainty scores for test and distant splits.

    This function uses sklearn.neighbors.KernelDensity with a Gaussian kernel and
    searches over a logarithmic range of bandwidths using GridSearchCV. The fitted
    model is evaluated on the provided splits and returns negative log density values
    which are commonly used as out of distribution or uncertainty scores
    higher means more uncertain or less likely under the train distribution.
    Parameters:  
        train_data_reduced: Reduced dimension training embeddings used to fit the KDE model.
        test_data_reduced: Reduced dimension test embeddings to score.
        distant_data_reduced: First distant or out of distribution split to score.
        distant02_data_reduced: Second distant or out of distribution split to score.
    """

    # KDE: find the best estimator
    params = { 'kernel': ['gaussian'],
        "bandwidth": np.logspace(-1, 1.5, 30)}
    grid = GridSearchCV(estimator=KernelDensity(), param_grid=params, n_jobs=-1, verbose=0)
    grid.fit(train_data_reduced)
    kde = grid.best_estimator_
    
    # Evaluate KDE on test and distant data
    test_uq = -1* kde.score_samples(test_data_reduced)
    d01_uq = -1* kde.score_samples(distant_data_reduced)
    d02_uq = -1* kde.score_samples(distant02_data_reduced)

    # == Creates and return DataFrames for visualization
    test_df = pd.DataFrame({
                'KDE_uq_values': test_uq,
                'meteor_evals': test_meteors,
                'bertscore_f1': test_bertsf1,
                'perplexities': test_perpl
    })

    d01_df = pd.DataFrame({
                'KDE_uq_values': d01_uq,
                'meteor_evals': d01_meteors,
                'bertscore_f1': d01_bertsf1,
                'perplexities': d01_perpl
    })

    d02_df = pd.DataFrame({
                'KDE_uq_values': d02_uq,
                'meteor_evals': d02_meteors,
                'bertscore_f1': d02_bertsf1,
                'perplexities': d02_perpl
    })
    return (d01_df, d02_df, test_df)


#======================== Visualizations 
def plot_spearman_correlation_heatmap_bertscore (df_agg):
    """
    Generates a heatmap of Spearman correlation (BERTScore) by layer, method, dimensionality reduction, and dimensionality.
    """
    plt.figure(figsize=(12, 6))
    
    # Extract MCD and TS values avg 
    mcd_value = df_agg[df_agg["Method"] == "MCD"]["p_BERTScore_mean"].mean()
    ts_value = df_agg[df_agg["Method"] == "TS"]["p_BERTScore_mean"].mean()

    # Filter out MCD and TS methods
    df_agg = df_agg[~df_agg["Method"].isin(["MCD", "TS"])]

    df_agg["Method_Combined"] = df_agg["Method"] + " - " + df_agg["Dimensionality Reduction"] + " (" + df_agg["Dimensionality"].astype(str) + ")"
    heatmap_data = df_agg.pivot(index="Layer", columns="Method_Combined", values="p_BERTScore_mean")
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".2f")
    #plt.title("Spearman Correlation (BERTScore) by Layer, Method, Dimensionality Reduction, and Dimensionality")
    plt.title(f"Spearman Correlation (BERTScore) Across Layers and Methods — MCD Correlation: {mcd_value:.2f}, TS correlation = {ts_value:.2f}")

    plt.ylabel("Model Layer")
    plt.xlabel("Method - Dimensionality Reduction (Dimensionality)")
    plt.xticks(rotation=45, ha="right")
    plt.savefig("Spearman_correlation_BERTScore", dpi=300, bbox_inches='tight')
    plt.show()

def plot_pearson_correlation_heatmap_bertscore (df_agg):
    """
    Generates a heatmap of Pearson correlation (BERTScore) by layer, method, dimensionality reduction, and dimensionality.
    """
    plt.figure(figsize=(12, 6))
    
    # Extract MCD and TS values avg 
    mcd_value = df_agg[df_agg["Method"] == "MCD"]["pearson_p_BERTScore_mean"].mean()
    ts_value = df_agg[df_agg["Method"] == "TS"]["pearson_p_BERTScore_mean"].mean()
    #mcd02_value = df_agg[df_agg["Method"] == "MCD02"]["pearson_p_BERTScore_mean"].mean()

    # Filter out MCD and TS methods
    df_agg = df_agg[~df_agg["Method"].isin(["MCD", "TS"])]

    df_agg["Method_Combined"] = df_agg["Method"] + " - " + df_agg["Dimensionality Reduction"] + " (" + df_agg["Dimensionality"].astype(str) + ")"
    heatmap_data = df_agg.pivot(index="Layer", columns="Method_Combined", values="pearson_p_BERTScore_mean")
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".2f")
    #plt.title("Spearman Correlation (BERTScore) by Layer, Method, Dimensionality Reduction, and Dimensionality")
    plt.title(f"Pearson Correlation (BERTScore) by Layer — DeepSeek 7B |  MCD Correlation={mcd_value:.2f}, TS Correlation={ts_value:.2f}")

    plt.ylabel("Model Layer")
    plt.xlabel("Method - Dimensionality Reduction (Dimensionality)")
    plt.xticks(rotation=45, ha="right")
    plt.savefig("DeepSeek_Pearson_correlation_BERTScore.svg", format='svg', bbox_inches='tight')
    plt.show()

def plot_pearson_correlation_heatmap_meteor (df_agg):
    """
    Generates a heatmap of Pearson correlation (BERTScore) by layer, method, dimensionality reduction, and dimensionality.
    """
    plt.figure(figsize=(12, 6))
    # Extract MCD and TS values avg 
    mcd_value = df_agg[df_agg["Method"] == "MCD"]["pearson_p_METEOR_mean"].mean()
    ts_value = df_agg[df_agg["Method"] == "TS"]["pearson_p_METEOR_mean"].mean()

    # Filter out MCD and TS methods
    df_agg = df_agg[~df_agg["Method"].isin(["MCD", "TS"])]
    
    df_agg["Method_Combined"] = df_agg["Method"] + " - " + df_agg["Dimensionality Reduction"] + " (" + df_agg["Dimensionality"].astype(str) + ")"
    heatmap_data = df_agg.pivot(index="Layer", columns="Method_Combined", values="pearson_p_METEOR_mean")
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title(f"Pearson Correlation (METEOR) by Layer — DeepSeek 7b |  MCD Correlation: {mcd_value:.2f}, TS correlation = {ts_value:.2f}")
    plt.ylabel("Model Layer")
    plt.xlabel("Method - Dimensionality Reduction (Dimensionality)")
    plt.xticks(rotation=45, ha="right")
    plt.savefig("DeepSeek_Pearson_correlation_meteor.svg", format='svg', bbox_inches='tight')
    plt.show()

    
def plot_spearman_correlation_heatmap_meteor (df_agg):
    """
    Generates a heatmap of Spearman correlation (BERTScore) by layer, method, dimensionality reduction, and dimensionality.
    """
    plt.figure(figsize=(12, 6))
    # Extract MCD and TS values avg 
    mcd_value = df_agg[df_agg["Method"] == "MCD"]["p_METEOR_mean"].mean()
    ts_value = df_agg[df_agg["Method"] == "TS"]["p_METEOR_mean"].mean()

    # Filter out MCD and TS methods
    df_agg = df_agg[~df_agg["Method"].isin(["MCD", "TS"])]
    
    df_agg["Method_Combined"] = df_agg["Method"] + " - " + df_agg["Dimensionality Reduction"] + " (" + df_agg["Dimensionality"].astype(str) + ")"
    heatmap_data = df_agg.pivot(index="Layer", columns="Method_Combined", values="p_METEOR_mean")
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title(f"Spearman Correlation (METEOR) Across Layers and Methods — MCD Correlation: {mcd_value:.2f}, TS correlation = {ts_value:.2f}")
    plt.ylabel("Model Layer")
    plt.xlabel("Method - Dimensionality Reduction (Dimensionality)")
    plt.xticks(rotation=45, ha="right")
    plt.savefig("Spearman_correlation_meteor", format='svg', bbox_inches='tight')
    plt.show()

    
def filter_dataframe_dimensionality(df_agg, dimensionality, layers):
    """
    Filters df_agg by a given Method, dimensionality used, and a list of Layers.
    Returns a new filtered dataframe.
    """
    return df_agg[(df_agg["Dimensionality"] == dimensionality) & 
                  (df_agg["Layer"].isin(layers))]


def remove_outliers_using_Mahalanobis(embeddings, percentile=95):
    """
    Detects and eliminates outliers in embeddings using the Mahalanobis distance at a global level.
    
    Parameters:
    - embeddings: shape np.array (n_samples, n_features)
    - percentile: Percentile to define the Mahalanobis distance threshold (default=95)
    
    Returns:
    - np.array with embeddings without outliers.
    """

    if len(embeddings) < 2:
        return embeddings  # It cannot be calculated with less than 2 points
    
    mean = np.mean(embeddings, axis=0)
    cov = np.cov(embeddings, rowvar=False)

    # use pseudoinverse for non-invertible matrices
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov) 
        
    distances = np.array([mahalanobis(sample, mean, inv_cov) for sample in embeddings])
    threshold = np.percentile(distances, percentile)

    # filter points within the threshold
    mask = distances < threshold
    return embeddings[mask]


    
