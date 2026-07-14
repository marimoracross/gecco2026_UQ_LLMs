#!/usr/bin/env python
# coding: utf-8
"""
# # Uncertainty Quantification in Large Language Models Using Feature Space Density and Clustering
# 
# Optimization with RVEA 
# 
# Optimization using Pymoo with RVEA to tune HDBSCAN hyperparameters on top of text embeddings and the model text generation. The optimiztion objectives are the metrix BERTScore, RougeL and the UQ based on HBDSCAN-KNN.
# 
# Optimizes:
# 
# - HDBSCAN parameters
# - embedding dimensionality
# - reduction method
# 
# 2 optimization objectives (minimize signed correlation, -1 is best.):
#Objective 1 minimize correlation between UQ and BERTScore F1 over the full evaluation set
#Objective 2 minimize correlation between UQ and ROUGE L over the full evaluation set

Early stop using the hypervolume

proposed windows (wide to apply to both models)

- BERT low=-0.85 high=-0.20
- ROUGE low=-0.85 high=-0.20

low is more negative (better)
high is less negative (worse)

Normalization to [0,1]

earlystop_min_gen = 15
earlystop_patience = 8
earlystop_rel_eps = 2e-4 
earlystop_smooth_k = 3 # to judge early stopping using the raw hypervolume (3 generations).
termination n_gen = 50 


"""


# load all libraries
import os
import torch
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import math
#import seaborn as sns
import time
import statistics
#from matplotlib.colors import ListedColormap
from numpy.linalg import inv, LinAlgError
import json
import csv
import re
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict, Any
import ast



# hugging face
from huggingface_hub import notebook_login
import evaluate

# pymoo
from pymoo.core.problem import ElementwiseProblem
#from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.termination import get_termination
#from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


from evaluate import load
from tqdm.auto import tqdm

# measure time
from contextlib import contextmanager

# utilities 
from utility_kde_clustering_optimization import *

# configuration
pd.reset_option('display.max_rows')
device = "cuda" if torch.cuda.is_available() else "cpu"

pd.set_option("mode.copy_on_write", True)

import nltk
nltk.download("wordnet", download_dir="nltk_data")
nltk.download("omw-1.4", download_dir="nltk_data")




# Constants
EMB_LAYER_FIXED = 32
TRAIN_COL = "train_model_answers"
EVAL_COL = "answer"

#EMB_DIM_CHOICES = [128, 256, 512, 1024]
EMB_DIM_CHOICES = [128, 256]
REDUCTION_CHOICES = ["PCA"]

VAR_SPECS = [
    ("hdb_min_cluster_size",  5,   20,  True),
    ("hdb_min_samples",       1,   20,  True),
    ("hdb_epsilon",           0.0, 0.5, False),
    ("knn_min_samples",       5,   35,  True),
    ("emb_dim_index",         0,   len(EMB_DIM_CHOICES)   - 1, True),
    ("red_index",             0,   len(REDUCTION_CHOICES) - 1, True),
]

XL = np.array([spec[1] for spec in VAR_SPECS], dtype=float)
XU = np.array([spec[2] for spec in VAR_SPECS], dtype=float)


# Base directory for the whole experiment.
# Each repetition (seed) will write to its own subfolder under this directory.
BASE_LOG_DIR = "/work/mamora/optimizacion/checkpoint/18Llama_checkpoint_HDBSCAN_2OBJsv18_EarlyStop_RepSeed/"


# helper functions

#------------------------------Repetitions and seeds ------------------------------------------------------

def seed_everything(seed: int) -> None:
    """
    Seeds Python, NumPy, and Torch RNGs so the run is reproducible.
    This is needed because your data selection is randomized, and pymoo seed only covers the optimizer.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ---------------------------- preprocess data ---------------------------------------------------------
def _clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _to_text(x: Any) -> str:
    """
    Convert model_answer into a single string.
    Handles:
      - list or tuple: joins items with newlines
      - string that looks like a list literal: parses and joins
      - None: empty string
    """
    if x is None:
        return ""

    # If the cell is already a list (common in pandas)
    if isinstance(x, (list, tuple)):
        parts = [str(p) for p in x if p is not None]
        return "\n".join(parts)

    s = str(x)

    # If it's a string representation of a list, try to parse
    t = s.strip()
    if t.startswith("[") and t.endswith("]"):
        # JSON first
        try:
            obj = json.loads(t)
            if isinstance(obj, list):
                return "\n".join(str(p) for p in obj if p is not None)
        except Exception:
            pass
        # Python literal fallback
        try:
            obj = ast.literal_eval(t)
            if isinstance(obj, list):
                return "\n".join(str(p) for p in obj if p is not None)
        except Exception:
            pass

    return s
    
# -------------------------------------------------------------------------
def _strip_wrappers(s: str) -> str:
    """
    Removes special wrapper tokens from a text string.

    This function strips beginning- and end-of-text markers commonly used by
    language models (e.g., ``<|begin_of_text|>`` and ``<|end_of_text|>``),
    replacing them with spaces while preserving the remaining content.

    Args:
        s: Input text that may contain wrapper tokens.

    Returns:
        The input text with wrapper tokens removed.
    """
    s = re.sub(r"<\|\s*begin_of_text\s*\|>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<\|\s*end_of_text\s*\|>", " ", s, flags=re.IGNORECASE)
    return s

# -------------------------------------------------------------------------
def _strip_leading_prompt_blocks(s: str, max_passes: int = 3) -> str:
    """
    Removes leading prompt/completion wrappers from a generated text.

    This function repeatedly strips wrapper tokens (e.g.,
    ``<|begin_of_text|>`` and ``<|end_of_text|>``) and removes an initial
    ``prompt: ... completion:`` block if present. If no corresponding
    ``completion:`` marker exists, only the first ``prompt:`` line is removed.
    The process is repeated up to ``max_passes`` times to handle nested or
    duplicated prompt wrappers.
    """
    out = s
    for _ in range(max_passes):
        out = _strip_wrappers(out).lstrip()

        if not re.match(r"(?i)^prompt\s*:", out):
            break

        if re.search(r"(?i)\bcompletion\s*:", out):
            out = re.sub(
                r"(?is)^prompt\s*:.*?\bcompletion\s*:\s*",
                "",
                out,
                count=1,
            )
        else:
            out = re.sub(
                r"(?im)^prompt\s*:\s*[^\r\n]*\r?\n?",
                "",
                out,
                count=1,
            )

    return out

# -------------------------------------------------------------------------
def normalize_record(question: str, true_answer: str, model_answer: Any) -> str:
    """
    Normalizes a model-generated answer by removing prompt artifacts and
    formatting inconsistencies.

    The function converts the model output to plain text, removes special
    wrapper tokens and prompt/completion blocks, strips any remaining
    ``completion:`` labels, and removes a repeated copy of the input question
    if it appears at the beginning of the response. The resulting text is then
    normalized by collapsing excess whitespace.

    Args:
        question: The input question associated with the generated response.
        true_answer: The reference answer. Included for API consistency but not
            used during normalization.
        model_answer: The model-generated response to normalize.

    Returns:
        A cleaned version of the model-generated answer suitable for
        evaluation.
    """
    q = _clean_whitespace(str(question))
    _ = _clean_whitespace(str(true_answer))  # kept for signature symmetry

    ma = _to_text(model_answer)
    ma = _strip_wrappers(ma)

    # Remove leading prompt blocks robustly (and idempotently)
    ma = _strip_leading_prompt_blocks(ma, max_passes=5)

    # If completion label remains anywhere, drop only the label
    ma = re.sub(r"(?i)\bcompletion\s*:\s*", "", ma)

    # If the question is still literally repeated at the start, remove it once
    q_esc = re.escape(q)
    ma = re.sub(rf"(?i)^\s*(?:in medicine:\s*)?{q_esc}\s*", "", ma, count=1)

    return _clean_whitespace(ma)

# -------------------------------------------------------------------------
def normalize_records(
    questions,
    true_answers,
    model_answers,
) -> List[str]:
    """
    Normalizes a collection of model-generated answers.

    This function applies :func:`normalize_record` to each corresponding
    question, reference answer, and model-generated answer. It first verifies
    that all input collections have the same length.

    Args:
        questions: Iterable of input questions.
        true_answers: Iterable of reference answers.
        model_answers: Iterable of model-generated answers.

    Returns:
        A list containing the normalized model-generated answers.

    Raises:
        ValueError: If the input collections do not have the same length.
    """

    qs = list(questions)
    tas = list(true_answers)
    mas = list(model_answers)

    if len(qs) != len(tas) or len(qs) != len(mas):
        raise ValueError(f"Length mismatch: {len(qs)=}, {len(tas)=}, {len(mas)=}")

    return [normalize_record(q, ta, ma) for q, ta, ma in zip(qs, tas, mas)]


# -------------------------------------------------------------------------
def truncate_by_min_length(df, pred_col="prediction", ref_col="reference"):
    """
    Return two lists: (pred_trunc, ref_trunc)

    For each row:
      - coerce pred/ref to string safely (NaN -> "")
      - truncate both to min(len(pred), len(ref))
    """

    # Safe coercion: NaN/None -> "", everything else -> str(x)
    def _to_text(x):
        if x is None:
            return ""
        # pd.isna handles np.nan and pandas NA types
        if pd.isna(x):
            return ""
        return x if isinstance(x, str) else str(x)

    preds = df[pred_col].map(_to_text).tolist()
    refs  = df[ref_col].map(_to_text).tolist()

    pred_trunc = []
    ref_trunc = []
    for p, r in zip(preds, refs):
        L = min(len(p), len(r))
        pred_trunc.append(p[:L])
        ref_trunc.append(r[:L])

    return pred_trunc, ref_trunc

#-------------------------------------------------------------------------------------------------------
# Measure time to identify bottleneck
class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def log(self, rec: dict):
        rec = dict(rec)
        rec["ts"] = time.time()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

@contextmanager
def timer(name: str, timings: dict):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0)

#-------------------------------------------------------------------------------------------------------

class BestCheckpointCallback(Callback):
    """
    Callback for monitoring the optimization process, logging per-generation
    statistics, tracking the best solution, saving Pareto front snapshots, and
    performing hypervolume-based early stopping during multi-objective
    optimization.
    """
    def __init__(self,
                 log_dir,
                 eval_df,
                 train_sentence_embeddings,
                 eval_sentence_embeddings,
                 filename="best_solution.json",
                 *,
                 bert_low: float = -0.85,
                 bert_high: float = -0.20,
                 rouge_low: float = -0.90,
                 rouge_high: float = -0.30,
                 hv_ref: tuple[float, float] = (1.05, 1.05),
                 earlystop_min_gen: int = 15,
                 earlystop_patience: int = 8,
                 earlystop_rel_eps: float = 2e-4,
                 earlystop_smooth_k: int = 3):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.eval_df = eval_df
        self.train_sentence_embeddings = train_sentence_embeddings
        self.eval_sentence_embeddings = eval_sentence_embeddings
 
        self.best_file = os.path.join(self.log_dir, filename)
        self.best_combined = None

        # progress log paths
        self.progress_csv = os.path.join(self.log_dir, "progress.csv")
        self.front_dir = os.path.join(self.log_dir, "fronts")
        
        # Early stop
        self.bert_low = float(bert_low)
        self.bert_high = float(bert_high)
        self.rouge_low = float(rouge_low)
        self.rouge_high = float(rouge_high)

        self.hv_ref = np.array(hv_ref, dtype=float)

        self.earlystop_min_gen = int(earlystop_min_gen)
        self.earlystop_patience = int(earlystop_patience)
        self.earlystop_rel_eps = float(earlystop_rel_eps)
        self.earlystop_smooth_k = int(earlystop_smooth_k)

        self.hv_history: list[float] = []
        self.best_hv_smooth: float | None = None
        self.no_improve_count: int = 0

        os.makedirs(self.front_dir, exist_ok=True)

    def _append_progress(self, row):
       """
       Append one generation summary row to the progress CSV.

       Expected row format: 
          best_obj_bert
          best_obj_rouge
          best_combined
          mean_obj_bert
          mean_obj_rouge
          mean_combined
       """
       header = [
           "gen",
           "best_obj_bert", "best_obj_rouge",
           "best_combined",
           "mean_obj_bert", "mean_obj_rouge",
           "mean_combined",
           "hv",
           "hv_smooth",
           "hv_best_smooth",
           "hv_rel_improve",
           "earlystop_no_improve",
       ]

       if len(row) != len(header):
          raise ValueError(f"Progress row has {len(row)} columns but header has {len(header)}")

       file_exists = os.path.exists(self.progress_csv)
       with open(self.progress_csv, "a", newline="", encoding="utf-8") as f:
          w = csv.writer(f)
          if not file_exists:
             w.writerow(header)
          w.writerow(row)


    def notify(self, algorithm):
       """
       Callback executed by pymoo at the end of each generation.

       This method:
       1. Safely extracts the current population objectives (F) and decision variables (X).
       2. Filters out individuals with non-finite objective values (NaN or inf),
          which can appear if an evaluation failed or produced invalid metrics.
       3. Computes per-generation statistics:
          - best and mean objective values
          - best combined score (derived from objectives)
       4. Logs progress to CSV for later visualization.
       5. Tracks and saves the globally best solution found so far.
       """
       gen = int(getattr(algorithm, "n_gen", -1))

       pop = getattr(algorithm, "pop", None)
       if pop is None:
          return

       F = pop.get("F")
       X = pop.get("X")
       if F is None or X is None:
          return

       F = np.asarray(F, dtype=float)
       X = np.asarray(X, dtype=float)

       # -------------------- IMPORTANT FIX --------------------
       # Pymoo gives F as a 2D array: (population_size, n_obj).
       # For the new formulation, n_obj must be 2.
       if F.ndim != 2 or F.shape[0] == 0 or F.shape[1] != 2:
          return
       
       # count embedding dims in the raw population (includes invalid F)
       # to evaluate what is happening with 516 and 1024 dimensions
       dim_idx_raw = np.clip(np.rint(X[:, 4]).astype(int), 0, len(EMB_DIM_CHOICES) - 1)
       counts_raw = {EMB_DIM_CHOICES[i]: int(np.sum(dim_idx_raw == i)) for i in range(len(EMB_DIM_CHOICES))}
       print(f"[Gen {gen:03d}] emb_dim counts RAW: {counts_raw}")  
          
       # Remove individuals with non-finite objective values (NaN or inf).
       mask = np.isfinite(F).all(axis=1)
       if not np.any(mask):
          return

       Fv = F[mask]
       Xv = X[mask]
       
       
       #Logging of how many candidates per dim are valid per generation.
       dim_idx = np.clip(np.rint(Xv[:, 4]).astype(int), 0, len(EMB_DIM_CHOICES) - 1)
       counts = {EMB_DIM_CHOICES[i]: int(np.sum(dim_idx == i)) for i in range(len(EMB_DIM_CHOICES))}
       print(f"[Gen {gen:03d}] emb_dim counts: {counts}")


       # Objectives are minimization:
       #   f1 = corr(UQ, BERTScore_F1) over full eval set
       #   f2 = corr(UQ, ROUGE_L) over full eval set
       obj_bert  = Fv[:, 0]  # corr(UQ, BERTScore F1)
       obj_rouge = Fv[:, 1]  # corr(UQ, ROUGE L)
       combined  = 0.5 * (obj_bert + obj_rouge)

       idx_best_gen = int(np.argmin(combined))
       best_combined_gen = float(combined[idx_best_gen])

       # Fixed window normalization and clipe
       Fn = normalize_F_fixed_window(
            Fv,
            bert_low=self.bert_low,
            bert_high=self.bert_high,
            rouge_low=self.rouge_low,
            rouge_high=self.rouge_high,
       )
     
       
       print(
         f"[Gen {gen:03d}] Fn ranges: "
         f"bert [{Fn[:,0].min():.3f},{Fn[:,0].max():.3f}] "
         f"rouge [{Fn[:,1].min():.3f},{Fn[:,1].max():.3f}]"  
       )
       # clipping effect
       print(
          f"[Gen {gen:03d}] clipped%: "
          f"bert_low {np.mean(Fn[:,0] == 0.0)*100:.1f}% "
          f"bert_high {np.mean(Fn[:,0] == 1.0)*100:.1f}% "
          f"rouge_low {np.mean(Fn[:,1] == 0.0)*100:.1f}% "
          f"rouge_high {np.mean(Fn[:,1] == 1.0)*100:.1f}%"
       )
       # HV on nondominated only
       hv = hv_on_nondominated(Fn, ref_point=self.hv_ref)

       # HV smoothing
       self.hv_history.append(hv)
       k = max(1, int(self.earlystop_smooth_k))
       hv_smooth = float(np.mean(self.hv_history[-k:]))

       #-------------------------------------early stop
       hv_rel_improve = 0.0

       # Do not early stop before earlystop_min_gen
       if gen >= self.earlystop_min_gen:

          if self.best_hv_smooth is None:
                self.best_hv_smooth = hv_smooth
                self.no_improve_count = 0
          else:
                denom = max(abs(self.best_hv_smooth), 1e-12)
                hv_rel_improve = (hv_smooth - self.best_hv_smooth) / denom

                if hv_rel_improve > self.earlystop_rel_eps:
                    self.best_hv_smooth = hv_smooth
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1
          if self.no_improve_count >= self.earlystop_patience:
             print(
                f"[Gen {gen:03d}] Early stop triggered. "
                f"no_improve_count={self.no_improve_count} "
                f"best_hv_smooth={self.best_hv_smooth:.6f} "
                f"hv_smooth={hv_smooth:.6f} "
                f"rel_improve={hv_rel_improve:.6e}",
                flush=True
             )

             raise RuntimeError("EARLY_STOP")

 
       print(
          f"[Gen {gen:03d}] best obj_bert={obj_bert[idx_best_gen]:.4f} "
          f"obj_rouge={obj_rouge[idx_best_gen]:.4f} "
          f"combined={best_combined_gen:.4f}"
       )

       # -------------------- Stats using Fv (filtered) --------------------
       best_obj_bert  = float(np.min(obj_bert))
       best_obj_rouge = float(np.min(obj_rouge))

       mean_obj_bert  = float(np.mean(obj_bert))
       mean_obj_rouge = float(np.mean(obj_rouge))

       mean_combined  = float(np.mean(combined))

       # -------------------- Progress logging --------------------
       # Update the CSV schema accordingly. Suggested columns per row:
       try:
           self._append_progress([
             gen,
             best_obj_bert, best_obj_rouge,
             best_combined_gen,
             mean_obj_bert, mean_obj_rouge,
             mean_combined,
             hv,
             hv_smooth,
             (float(self.best_hv_smooth) if self.best_hv_smooth is not None else float("nan")),
             hv_rel_improve,
             self.no_improve_count,
           ])
       except Exception as e:
           print(f"[Gen {gen:03d}] progress logging failed: {e}")

       # -------------------- Save Pareto snapshot --------------------
       try:
        np.savez_compressed(
            os.path.join(self.front_dir, f"front_gen_{gen:03d}.npz"),
            F=Fv,
            X=Xv,
            obj_bert=obj_bert,
            obj_rouge=obj_rouge,
            combined=combined,
            mask=mask,
            F_raw=F,
            X_raw=X,
        )
       except Exception as e:
        print(f"[Gen {gen:03d}] pareto snapshot failed: {e}")

       # -------------------- Global best logic --------------------
       if (self.best_combined is None) or (best_combined_gen < self.best_combined):
           self.best_combined = best_combined_gen

           x_star = Xv[idx_best_gen]
           hdb_params, emb_params, reduction_method = decode_x(x_star)

           F_star, corr_details, timings = evaluate_configuration(
               hdb_params=hdb_params,
               emb_params=emb_params,
               df_eval=self.eval_df,
               train_sentence_embeddings=self.train_sentence_embeddings,
               eval_sentence_embeddings=self.eval_sentence_embeddings,
               reduction_method=reduction_method,
           )

           best_solution = {
              "generation": gen,
              "x_star": x_star.tolist(),
              "F": [float(F_star[0]), float(F_star[1])],
              "obj_bert": float(F_star[0]),
              "obj_rouge": float(F_star[1]),
              "combined_mean": float(0.5 * (F_star[0] + F_star[1])),
              "corr_details": corr_details,  # {"all": {"bert": ..., "rouge": ...}}
               "hdb_params": hdb_params,
              "embedding_params": emb_params,
              "reduction_method": reduction_method,
              "timings_s": timings,
           }

           tmp_path = self.best_file + ".tmp"
           with open(tmp_path, "w", encoding="utf-8") as f:
               json.dump(best_solution, f, indent=2)
           os.replace(tmp_path, self.best_file)

           print(f"[Gen {gen:03d}] *** NEW GLOBAL BEST SAVED to {self.best_file} ***")

#-------------------------------------------------------------------------------------------------------
def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the Pearson correlation coefficient between two arrays while
    safely handling invalid or insufficient data.

    Non-finite values (NaN or inf) are ignored. If fewer than three valid
    paired observations remain, or if the computed correlation is not finite,
    the function returns ``0.0``.

    Args:
        x: First input array.
        y: Second input array.

    Returns:
        The Pearson correlation coefficient between the valid elements of
        ``x`` and ``y``, or ``0.0`` if the correlation cannot be reliably
        computed.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return 0.0

    r = float(np.corrcoef(x[m], y[m])[0, 1])
    return r if np.isfinite(r) else 0.0
    
    
#-------------------------------------------------------------------------------------------------------
def sanity_checks(eval_uq, quality, name="quality"):
    """
    Performs basic validation checks on uncertainty estimates and quality
    scores by computing their Pearson correlation.

    The function verifies that both inputs have the same length, removes
    non-finite values, and reports the correlation between uncertainty scores
    and a quality metric. Since higher uncertainty is expected to correspond
    to lower-quality predictions, a positive correlation triggers a warning.

    Args:
        eval_uq: Array of uncertainty estimates.
        quality: Array of quality scores, where higher values indicate better
            predictions.
        name: Name of the quality metric, used in diagnostic messages.

    Raises:
        ValueError: If the input arrays have different lengths.
    """

    eval_uq = np.asarray(eval_uq, dtype=float).reshape(-1)
    quality = np.asarray(quality, dtype=float).reshape(-1)

    if eval_uq.shape[0] != quality.shape[0]:
        raise ValueError(f"[sanity_checks] size mismatch: eval_uq={eval_uq.shape} vs {name}={quality.shape}")

    m = np.isfinite(eval_uq) & np.isfinite(quality)
    if int(m.sum()) < 3:
        print(f"[sanity_checks] not enough finite pairs for correlation: {int(m.sum())}")
        return

    r = float(np.corrcoef(eval_uq[m], quality[m])[0, 1])
    print(f"[sanity_checks] corr(eval_uq, {name}) = {r:.6f}")

    # If "higher UQ means worse answers", then corr(UQ, quality) should be negative
    # where quality is 1 for correct, 0 for incorrect (or any score where higher is better).
    if r > 0:
        print(f"[sanity_checks] WARNING: correlation is positive, check if your UQ sign is flipped.")    
#-------------------------------------------------------------------------------------------------------


def evaluate_configuration(
    hdb_params,
    emb_params,
    df_eval,
    train_sentence_embeddings,
    eval_sentence_embeddings,
    reduction_method,
):
    """
    Evaluate one configuration with the defined objectives.

    Objectives (minimization):
       Objective 1 minimize correlation between UQ and BERTScore F1 over the full evaluation set
       Objective 2 minimize correlation between UQ and ROUGE L over the full evaluation set
    Returns
      F: np.ndarray shape (2,)
      corr_bundle: dict with results
      timings: dict with per block timings
    """
    timings = {}

    # Penalty defaults
    f1 = f2 = 1e3

    # Default correlation bundle, always defined
    corr_bundle = {"all": {"bert": 0.0, "rouge": 0.0}}

    with timer("total", timings):
        try:
            #predictions = df_eval["model_answer"].tolist()
            #eval_pred_df = pd.DataFrame({"answer": predictions})

            #with timer("eval_embeddings", timings):
            #    eval_embeddings = generate_embeddings_simple(
            #        eval_pred_df, model, tokenizer, device,
            #        EMB_LAYER_FIXED, answer_length, "answer"
            #    )

            with timer("scaling", timings):
                train_embeddings, eval_embeddings = standard_scaler_normalization(
                    train_sentence_embeddings, eval_sentence_embeddings
                )

            with timer("reduction", timings):
                train_data_reduced, eval_data_reduced = reduce_data_dimensionality(
                    train_embeddings, eval_embeddings, emb_params["dimension"], reduction_method
                )

            with timer("uq_hdbscan_knn", timings):
                eval_uq = compute_hdbscan_KNN(
                    train_data_reduced,
                    eval_data_reduced,
                    hdb_params["min_cluster_size"],
                    hdb_params["min_samples"],
                    hdb_params["cluster_selection_epsilon"],
                    hdb_params["knn_min_samples"],
                )

            # to avoid NaN
            # Force eval_uq to be finite and 1D
            eval_uq = np.asarray(eval_uq, dtype=float).reshape(-1)
            eval_uq = np.nan_to_num(eval_uq, nan=0.0, posinf=0.0, neginf=0.0)

            # Quality vectors (must be finite and same length)
            f1_vec = np.asarray(df_eval["bertscore_f1"], dtype=float).reshape(-1)
            rL_vec = np.asarray(df_eval["rougeL"], dtype=float).reshape(-1)
            f1_vec = np.nan_to_num(f1_vec, nan=0.0, posinf=0.0, neginf=0.0)
            rL_vec = np.nan_to_num(rL_vec, nan=0.0, posinf=0.0, neginf=0.0)

            n = min(len(eval_uq), len(f1_vec), len(rL_vec))
            if n == 0:
                raise ValueError("Empty vectors for correlation")

            eval_uq = eval_uq[:n]
            f1_vec = f1_vec[:n]
            rL_vec = rL_vec[:n]

            # Compute correlations over the whole concatenated eval set (no splits)
            with timer("corr", timings):
              corr_bert  = safe_corr(eval_uq, f1_vec)
              corr_rouge = safe_corr(eval_uq, rL_vec)

            # Store raw correlations for logging
            corr_bundle = {
               "all": {
                "bert": float(corr_bert),
                "rouge": float(corr_rouge),
               }}
            
            # Two simple minimization objectives
            f1 = corr_bert     # Objective 1: minimize corr(UQ, BERTScore_F1)
            f2 = corr_rouge    # Objective 2: minimize corr(UQ, ROUGE_L)

        except Exception as e:
            timings["exception"] = repr(e)

    F = np.array([f1, f2], dtype=float)
    return F, corr_bundle, timings


#---------------------------------------------------------------------------------------------------
def decode_x(x):
    """
    Decodes an optimization solution vector into configuration parameters.

    The function converts a continuous decision vector produced by the
    optimizer into valid HDBSCAN, embedding, and dimensionality reduction
    parameters by clipping values to their allowed ranges and rounding
    integer-valued variables as needed.

    Args:
        x: Decision vector representing a candidate solution.

    Returns:
        A tuple containing:
            - hdb_params: Dictionary of HDBSCAN and k-nearest neighbor
              parameters.
            - emb_params: Dictionary specifying the embedding dimension and
              fixed embedding layer.
            - reduction_method: Selected dimensionality reduction method.
    """

    params = {}
    for i, (name, low, high, is_int) in enumerate(VAR_SPECS):
        v = x[i]
        if is_int:
            v = int(np.clip(np.round(v), low, high))
        else:
            v = float(np.clip(v, low, high))
        params[name] = v

    hdb_params = {
        "min_cluster_size": params["hdb_min_cluster_size"],
        "min_samples": params["hdb_min_samples"],
        "cluster_selection_epsilon": params["hdb_epsilon"],
        "knn_min_samples": params["knn_min_samples"],
    }

    emb_dim = EMB_DIM_CHOICES[params["emb_dim_index"]]
    reduction_method = REDUCTION_CHOICES[params["red_index"]]

    emb_params = {"dimension": emb_dim, "layer": EMB_LAYER_FIXED}
    return hdb_params, emb_params, reduction_method

#---------------------------------------------------------------------------------------------------
def data_for_experiment(root_folder_path,
                          num_samples,num_samples_testing,num_samples_distant,layer_embedding, random_state=1):
    """
    Prepare data for experiments. Randomly select data  (test, train, d01 and d02), generate embeddings for train data.

    return:
        eval_df: evala data (concat of test, d01 and d02)
        train_sentence_embeddings.
    """
    # randomly select samples 
    train_samples_df, test_samples_df, dist01_samples_df, dist02_samples_df = load_data_random(root_folder_path,
                          num_samples,num_samples_testing,num_samples_distant,layer_embedding, random_state)     

    # concat eval data
    
    test_block = (
       test_samples_df[["question", "test_best_answers", "test_model_answers", "bertscore_f1", "meteor_evals"]]
       .rename(columns={"test_best_answers": "answer", "test_model_answers": "model_answer"})
    )
    test_block["split"] = "test"

    d01_block = (
       dist01_samples_df[["question", "distant_true_answers", "distant_model_answers", "bertscore_f1", "meteor_evals"]]
       .rename(columns={"distant_true_answers": "answer", "distant_model_answers": "model_answer"})
    )
    d01_block["split"] = "d01"

    d02_block = (
       dist02_samples_df[["question", "distant_true_answers", "distant_model_answers", "bertscore_f1", "meteor_evals"]]
       .rename(columns={"distant_true_answers": "answer", "distant_model_answers": "model_answer"})
    )
    d02_block["split"] = "d02"

    eval_df = pd.concat([test_block, d01_block, d02_block], ignore_index=True)
    
    return train_samples_df,  eval_df

#----------------------------------------------------------------------------------------------------
class HDBSCANUQProblem(ElementwiseProblem):
    """
    Multi-objective optimization problem for tuning HDBSCAN-based uncertainty
    estimation parameters.

    This class defines the search space and evaluation data required by pymoo
    to optimize clustering, embedding, and dimensionality reduction
    configurations using two objective functions.
    """

    def __init__(
        self,
        eval_df,
        train_sentence_embeddings,
        eval_sentence_embeddings,
        log_dir,
    ):
        self.eval_df = eval_df
        self.train_sentence_embeddings = train_sentence_embeddings
        self.eval_sentence_embeddings = eval_sentence_embeddings
        
        self.eval_counter = 0
        self.timing_logger = JsonlLogger(os.path.join(log_dir, "timings_eval.jsonl"))

        super().__init__(
            n_var=len(VAR_SPECS),
            n_obj=2,    # 2 objectives
            n_constr=0 ,
            xl=XL,
            xu=XU,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate a single candidate solution (one individual) for pymoo/NSGA-II.

        Parameters 
        x : np.ndarray
           Decision vector of shape (n_var,). This contains the raw optimization variables
           defined by VAR_SPECS (e.g., HDBSCAN params, embedding dimension index, reduction index).
        out : dict
           Dictionary where pymoo expects the evaluation results to be written.
           The required key is:
             - out["F"]: objective vector of shape (n_obj,)
           Since this is a minimization problem, smaller values are better.
        *args, **kwargs
           Unused but required by pymoo's API.

        """
        self.eval_counter += 1

        hdb_params, emb_params, reduction_method = decode_x(x)

        # Default penalty: large values because pymoo minimizes  
        F_penalty = np.array([1e3, 1e3], dtype=float)


        try:
           F, corr_bundle, timings = evaluate_configuration(
              hdb_params=hdb_params,
              emb_params=emb_params,
              df_eval=self.eval_df,
              train_sentence_embeddings=self.train_sentence_embeddings,
              eval_sentence_embeddings=self.eval_sentence_embeddings,
              reduction_method=reduction_method,
            )

           F = np.asarray(F, dtype=float)

           # Force finiteness. If invalid, assign penalty
           if F.shape != (2,) or (not np.isfinite(F).all()):
              F = F_penalty
              timings = dict(timings) if isinstance(timings, dict) else {}
              timings["non_finite_F"] = True
              corr_bundle = {"all": {"bert": 0.0, "rouge": 0.0}}

        except Exception as e:
           F = F_penalty
           corr_bert, corr_rouge = 0.0, 0.0
           timings = {"exception": repr(e)}
           
        
        # Always set out["F"] to finite values
        out["F"] = F

        # log for debugging/bottlenecks
        if hasattr(self, "timing_logger") and self.timing_logger is not None:
           self.timing_logger.log({
               "eval_id": int(self.eval_counter),
               "F": [float(F[0]), float(F[1])],
               "corr_bundle": corr_bundle,
               "timings_s": timings,
               "hdb_params": hdb_params,
               "emb_params": emb_params,
               "reduction_method": reduction_method,
       })

#-------------------------------------------------------------------
def precompute_train_embeddings(train_samples_df, model, tokenizer, device, answer_length):
    emb = generate_embeddings_simple(
        train_samples_df,
        model,
        tokenizer,
        device,
        EMB_LAYER_FIXED,
        answer_length,
        TRAIN_COL,
    )
    return emb


#-----------------------------------------------------------------
def to_text(x):
    # Handles NaN, None, floats, etc.
    if x is None:
        return ""
    # pandas NaN is float and != itself
    try:
        if isinstance(x, float) and (x != x):
            return ""
    except Exception:
        pass
    return str(x)
    

def compute_rougeL_per_sample(predictions, references, use_stemmer=True):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)
    scores = []

    for pred, ref in zip(predictions, references):
        pred_txt = to_text(pred)
        ref_txt = to_text(ref)

        s = scorer.score(ref_txt, pred_txt)
        scores.append(float(s["rougeL"].fmeasure))

    return scores
    
#------------------------------------------------------------------
# Early stoping     
#------------------------------------------------------------------
def normalize_F_fixed_window(F: np.ndarray,
                             bert_low: float,
                             bert_high: float,
                             rouge_low: float,
                             rouge_high: float) -> np.ndarray:
    """
    Normalize objectives F (corr values) into [0, 1] using fixed, experiment specific windows.

    For minimization problems:
      lower corr (more negative) is better.
    We map:
      low  -> 0.0  (good)
      high -> 1.0  (bad)
    Values outside the window are clipped to [0, 1].

    Parameters
    ----------
    F : array, shape (n_points, 2)
        Raw objectives: [corr(UQ, BERTScoreF1), corr(UQ, ROUGEL)].
    bert_low, bert_high, rouge_low, rouge_high : float
        Fixed normalization bounds per objective.

    Returns
    -------
    Fn : array, shape (n_points, 2)
        Normalized objectives in [0, 1].
    """
    Fn = np.empty_like(F, dtype=float)

    denom_bert = (bert_high - bert_low)
    denom_rouge = (rouge_high - rouge_low)

    if denom_bert <= 0 or denom_rouge <= 0:
        raise ValueError("Normalization bounds invalid: high must be greater than low")

    Fn[:, 0] = (F[:, 0] - bert_low) / denom_bert
    Fn[:, 1] = (F[:, 1] - rouge_low) / denom_rouge

    return np.clip(Fn, 0.0, 1.0)


def hv_on_nondominated(Fn: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Compute hypervolume on the nondominated set only (minimization space).

    Fn must be normalized so that "good" is near 0 and "bad" near 1.
    ref_point must be slightly worse than all expected points, eg [1.05, 1.05].
    """
    if Fn.size == 0:
        return float("nan")

    nd = NonDominatedSorting().do(Fn, only_non_dominated_front=True)
    Fn_nd = Fn[nd] if nd is not None and len(nd) > 0 else Fn

    hv = HV(ref_point=ref_point)
    return float(hv.do(Fn_nd))



#==================== Main =======================================

def main(seed: int, base_log_dir: str) -> None:
    seed_everything(seed)

    run_dir = os.path.join(base_log_dir, f"seed_{seed:03d}")
    os.makedirs(run_dir, exist_ok=True)

    # Optional 
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"seed": seed, "run_dir": run_dir, "created_at_unix": time.time()},
            f,
            indent=2,
        )
        
    # Variables    
    based_model_name = "meta-llama/Llama-3.1-8B"
    model_name = "mariamoracrossitcr/Llama-3.1-8B-medquad-V2"
    model_name_tokenizer = "meta-llama/Llama-3.1-8B"
    answer_length = 200

    root_folder_path = "/data/mamora/medquad/"
    layer_embedding = "tscaling/"  # not in used just for backward compatibility

    num_samples = 5000
    num_samples_testing = 600
    num_samples_distant = 75


    #----------------------------------
    model, tokenizer = load_model_and_tokenizer(
        based_model_name, model_name, model_name_tokenizer, answer_length, device
    )

    # load samples ramdomly
    train_samples_df, eval_df = data_for_experiment(
        root_folder_path, num_samples, num_samples_testing, num_samples_distant, layer_embedding, seed
    )

    # Improve the use of resources
    train_sentence_embeddings = precompute_train_embeddings(
        train_samples_df, model, tokenizer, device, answer_length
    )

    # normalize prediction delete question and prompt from model answer. 
    eval_df["model_answer"] = normalize_records(eval_df["question"].tolist(), eval_df["answer"].tolist(), 
                                    eval_df["model_answer"].tolist())
                          
    # compute rouge for test, D01, D02    
    references = eval_df["answer"].tolist()
    predictions = eval_df["model_answer"].tolist()                       
    eval_df["rougeL"] = compute_rougeL_per_sample(predictions, references, use_stemmer=True)
    
    #"eval_embeddings
    eval_pred_df = pd.DataFrame({"answer": predictions})
    eval_sentence_embeddings = generate_embeddings_simple(
                    eval_pred_df, model, tokenizer, device,
                    EMB_LAYER_FIXED, answer_length, "answer"
                )
    
    eval_sentence_embeddings = np.asarray(eval_sentence_embeddings, dtype=np.float32)
    train_sentence_embeddings = np.asarray(train_sentence_embeddings, dtype=np.float32)
    
    problem = HDBSCANUQProblem(
        eval_df=eval_df,
    	train_sentence_embeddings=train_sentence_embeddings,
    	eval_sentence_embeddings=eval_sentence_embeddings,
    	log_dir=run_dir,
    )

    # to save best soluction and Pareto front
    callback = BestCheckpointCallback(
        log_dir=run_dir,
        eval_df=eval_df,
        train_sentence_embeddings=train_sentence_embeddings,
        eval_sentence_embeddings=eval_sentence_embeddings,
        filename="best_solution.json",
        bert_low=-0.85,
        bert_high=-0.20,
        rouge_low=-0.85,
        rouge_high=-0.20,
        hv_ref=(1.05, 1.05),
        earlystop_min_gen=8,
        earlystop_patience=6,  # Stops after N consecutive non improvements
        earlystop_rel_eps=2e-3, #Ignores tiny relative improvements. 
        earlystop_smooth_k=5,   # Uses last N gens average HV     
    )
    
    """
    Definitives values:
        earlystop_min_gen=8,
        earlystop_patience=6,  # Stops after n consecutive non improvements
        earlystop_rel_eps=2e-3, #Ignores tiny relative improvements. 
        earlystop_smooth_k=5,   # Uses last y gens average HV     
 
    """
    
    """
    The RVEA configuration follows the recommendations of Cheng et al. for decomposition-based multi-objective 
    evolutionary optimization. Reference directions were generated using the Das and Dennis systematic approach 
    with 59 partitions, yielding 60 uniformly distributed reference vectors for the two-objective problem. 
    The population size was set equal to the number of reference directions, as recommended in the original 
    RVEA formulation. The optimization was allowed to proceed for up to 50 generations, with an additional
    hypervolume-based early stopping criterion to terminate the search when the Pareto front no 
    longer exhibited significant improvement.
    
    Cheng, R., Jin, Y., Olhofer, M., & Sendhoff, B. (2016). A Reference Vector Guided Evolutionary 
    Algorithm for Many-Objective Optimization. IEEE Transactions on Evolutionary Computation, 20(5), 773–791.
    """
    
    # RVEA needs reference directions for n_obj objectives
    n_obj = problem.n_obj

    # For 2 objectives, "das-dennis" gives a clean spread of directions.
    # n_partitions controls how many directions are used (more partitions = more directions).
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=59)

    # Recommended: set pop_size equal to number of reference directions
    algorithm = RVEA(
        ref_dirs=ref_dirs,
        pop_size=len(ref_dirs),
    )

   
    termination = get_termination("n_gen", 50)

    try:
       res = minimize(
          problem,
          algorithm,
          termination,
          seed=seed,
          verbose=True,
          callback=callback,
       )
    except RuntimeError as e:
       if str(e) == "EARLY_STOP":
          print("Early stop caught, ending optimization.", flush=True)
          res = None
       else:
          raise
    

#------------------------------ main call ---------------------------------------
# Receives more than one seed 
# Use form: 
#    python run.py --seed_start 1 --seed_end 10

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run one or more full RVEA repetitions with given seed(s)."
    )

    parser.add_argument("--seed", type=int, nargs="*", default=None,
                        help="Explicit seeds, e.g. --seed 1 2 3")

    parser.add_argument("--seed_start", type=int, default=None,
                        help="Range start (inclusive), e.g. 1")
    parser.add_argument("--seed_end", type=int, default=None,
                        help="Range end (inclusive), e.g. 10")

    parser.add_argument(
        "--base_log_dir",
        type=str,
        default=BASE_LOG_DIR,
        help="Experiment root directory. Each seed writes to base_log_dir/seed_XXX/",
    )

    args = parser.parse_args()

    seeds = []
    if args.seed:
        seeds.extend(args.seed)
    if args.seed_start is not None or args.seed_end is not None:
        if args.seed_start is None or args.seed_end is None:
            raise SystemExit("Provide both --seed_start and --seed_end")
        seeds.extend(range(args.seed_start, args.seed_end + 1))

    if not seeds:
        raise SystemExit("Provide --seed or (--seed_start and --seed_end)")

    for s in seeds:
        main(seed=s, base_log_dir=args.base_log_dir)

