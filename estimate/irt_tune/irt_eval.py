import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import requests
from bs4 import BeautifulSoup
import json
import hashlib
from tqdm.auto import tqdm
import argparse
import numpy as np
import pandas as pd
from datasets.utils.logging import set_verbosity_error
from datasets import disable_progress_bars, get_dataset_split_names
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from py_irt.dataset import Dataset as IRTDataset
from py_irt.models import (
    OneParamLog,
    TwoParamLog,
    ThreeParamLog,
    FourParamLog,
    OneParamWithGuessLog,
)
from skelo.model.elo import EloEstimator
from skelo.model.glicko2 import Glicko2Estimator


set_verbosity_error()
disable_progress_bars()

# Original dataset names (dataset_name, config_name, dataset_split)
ORIGINAL_DATASET_NAMES = {
    "ARC": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "GSM8K": ("gsm8k", "main", "test"),
    "HellaSwag": ("Rowan/hellaswag", None, "validation"),
    "Winogrande": ("winogrande", "winogrande_debiased", "validation"),
}


# Functions to preprocess original dataset questions
# Which compute the ``example'' column of the corresponding result dataset
# This is necessary to ensure the order of the questions is the same in both results and original datasets
ORIGINAL_DATASET_PREPROCESS_FUNCS = {
    "GSM8K": lambda sample: {"example": sample["question"]},
    "ARC": lambda sample: {
        "example": ("Question: " + sample["question"] + "\nAnswer:")
    },
    "Winogrande": lambda sample: {"example": sample["sentence"]},
    "HellaSwag": lambda sample: {
        "example": sample["activity_label"]
        + ": "
        + sample["ctx_a"]
        + " "
        + sample["ctx_b"].capitalize()
    },
}

# The following models are ignored because of some sanity checks by human inspection
# Note we do not consider models with zero hub likes in human inspection
IGNORE_MODEL_SHA = [
    # These models crawled accuracy calculated from eval results do not match the leaderboard's
    "a5e85ae1941e31bb705adbcafce9b0dfd6f3a48b",
    "f6c9be08f16fe4d3a719bee0a4a7c7415b5c65df",
    "24fb8e1e9cc78e0aa7ef154b026c4a83296e3fc4",
    "89f594b32aea9bf5de0abe3877f20ff302549934",
    "24ebae726954e4c1f24a8b2cbe0ca863012a7338",
    "b76e2592849352c5073ebddec5748975f16e4895",
]


# Get the leaderboard data from the huggingface leaderboard page
def get_leaderboard_dataframe():
    url = "https://huggingfaceh4-open-llm-leaderboard.hf.space/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    # We can get the json format data from the second script element
    script_elements = soup.find_all("script")
    json_data = json.loads(str(script_elements[1])[31:-10])
    # component_index sometimes changes when they update the space
    # We can use this "for" loop to avoid changing component index manually
    for component_index in range(10, 50, 1):
        try:
            result_list = []
            i = 0
            while True:
                try:
                    results = json_data["components"][component_index]["props"][
                        "value"
                    ]["data"][i]
                    columns = json_data["components"][component_index]["props"][
                        "headers"
                    ]
                    try:
                        results_json = {"T": results[0], "Model": results[-1]}
                        # If there are less than 15 columns (this number can definetly change)
                        # We know that we are trying wrong component index, so breaking loop to try next component index.
                        if len(columns) < 15:
                            break
                        for col_index, col_name in enumerate(columns[2:-1], start=2):
                            results_json[col_name] = results[col_index]
                    # Wrong component index, so breaking loop to try next component index.
                    # More than one component index can give you some results but we must find the right component index to get all results we want.
                    except IndexError:
                        break
                    result_list.append(results_json)
                    i += 1
                # No rows to extract so return the list
                # We know it is the right component index because we didn't break out of loop on the other exception.
                except IndexError:
                    return pd.DataFrame(result_list)
        except (KeyError, TypeError):
            continue
    return pd.DataFrame(result_list)


# Get the original dataset sorted to the same order as the eval results
def get_sorted_original_dataset(dataset_name):
    original_dataset = load_dataset(
        ORIGINAL_DATASET_NAMES[dataset_name][0],
        ORIGINAL_DATASET_NAMES[dataset_name][1],
        split=ORIGINAL_DATASET_NAMES[dataset_name][2],
    )
    return (
        original_dataset.map(ORIGINAL_DATASET_PREPROCESS_FUNCS[dataset_name])
        .sort("example")
        .add_column("sorted_index", list(range(len(original_dataset))))
    )


# Load the evaluation results and filter them
def load_and_filter_eval_result(
    minimum_hub_likes=None,
    pretrained_only=False,
    available_on_hub_only=False,
    not_merged_only=True,
    not_flagged_only=True,
):
    df_dict = {}
    for dataset_name in ["ARC", "GSM8K", "HellaSwag", "Winogrande"]:
        # Load the result dataset
        ds_list = []
        # Load all 16 splits, each split is a parquet file
        for idx in range(16):
            ds_list.append(
                load_dataset(
                    "parquet",
                    data_files=f"data/{dataset_name}/leaderboard_performance/splits/{idx}.parquet",
                )["train"]
            )
        df_dict[dataset_name] = concatenate_datasets(ds_list).to_pandas()

    # There are some duplicates in the datasets, we need to drop them first
    for name, df in df_dict.items():
        df.drop_duplicates(
            subset=["model_sha"],
            keep="first",
            inplace=True,
        )
    # Convert "model_sha" to the index for each DataFrame and rename the "accuracies" column
    for name, df in df_dict.items():
        df.set_index("model_sha", inplace=True)
        df.rename(columns={"accuracies": f"eval_results_{name.lower()}"}, inplace=True)
    # Assuming all DataFrames should have the same "model_sha" entries for a valid intersection
    eval_results = pd.concat(df_dict.values(), axis=1, join="inner").reset_index()
    # Columns to keep from the first DataFrame only
    cols_to_keep_once = ["model_name", "eval_timestamp"]
    # Identify duplicate columns after the first occurrence
    duplicated_cols = eval_results.columns.duplicated()
    # Mark the first occurrence of the specific columns to keep as not duplicated
    for col in cols_to_keep_once:
        first_occurrence_idx = eval_results.columns.get_loc(col)
        if isinstance(first_occurrence_idx, (np.ndarray, list)):
            # If get_loc returns multiple positions for duplicates, mark the first as not duplicated
            duplicated_cols[first_occurrence_idx[0]] = False
    # Drop columns marked as duplicated except for the first occurrence of specified columns
    eval_results = eval_results.loc[:, ~duplicated_cols]
    # Finally we drop models that do not pass sanity checks by human inspection
    eval_results = eval_results[~eval_results["model_sha"].isin(IGNORE_MODEL_SHA)]

    # Filter the result dataset
    leaderboard = get_leaderboard_dataframe()
    # Fix issue with the "Hub License" column
    leaderboard["Hub License"] = leaderboard["Hub License"].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )
    # Filter by model type
    if pretrained_only:
        leaderboard = leaderboard[leaderboard["Type"] == "pretrained"]
    # Filter by minimum hub likes
    if minimum_hub_likes is not None:
        leaderboard = leaderboard[leaderboard["Hub ❤️"] >= minimum_hub_likes]
    # Filter by available on hub status
    if available_on_hub_only:
        leaderboard = leaderboard[leaderboard["Available on the hub"] == True]
    # Filter by merged status
    if not_merged_only:
        leaderboard = leaderboard[leaderboard["Merged"] == False]
    # Filter by flagged status
    if not_flagged_only:
        leaderboard = leaderboard[leaderboard["Flagged"] == False]
    # There are some duplicates in the datasets, we need to drop them last
    leaderboard.drop_duplicates(
        subset=["Model sha"],
        keep="first",
        inplace=True,
    )

    # Concatenate the filtered leaderboard with the result dataset
    eval_results = eval_results.merge(
        leaderboard, left_on="model_sha", right_on="Model sha", how="inner"
    )
    # Drop the columns that are not needed
    eval_results = eval_results.drop(columns=["T", "Model", "Model sha", "Weight type"])
    # Rename the columns
    eval_results = eval_results.rename(
        columns={
            "Average ⬆️": "acc_avg",
            "ARC": "acc_arc",
            "HellaSwag": "acc_hellaswag",
            "MMLU": "acc_mmlu",
            "TruthfulQA": "acc_truthfulqa",
            "Winogrande": "acc_winogrande",
            "GSM8K": "acc_gsm8k",
            "Type": "model_type",
            "Architecture": "model_architecture",
            "Precision": "model_precision",
            "Hub ❤️": "n_hub_likes",
            "Hub License": "hub_license",
            "#Params (B)": "n_params",
            "Available on the hub": "available_on_hub",
            "Merged": "is_merged",
            "Flagged": "is_flagged",
            "MoE": "is_moe",
        }
    )
    return eval_results


# Convert the evaluation results to game records like format
def eval_results_to_game_records(eval_results, dataset_name):
    num_problems = len(eval_results["eval_results_" + dataset_name.lower()].iloc[0])
    problem_idx_to_sha = {
        idx: hashlib.sha1(str(idx).encode()).hexdigest() for idx in range(num_problems)
    }
    # Convert eval results to match records
    game_records = []
    for _, row in eval_results.iterrows():
        # Time stamp maybe useful depending on the rating system
        eval_timestamp = row["eval_timestamp"]
        # Model SHA is the ``player name'' or each model
        model_sha = row["model_sha"]
        for idx, result in enumerate(row["eval_results_" + dataset_name.lower()]):
            # Problem index SHA is the ``player name'' or each dataset sample
            # TODO:
            winner = (
                f"model_{model_sha}" if result else f"problem_{problem_idx_to_sha[idx]}"
            )
            loser = (
                f"model_{model_sha}"
                if not result
                else f"problem_{problem_idx_to_sha[idx]}"
            )
            # We only need the winner and loser names and the time stamp for each ``match''
            game_records.append(
                {
                    "eval_timestamp": eval_timestamp,
                    "winner_name": winner,
                    "loser_name": loser,
                }
            )
    # Creating the new dataframe
    game_records = (
        pd.DataFrame(
            game_records, columns=["eval_timestamp", "winner_name", "loser_name"]
        )
        # Shuffle the records, but we have tested that the order does not affect the Elo/Glicko2 ratings
        .sample(frac=1, random_state=42)
        .sort_values(by="eval_timestamp", ascending=True)
        .reset_index(drop=True)
    )
    return game_records


# Fit Elo or Glicko2 ratings to the game records
def fit_elo_glicko2(game_records, eval_results, dataset_name, rating_system="glicko2"):
    num_problems = len(eval_results["eval_results_" + dataset_name.lower()].iloc[0])
    problem_sha_to_idx = {
        hashlib.sha1(str(idx).encode()).hexdigest(): idx for idx in range(num_problems)
    }
    # Estimate the ratings usign skelo package
    assert rating_system in ["elo", "glicko2"]
    RatingEstimator = EloEstimator if rating_system == "elo" else Glicko2Estimator
    model = RatingEstimator(
        key1_field="winner_name",
        key2_field="loser_name",
        timestamp_field="eval_timestamp",
        initial_time=game_records["eval_timestamp"].min(),
        initial_value=(1500.0, 350.0, 0.06),
    ).fit(game_records, len(game_records) * [1])
    ratings = model.rating_model.to_frame()

    # Only keep the latest ratings, i.e., the ratings with the latest valid_to timestamp
    ratings = ratings[ratings["valid_to"].isna()].drop(
        columns=["valid_from", "valid_to"]
    )
    # For Elo, ratings are scalar values, for Glicko2, ratings are tuples of (rating, rating deviation, volatility)
    # Expand the Glicko2 ratings to separate columns
    if rating_system == "glicko2":
        ratings[["rating", "rating_deviation", "rating_volatility"]] = ratings[
            "rating"
        ].apply(lambda x: pd.Series(x))
    # Separate the model ratings and problem ratings
    model_elo_ratings = (
        ratings[ratings["key"].str.startswith("model_")]
        .assign(key=lambda row: row["key"].apply(lambda key: key.split("_")[1]))
        .rename({"key": "model_sha"}, axis=1)
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )
    problem_elo_ratings = (
        ratings[ratings["key"].str.startswith("problem_")]
        .assign(
            key=lambda row: row["key"]
            .apply(lambda key: problem_sha_to_idx[key.split("_")[1]])
            .astype(int)
        )
        .rename({"key": "sorted_index"}, axis=1)
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )
    # Renormalize the model and problem ratings seperatedly
    for ratings in [problem_elo_ratings, model_elo_ratings]:
        # Scale to a range of [500, 2500] with mean 1500
        scale_factor = (2500 - 500) / (
            ratings["rating"].quantile(0.99) - ratings["rating"].quantile(0.01)
        )
        ratings["rating"] = (
            ratings["rating"] - ratings["rating"].median()
        ) * scale_factor + 1500
        if rating_system == "glicko2":
            ratings["rating_deviation"] *= scale_factor
    return problem_elo_ratings, model_elo_ratings


# Convert the evaluation results to item responses like format
def eval_results_to_item_responses(eval_results, dataset_name):
    item_responses = pd.concat(
        [
            eval_results[["model_sha"]],
            pd.DataFrame(
                eval_results[f"eval_results_{dataset_name.lower()}"].tolist(),
                index=eval_results.index,
            )
            .astype(float)
            .rename(columns=lambda idx: f"problem_{idx}"),
        ],
        axis=1,
    )
    return item_responses


# Fit IRT model to the item responses
def fit_irt(
    item_responses, eval_results, dataset_name, irt_model="4PL", lr=0.1, epochs=1000, diff_param = None
):
    num_problems = len(eval_results["eval_results_" + dataset_name.lower()].iloc[0])
    print(num_problems)
    assert irt_model in ["1PL", "2PL", "3PL", "4PL", "1gPL"]
    IRTModel = {
        "1PL": OneParamLog,
        "2PL": TwoParamLog,
        "3PL": ThreeParamLog,
        "4PL": FourParamLog,
        "1gPL": OneParamWithGuessLog,
    }[irt_model]
    # Train the IRT model
    trainer = IRTModel.train(
        IRTDataset.from_pandas(
            item_responses,
            subject_column="model_sha",
            item_columns=[f"problem_{idx}" for idx in range(num_problems)],
        ),
        # Use the default optimizer and hyperparameters of the paper
        lr=lr,
        epochs=epochs,
        verbose= False, ## modified 是否显示loss
        diff_param = diff_param
    )
    irt_results = trainer.irt_model.export()
    problem_irt_results = {
        "sorted_index": range(len(irt_results["diff"])),
        "difficulty": np.array(irt_results["diff"]),
        "difficulty_std": np.array(irt_results["diff_std"]),
        # Discrimination and feasibility are only available for 2PL, 3PL and 4PL models
        "discrimination": np.array(irt_results["disc"]) if irt_model in ["2PL", "3PL", "4PL"] else float("NaN"),
        # Feasibility is only available for 3PL models
        "guessing": np.array(irt_results["lambdas"]) if irt_model in ["3PL", "1gPL"] else float("NaN"),
        # Feasibility is only available for 4PL models
        "feasibility": np.array(irt_results["lambdas"]) if irt_model in ["4PL"] else float("NaN"),
    }
    problem_irt_results = pd.DataFrame(problem_irt_results)
    # Model ability is also available
    model_irt_results = pd.DataFrame(
        {
            "model_sha": item_responses["model_sha"],
            "ability": np.array(irt_results["ability"]),
        }
    )
    return problem_irt_results, model_irt_results


def main(minimum_hub_likes, lr, model, epochs, huggingface_path_prefix, version_tag, elo_glicko=False, irt=True, overwrite=False, done_list=None):
    if minimum_hub_likes is None:
        eval_results = load_and_filter_eval_result(not_merged_only=False, not_flagged_only=False)
    else:
        eval_results = load_and_filter_eval_result(minimum_hub_likes=minimum_hub_likes)
    # model_merged_table = eval_results.copy()
    for dataset_name in tqdm(
        ["ARC", "GSM8K", "HellaSwag", "Winogrande"],
        desc="Processing datasets",
    ):
        split_name = f"minlikes_{minimum_hub_likes}_model_{model}_lr_{lr}_epochs_{epochs}"
        if (not overwrite) and (split_name in done_list[dataset_name]):
            print(f"{split_name} has finished.")
            continue

        # Load the original dataset, specific split and sorted to the same order as the eval results
        original_dataset = get_sorted_original_dataset(dataset_name).to_pandas()
        original_dataset["model_avg_acc"] = eval_results[
            f"eval_results_{dataset_name.lower()}"
        ].apply(lambda x: np.array(x).astype(float)).sum() / len(eval_results)

        data_merged_table = original_dataset
        
        # Elo/Glicko2 ratings
        if elo_glicko:
            game_records = eval_results_to_game_records(
                eval_results, dataset_name=dataset_name
            )
            problem_elo_ratings, model_elo_ratings = fit_elo_glicko2(
                game_records, eval_results, dataset_name=dataset_name
            )
            data_merged_table = data_merged_table.merge(
                problem_elo_ratings, on="sorted_index", how="inner"
            )
        
        # IRT ratings
        if irt:
            item_responses = eval_results_to_item_responses(
                eval_results, dataset_name=dataset_name
            )
            problem_irt_results, model_irt_results = fit_irt(
                item_responses, eval_results, dataset_name=dataset_name,
                irt_model=model, lr=lr, epochs=epochs
            )
            data_merged_table = data_merged_table.merge(
                problem_irt_results, on="sorted_index", how="inner"
            )
        
        Dataset.from_pandas(
            data_merged_table.reset_index(drop=True)
        ).push_to_hub(huggingface_path_prefix + dataset_name, version_tag, split=split_name)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--minimum_hub_likes",
        type=int,
        default=None,
        help="Minimum hub likes to filter the models",
    )
    argparser.add_argument(
        "--huggingface_path_prefix",
        type=str,
        default="mcding-org/Easy2Hard-",
        help="Path to the Hugging Face dataset",
    )
    argparser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
    )
    argparser.add_argument(
        "--model",
        type=str,
        help="IRT model",
    )
    argparser.add_argument(
        "--version_tag",
        type=str,
        default="v4",
        help="Version tag for the Hugging Face dataset",
    )
    argparser.add_argument(
        "--overwrite",
        action='store_true'
    )
    args = argparser.parse_args()

    dataset_name = ["ARC", "GSM8K", "HellaSwag", "Winogrande"]
    done_list = {}
    for dataset_name in ["ARC", "GSM8K", "HellaSwag", "Winogrande"]:
        try:
            done_list[dataset_name] = get_dataset_split_names(f"mcding-org/Easy2Hard-{dataset_name}", args.version_tag)
            print(f"{dataset_name}: {len(done_list[dataset_name])}")
        except:
            done_list[dataset_name] = []

    for minimum_hub_likes in [None, 0, 5, 10, 25, 50, 100]:
        for epochs in [800, 1600, 3200]:
            try:
                main(minimum_hub_likes, args.lr, args.model, epochs, args.huggingface_path_prefix, args.version_tag, overwrite=args.overwrite, done_list=done_list)
            except Exception as e:
                print(e)
