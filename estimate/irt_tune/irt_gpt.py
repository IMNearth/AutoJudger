import numpy as np
import argparse
from datasets import Dataset, DatasetDict, load_dataset, get_dataset_config_names, concatenate_datasets
import math

CACHE_DIR = "/fs/cml-projects/E2H/Huggingface_cache"

def get_GPT_reuslt(gpt_dataset, m, num_gpt_results=3):
    def get_result(example, bias=0):
        result = 0
        count = 0
        for n in range(num_gpt_results):
            scoreA = example[f"score_QuestionA_{n}"]
            scoreB = example[f"score_QuestionB_{n}"]
            if scoreA and scoreB:
                if (not math.isnan(scoreA)) and (not math.isnan(scoreB)):
                    result += scoreA-scoreB
                    count += 1
        if count>=3:
            avg_score = result/count
            example["gpt_result"] = int(avg_score<0) if abs(avg_score)>=bias else float("NaN")
        else:
            example["gpt_result"] = float("NaN")
        return example
    return gpt_dataset.map(lambda x:get_result(x, bias=m), load_from_cache_file=False,)["gpt_result"]


def expand_irt_params(example):
    if example["irt_params"].startswith("ACC_UNC_"):
        irt_param = example["irt_params"].split("_")
        example["acc_uncertainty"] = float(irt_param[2])
        for key in ["irt_minlikes", "irt_model", "irt_lr", "irt_epochs"]:
            example[key] = None
    else:
        irt_param = example["irt_params"].split("_")
        example["irt_minlikes"] = irt_param[1]
        example["irt_model"] = irt_param[3]
        example["irt_lr"] = float(irt_param[5])
        example["irt_epochs"] = int(irt_param[7])
        example["acc_uncertainty"] = None
    return example


def main(repo_name, version_tag):
    irt_results_dict = {}
    for dataset_name in ["GSM8K", "ARC", "HellaSwag", "Winogrande"]:
        config_names = get_dataset_config_names(f"mcding-org/Easy2Hard-{dataset_name}-GPT")
        gpt_dataset = concatenate_datasets([load_dataset(f"mcding-org/Easy2Hard-{dataset_name}-GPT", config, split="default", cache_dir=CACHE_DIR) for config in config_names if not config.endswith("human")])
        irt_dataset = load_dataset(f"mcding-org/Easy2Hard-{dataset_name}", "v4", cache_dir=CACHE_DIR)

        GPT_diff_list = [0.5*n for n in range(11)]

        GPT_diff_results = {f"GPT_{m}":get_GPT_reuslt(gpt_dataset, m) for m in GPT_diff_list}

        def get_IRT_reuslt(irt_split):
            in_time_reverse = 1
            if all([n is not None for n in irt_split["discrimination"]]):
                if np.mean(np.array(irt_split["discrimination"])<0)>0.5:
                    in_time_reverse = -1
            diff_0 = [n*in_time_reverse for n in irt_split[gpt_dataset['question0']]["difficulty"]]
            diff_1 = [n*in_time_reverse for n in irt_split[gpt_dataset['question1']]["difficulty"]]
            diff_std = [max(n0, n1) for n0, n1 in zip(irt_split[gpt_dataset['question0']]["difficulty_std"], irt_split[gpt_dataset['question1']]["difficulty_std"])]
            return (diff_0, diff_1, diff_std)
    
        def get_model_acc_result(irt_split, acc_uncertainty):
            diff_0 = [-1*n for n in irt_split[gpt_dataset['question0']]["model_avg_acc"]]
            diff_1 = [-1*n for n in irt_split[gpt_dataset['question1']]["model_avg_acc"]]
            return (diff_0, diff_1, acc_uncertainty)

        irt_diff_results = {irt_param:get_IRT_reuslt(irt_split) for irt_param, irt_split in irt_dataset.items()}
        true_acc_results = {f"ACC_UNC_{0.05*n}":get_model_acc_result(list(irt_dataset.values())[0], 0.05*n) for n in range(11)}
        irt_diff_results.update(true_acc_results)

        results = {diff_key:{irt_param:[] for irt_param in irt_diff_results.keys()} for diff_key in GPT_diff_results}
        ratios = {irt_param:[] for irt_param in list(irt_diff_results.keys())}

        for irt_param, irt_value in irt_diff_results.items():
            if isinstance(irt_value[2], list): # IRT
                ratios[irt_param] = np.nanmean([int(abs(a0-a1)>std) for a0, a1, std in zip(irt_value[0], irt_value[1], irt_value[2])])
                irt_result = [int(a0<a1) if abs(a0-a1)>std else float("NaN") for a0, a1, std in zip(irt_value[0], irt_value[1], irt_value[2])]
            else: # Model Accuracy 
                ratios[irt_param] = np.nanmean([int(abs(a0-a1)>irt_value[2]) for a0, a1 in zip(irt_value[0], irt_value[1])])
                irt_result = [int(a0<a1) if abs(a0-a1)>irt_value[2] else float("NaN") for a0, a1 in zip(irt_value[0], irt_value[1])]
                
            for GPT_diff_key, GPT_diff_value in GPT_diff_results.items():
                results[GPT_diff_key][irt_param] += [(float(gpt==irt) if not (math.isnan(gpt) or math.isnan(irt)) else float("NaN")) for gpt, irt in zip(GPT_diff_value, irt_result)]

        for diff_key in results.keys():
            for key, value in results[diff_key].items():
                nan_list = [math.isnan(n) for n in value]
                irt_gpt_ratio = 1-np.mean(nan_list) if nan_list else 1
                irt_gpt_acc = 0.0 if irt_gpt_ratio==0.0 else np.nanmean(value)
                results[diff_key][key] = [irt_gpt_acc, irt_gpt_ratio]
        df_dict = {"irt_params":results["GPT_0.0"].keys(), "irt_std_filter_ratio":ratios.values()}
        for diff_key in results.keys():
            df_dict.update({diff_key+"_acc":[values[0] for values in results[diff_key].values()]})
            df_dict.update({diff_key+"_ratio":[values[1] for values in results[diff_key].values()]})

        irt_results_dict[dataset_name] = Dataset.from_dict(df_dict).map(
            lambda x: expand_irt_params(x),
            load_from_cache_file=False,
        ).select_columns(
            ["acc_uncertainty", "irt_minlikes", "irt_model", "irt_lr", "irt_epochs", "irt_std_filter_ratio"] + sum([[key+"_acc", key+"_ratio"]for key in results.keys()], [])
        ).sort(
            ["acc_uncertainty", "irt_minlikes", "irt_model", "irt_lr", "irt_epochs"]
        )
    
    DatasetDict(irt_results_dict).push_to_hub(repo_name, version_tag)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--repo_name",
        type=str,
        default="mcding-org/Easy2Hard-IRT-tune",
        help="Path to the Hugging Face dataset",
    )
    argparser.add_argument(
        "--version_tag",
        type=str,
        default="v4",
        help="Version tag for the Hugging Face dataset",
    )
    args = argparser.parse_args()
    
    main(args.repo_name, args.version_tag)