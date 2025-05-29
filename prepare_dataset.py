import sys
# sys.path.append('../..')
import pandas as pd
from utils.util import irt, get_model_param
import os, warnings, json
from estimate.irt_gradient_descent import calculate_abilities_df_binary_single
from src.problem_assistant import Problem_Assiastant
from tqdm import tqdm
import csv
import warnings
warnings.filterwarnings('ignore')
import argparse


def save_data(folder, **kwargs):
    for name, data in kwargs.items():
        file_path = os.path.join(folder, f'{name}.json')
        if isinstance(data, pd.DataFrame):
            data.to_json(file_path, orient='records', lines=True)
        elif isinstance(data, (list, dict)):
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
        else:
            raise ValueError(f"Unsupported data type for {name}: {type(data)}")


# Function to process CSV file and convert it to a dictionary
def process_file_to_dict(file_path):
    result_dict = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        model_names = headers[1:]
        
        for model_name in model_names:
            result_dict[model_name] = {}
        
        for row in reader:
            model_sha = row[0]
            for i, value in enumerate(row[1:]):
                model_name = model_names[i]
                result_dict[model_name][model_sha] = value # int(float(value))
    return result_dict


def save_dict_to_json(result_dict, output_path):
    with open(output_path, 'w') as file:
        for model_name, model_data in result_dict.items():
            json_line = json.dumps({model_name: model_data})
            file.write(json_line + '\n')


def process_benchmark(benchmark, epochs, root_path='data', max_num_ratio=0.02, rand_cnt=1):
    print(f"\nProcessing benchmark: {benchmark}")
    
    orig_pt = f'model_performance/{benchmark}/{benchmark}.csv'
    df = pd.read_csv(orig_pt)
    
    train_df = df[['model_sha'] + train_model_list]
    test_df = df[['model_sha'] + test_model_list]
    full_model_list = list(train_model_list) + list(test_model_list)
    full_df = df[['model_sha'] + full_model_list]
    
    print("Running IRT analysis...")
    all_prob_df, all_model_df = irt(full_df, epochs=epochs)
    all_prob_df[['loc_diff', 'scale_diff']] = all_prob_df[['difficulty', 'difficulty_std']]
    train_prob_df, train_model_df = irt(train_df, epochs=epochs)
    train_prob_df[['loc_diff', 'scale_diff']] = train_prob_df[['difficulty', 'difficulty_std']]
    
    prob_ast = Problem_Assiastant(df, info_df, train_model_list, problem_difficulty=train_prob_df)
    
    # create directories for saving data
    data_folder = os.path.join(root_path, benchmark)
    train_folder = os.path.join(data_folder, 'train')
    test_folder = os.path.join(data_folder, 'test')
    full_folder = os.path.join(data_folder, 'full')
    os.makedirs(full_folder, exist_ok=True)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    save_data(
        train_folder,
        train_model_list=train_model_list,
        train_prob_df=train_prob_df,
        train_model_df=train_model_df
    )
    
    save_data(
        test_folder,
        test_model_list=test_model_list,
        all_prob_df=all_prob_df,
        all_model_df=all_model_df
    )
    
    save_data(
        full_folder,
        all_prob_df=all_prob_df,
        all_model_df=all_model_df
    )
    
    print(f"Converting CSV to JSON for {benchmark}...")
    file_path = os.path.join(full_folder, f'{benchmark}.csv')
    output_path = os.path.join(full_folder, f'{benchmark}.json')
    cmd = f'cp {orig_pt} {full_folder}'
    os.system(cmd)
    
    result_dict = process_file_to_dict(file_path)
    save_dict_to_json(result_dict, output_path)
    print(f"CSV to JSON conversion completed for {benchmark}")
    
    # calculate abilities for test models
    results = []
    for model_under_test in tqdm(test_model_list, desc=f"Evaluating models for {benchmark}"):
        random_data, selected_diff = prob_ast.get_random_problem([model_under_test], 1, refresh=True)
        model_result_df = calculate_abilities_df_binary_single(selected_diff, random_data)
        results.append({"model_sha": model_under_test, "ability": model_result_df['ability'].iloc[0]})
    
    with open(f'{test_folder}/test_model_df_fullPb.json', 'w') as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")
    
    print(f"Completed processing for benchmark: {benchmark}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset for benchmark')
    parser.add_argument('--benchmark', type=str, default='SEEDBench_IMG', help='Benchmark name')
    parser.add_argument('--root_path', type=str, default='data', help='Root path for data')
    parser.add_argument('--max_num_ratio', type=float, default=0.02, help='Maximum number ratio for IRT')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for IRT analysis')
    parser.add_argument('--rand_cnt', type=int, default=1, help='Random count for problem selection')
    args = parser.parse_args()

    info_dict, info_df = get_model_param(os.path.join(args.root_path, 'model_information_new.csv'), return_df=True)
    with open(os.path.join(args.root_path, 'test_model_list.json'), 'r') as f:
        test_model_list = json.load(f)

    all_model_list = []
    train_model_list = []
    for key, sub_df in info_dict.items():
        if key == 'All':
            continue
        model_list = sub_df['models'].tolist()
        all_model_list += model_list

    train_model_list = list(set(all_model_list) - set(test_model_list))
    print(f"Total models: {len(all_model_list)}, Train models: {len(train_model_list)}, Test models: {len(test_model_list)}")
    
    process_benchmark(args.benchmark, args.epochs, args.root_path, args.max_num_ratio, args.rand_cnt)
    print("Dataset preparation completed.")
