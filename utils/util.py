import sys
# sys.path.append('..')
import pandas as pd
import numpy as np
from estimate.irt_tune.irt_eval import fit_irt
import random, json, os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr


import base64, copy
from io import BytesIO
from qwen_vl_utils.vision_process import smart_resize
from PIL import Image

IMAGE_FACTOR = 28
def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")
def resize_base64_image(base64_data, resized_height, resized_width):
    import re
    header, encoded = base64_data.split(',', 1)
    match = re.match(r"data:image/(\w+);base64", header)
    img_format = match.group(1).upper() if match else "PNG"

    data = base64.b64decode(encoded)
    with BytesIO(data) as bio:
        image_obj = copy.deepcopy(Image.open(bio))

    image = to_rgb(image_obj)
    resized_height, resized_width = smart_resize(resized_height, resized_width, factor=1)
    image = image.resize((resized_width, resized_height))

    buffered = BytesIO()
    image.save(buffered, format=img_format)
    new_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"{header},{new_base64}"


def resize_base64_image_pixel(base64_data, min_pixels, max_pixels):
    import re
    header, encoded = base64_data.split(',', 1)
    match = re.match(r"data:image/(\w+);base64", header)
    img_format = match.group(1).upper() if match else "PNG"

    data = base64.b64decode(encoded)
    with BytesIO(data) as bio:
        image_obj = copy.deepcopy(Image.open(bio))

    image = to_rgb(image_obj)
    # resized_height, resized_width = smart_resize(resized_height, resized_width, factor=1)
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=1,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))

    buffered = BytesIO()
    image.save(buffered, format=img_format)
    new_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"{header},{new_base64}"


def irt(item_responses, epochs=1000, diff_param = None):
    n_p = len(item_responses['model_sha'])
    model_sha = item_responses['model_sha'].copy()
    item_responses.loc[:, 'model_sha'] = [f'problem_{i}' for i in range(n_p) ]
    # 设置 'model_sha' 为索引并转置
    transposed_responses = item_responses.set_index('model_sha').T
    # 重置索引并修改列名
    transposed_responses = transposed_responses.reset_index().rename(columns={'index': 'model_sha'})
    # 修改 'model_sha' 列名
    transposed_responses.columns.name = None

    eval_results = {
        'eval_results_dataset1': pd.DataFrame({
            f'metric{i}': [0.8, 0.6, 0.9] for i in range(n_p)
        })
    }

    # 数据集名称
    dataset_name = 'dataset1'

    # 调用 fit_irt 函数
    problem_irt_results, model_irt_results = fit_irt(
        item_responses=transposed_responses,
        eval_results=eval_results,
        dataset_name=dataset_name,
        irt_model="1PL",
        lr=0.1,
        epochs=epochs,
        diff_param = diff_param
    )
    # 将 model_sha 列添加到 problem_irt_results 的第一列
    item_responses.loc[:, 'model_sha'] = model_sha
    problem_irt_results.insert(0, 'model_sha', model_sha)
    return problem_irt_results, model_irt_results

def irt_single(item_responses, epochs=100, diff_param=None):
    """
    每次预测一个model的result，最后把它们合在一起，调用irt。
    """
    results = []

    for model in item_responses.columns[1:]:
        single_response_df = item_responses[['model_sha', model]]
        problem_irt_results, model_irt_results = irt(single_response_df, epochs=epochs, diff_param=diff_param)
        results.append(model_irt_results)

    # 合并所有结果
    final_result_df = pd.concat(results, ignore_index=True)
    return problem_irt_results, final_result_df


def get_model_param(info_path, return_df = False):
    model_information = pd.read_csv(info_path)[['models', '参数量','单位','发布时间']]
    model_withParams = model_information[['models','参数量', '单位']]
    # model_withParams.sort_values(by=['单位', '参数量'], ascending=[True, False], inplace=True)
    # 创建一个字典来保存不同条件的数据
    dataframes_dict = {
        # '单位为M': model_withParams[model_withParams['单位'] == 'M'],
        '<5B': model_withParams[(model_withParams['单位'] == 'M') | ((model_withParams['单位'] == 'B') & (model_withParams['参数量'] < 5))],
        '<9B': model_withParams[(model_withParams['单位'] == 'B') & (model_withParams['参数量'] >=5) & (model_withParams['参数量'] < 9)],
        '<16B': model_withParams[(model_withParams['单位'] == 'B') & (model_withParams['参数量'] >=9) & (model_withParams['参数量'] < 16)],
        # '<80B': model_withParams[(model_withParams['单位'] == 'B') & (model_withParams['参数量'] >= 16)],
        # 'Other': model_withParams[~((model_withParams['单位'] == 'M') | (model_withParams['单位'] == 'B'))],
        'Other': model_withParams[((model_withParams['单位'] == 'B') & (model_withParams['参数量'] >= 16)) | (~((model_withParams['单位'] == 'M') | (model_withParams['单位'] == 'B')))],
        'All' : model_withParams
    }
    if return_df:
        return dataframes_dict, model_information
    return dataframes_dict



def compare_abilities(merged_abilities_df):
    """
    Compare each ability column with ability_full and return the comparison metrics.
    """
    metrics = {}

    for col in merged_abilities_df.columns:
        if col == 'model_sha' or col == 'full':
            continue
        metrics[f'mse_{col}'] = float(mean_squared_error(merged_abilities_df['full'], merged_abilities_df[col]))
        corr, _ = spearmanr(merged_abilities_df['full'], merged_abilities_df[col])
        metrics[f'spearman_corr_{col}'] = float(corr)

    # # 均方误差 (MSE)
    # # 斯皮尔曼等级相关系数 (Spearman Rank Correlation Coefficient)
    # if 'ability_test' in merged_abilities_df.columns:
    #     metrics['mse_test'] = float(mean_squared_error(merged_abilities_df['ability_full'], merged_abilities_df['ability_test']))
    #     metrics['spearman_corr_test'], _ = spearmanr(merged_abilities_df['ability_full'], merged_abilities_df['ability_test'])
    #     metrics['spearman_corr_test'] = float(metrics['spearman_corr_test'])
    # if 'ability_irt' in merged_abilities_df.columns:
    #     metrics['mse_irt'] = float(mean_squared_error(merged_abilities_df['ability_full'], merged_abilities_df['ability_irt']))
    #     metrics['spearman_corr_irt'], _ = spearmanr(merged_abilities_df['ability_full'], merged_abilities_df['ability_irt'])
    #     metrics['spearman_corr_irt'] = float(metrics['spearman_corr_irt'])
    # if 'ability_grad' in merged_abilities_df.columns:
    #     metrics['mse_grad'] = float(mean_squared_error(merged_abilities_df['ability_full'], merged_abilities_df['ability_grad']))
    #     metrics['spearman_corr_grad'], _ = spearmanr(merged_abilities_df['ability_full'], merged_abilities_df['ability_grad'])
    #     metrics['spearman_corr_grad'] = float(metrics['spearman_corr_grad'])
    return metrics


def merge_abilities(ability_dict):
    merged_df = ability_dict['all_model_df'].rename(columns={'ability': 'full'})
    for k, v in ability_dict.items():
        if k == 'all_model_df':
            continue
        merged_df = merged_df.merge(v.rename(columns={'ability': k}), on='model_sha')
    return merged_df



def save_results(pt, train_model_list, test_model_list, merged_abilities_df):
    import re
    os.makedirs(pt, exist_ok=True)
    dicts = { 'train_model_list': train_model_list,
              'test_model_list': test_model_list}
    metrics = compare_abilities(merged_abilities_df)
    merged_abilities_df.to_csv(f'{pt}/merged_abilities_df.csv', index=False)
    json.dump(dicts, open(f'{pt}/model_list.json', 'w'))
    json.dump(metrics, open(f'{pt}/metrics.json', 'w'))


def parse_json(output_text):
    # 尝试三种方式解析，如果失败了就用下一种
    import re
    pattern = r'\{.*?\}'
    matches = re.findall(pattern, output_text, re.DOTALL)[-1:]
    try:
        # print(matches[-1])
        question_id = matches[-1].split("Question ID")[-1]
        #去除question两侧的非数字字符
        question_id = re.sub(r'\D', '', question_id)
        summary = matches[-1].split("Question ID")[0].split("Summary")[-1]
        #去除summary两侧的"""," ","\n","{","}"
        # summary = re.sub(r'["\n{}]', '', summary)
        summary = summary.strip(" \n{}:\"")
        thought = matches[-1].split("Question ID")[0].split("Summary")[0].split("Thought")[-1]
        #去除thought两侧的"""," ","\n","{","}"
        # thought = re.sub(r'["\n{}]', '', thought)
        thought = thought.strip(" \n{}:\"")
    except:
        try:
            # 1. 提取 JSON 部分
            json_str = matches[-1]  # 取最后一个匹配项

            # 2. 按 ",\n    " 切分为三块
            parts = json_str.split(",\n")

            # 3. 用正则提取值
            result_dict = {}
            for part in parts:
                key_match = re.search(r'"(.*?)"', part)  # 提取 Key
                value_match = re.search(r':\s*(".*?"|\d+)', part)  # 提取 Value（支持字符串和整数）
                if key_match and value_match:
                    key = key_match.group(1)
                    value = value_match.group(1)
                    # 如果值是字符串，去掉双引号
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    # 如果值是整数，转换为整数类型
                    elif value.isdigit():
                        value = int(value)
                    result_dict[key] = value

            thought = result_dict["Thought"]
            summary = result_dict["Summary"]
            question_id = int(result_dict["Question ID"])
        except:
            for match in matches:
                result_dict = json.loads(match)
            thought = result_dict["Thought"]
            summary = result_dict["Summary"]
            question_id = int(result_dict["Question ID"])
    return_dict = {"Question ID": int(question_id), "Summary": summary, "Thought": thought}
    return return_dict
def parse_json_RAG(output_text):
    # 尝试三种方式解析，如果失败了就用下一种
    import re
    pattern = r'\{.*?\}'
    matches = re.findall(pattern, output_text, re.DOTALL)[-1:]
    try:
        # print(matches[-1])
        question = matches[-1].split("Question")[-1]
        question = question.strip(" \n{}:\"")
        #去除question两侧的非数字字符
        # question_id = re.sub(r'\D', '', question_id)
        summary = matches[-1].split("Question")[0].split("Summary")[-1]
        #去除summary两侧的"""," ","\n","{","}"
        # summary = re.sub(r'["\n{}]', '', summary)
        summary = summary.strip(" \n{}:\"")
        thought = matches[-1].split("Question")[0].split("Summary")[0].split("Thought")[-1]
        #去除thought两侧的"""," ","\n","{","}"
        # thought = re.sub(r'["\n{}]', '', thought)
        thought = thought.strip(" \n{}:\"")
    except:
        try:
            # 1. 提取 JSON 部分
            json_str = matches[-1]  # 取最后一个匹配项

            # 2. 按 ",\n    " 切分为三块
            parts = json_str.split(",\n")

            # 3. 用正则提取值
            result_dict = {}
            for part in parts:
                key_match = re.search(r'"(.*?)"', part)  # 提取 Key
                value_match = re.search(r':\s*(".*?"|\d+)', part)  # 提取 Value（支持字符串和整数）
                if key_match and value_match:
                    key = key_match.group(1)
                    value = value_match.group(1)
                    # 如果值是字符串，去掉双引号
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    # 如果值是整数，转换为整数类型
                    elif value.isdigit():
                        value = int(value)
                    result_dict[key] = value

            thought = result_dict["Thought"]
            summary = result_dict["Summary"]
            question = result_dict["Question"]
        except:
            for match in matches:
                result_dict = json.loads(match)
            thought = result_dict["Thought"]
            summary = result_dict["Summary"]
            question = result_dict["Question"]
    return_dict = {"Question": question, "Summary": summary, "Thought": thought}
    return return_dict
