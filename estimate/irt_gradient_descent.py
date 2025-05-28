import numpy as np
from scipy.optimize import minimize
import pandas as pd

def calculate_ability(difficulty, responses):

    """
    根据IRT中的难度参数和模型作答情况，使用梯度下降法计算模型的能力值（theta）。
    
    参数：
    - difficulty: list 或 numpy 数组，包含题目的难度参数。
    - responses: list 或 numpy 数组，包含模型对题目的作答情况（1 为正确，0 为错误）。
    
    返回：
    - 模型的能力值（theta）。
    """
    difficulty = np.array(difficulty)
    responses = np.array(responses)
    
    # 对数似然函数
    def log_likelihood(theta, difficulty, responses):
        probabilities = 1 / (1 + np.exp(-(theta - difficulty)))
        log_likelihood = np.sum(responses * np.log(probabilities) + (1 - responses) * np.log(1 - probabilities))
        return -log_likelihood  # 负号用于最小化

    # 初始能力猜测
    initial_theta = 0.0
    # 最大化对数似然
    result = minimize(log_likelihood, initial_theta, args=(difficulty, responses), method='BFGS')

    # 返回模型能力值
    return result.x[0]


def calculate_abilities_df(difficulty_df, responses_matrix_df):
    """
    根据IRT中的难度参数和多个学生的作答情况，使用梯度下降法同时计算每个学生的能力值（theta）。
    
    参数：
    - difficulty: list 或 numpy 数组，包含题目的难度参数。
    - responses_matrix: 2D list 或 numpy 数组，包含多个学生对题目的作答情况（1 为正确，0 为错误）。
    
    返回：
    - 每个学生的能力值（theta）列表。
    """
    difficulty = np.array(difficulty_df['difficulty'])
    responses_matrix = np.array(responses_matrix_df.iloc[:, 1:].T.values.tolist())
    model_list = responses_matrix_df.columns[1:].tolist()
    
    num_students = responses_matrix.shape[0]

    # 对数似然函数
    def log_likelihood(thetas, difficulty, responses_matrix):
        log_likelihood = 0
        for i in range(num_students):
            theta = thetas[i]
            responses = responses_matrix[i]
            probabilities = 1 / (1 + np.exp(-(theta - difficulty)))
            log_likelihood += np.sum(responses * np.log(probabilities) + (1 - responses) * np.log(1 - probabilities))
        return -log_likelihood  # 负号用于最小化

    # 初始能力猜测
    initial_thetas = np.zeros(num_students)
    # 最大化对数似然
    result = minimize(log_likelihood, initial_thetas, args=(difficulty, responses_matrix), method='BFGS')

    # 返回每个学生的能力值
    result_df = pd.DataFrame({'model_sha': model_list, 'ability': result.x})
    return result_df


def calculate_abilities_df_single(difficulty_df, responses_matrix_df):
    """
    根据IRT中的难度参数和多个学生的作答情况，使用梯度下降法计算每个学生的能力值（theta）。
    每次预测一个model的结果，最后将它们合并起来。
    
    参数：
    - difficulty_df: DataFrame，包含题目的难度参数。
    - responses_matrix_df: DataFrame，包含多个学生对题目的作答情况（1 为正确，0 为错误）。
    
    返回：
    - 每个学生的能力值（theta）列表。
    """
    results = []

    for model in responses_matrix_df.columns[1:]:
        single_response_df = responses_matrix_df[['model_sha', model]]
        result_df = calculate_abilities_df(difficulty_df, single_response_df)
        results.append(result_df)
    # 合并所有结果
    final_result_df = pd.concat(results, ignore_index=True)
    return final_result_df


def calculate_ability_binary(difficulty, responses):
    """
    根据IRT中的难度参数和模型作答情况，使用二分算法计算模型的能力值（theta）。
    
    参数：
    - difficulty: list 或 numpy 数组，包含题目的难度参数。
    - responses: list 或 numpy 数组，包含模型对题目的作答情况（1 为正确，0 为错误）。
    
    返回：
    - 模型的能力值（theta）。
    """
    difficulty = np.array(difficulty)
    responses = np.array(responses)
    target = responses.sum()
    upper = 30
    lower = -30
    ability = 0

    def predscore(ability, difficulty):
        scores = 1. / (1 + np.exp(difficulty - ability))
        return scores.sum()

    while upper - lower > 1e-5:
        current_score = predscore(ability, difficulty)
        if current_score > target:
            upper = ability
        else:
            lower = ability
        ability = (upper + lower) / 2

    return ability
def calculate_ability_acc(difficulty, responses):
    """
    根据IRT中的难度参数和模型作答情况，使用二分算法计算模型的能力值（theta）。
    
    参数：
    - difficulty: list 或 numpy 数组，包含题目的难度参数。
    - responses: list 或 numpy 数组，包含模型对题目的作答情况（1 为正确，0 为错误）。
    
    返回：
    - 模型的能力值（theta）。
    """
    difficulty = np.array(difficulty)
    responses = np.array(responses)


    return responses.mean()

def calculate_abilities_df_binary_single(difficulty_df, responses_matrix_df):
    """
    根据IRT中的难度参数和多个学生的作答情况，使用梯度下降法计算每个学生的能力值（theta）。
    每次预测一个model的结果，最后将它们合并起来。
    
    参数：
    - difficulty_df: DataFrame，包含题目的难度参数。
    - responses_matrix_df: DataFrame，包含多个学生对题目的作答情况（1 为正确，0 为错误）。
    
    返回：
    - 每个学生的能力值（theta）列表。
    """
    difficulty = np.array(difficulty_df['difficulty'])
    model_list = responses_matrix_df.columns[1:].tolist()
    results = []

    for model in model_list:
        single_response = np.array(responses_matrix_df[['model_sha', model]].iloc[:, 1:].T.values.tolist())
        result_df = pd.DataFrame({'model_sha': model, 'ability': [calculate_ability_binary(difficulty, single_response)]})
        results.append(result_df)

    # 合并所有结果
    final_result_df = pd.concat(results, ignore_index=True)
    return final_result_df
def calculate_abilities_df_acc(difficulty_df, responses_matrix_df):
    """
    根据IRT中的难度参数和多个学生的作答情况，使用梯度下降法计算每个学生的能力值（theta）。
    每次预测一个model的结果，最后将它们合并起来。
    
    参数：
    - difficulty_df: DataFrame，包含题目的难度参数。
    - responses_matrix_df: DataFrame，包含多个学生对题目的作答情况（1 为正确，0 为错误）。
    
    返回：
    - 每个学生的能力值（theta）列表。
    """
    difficulty = np.array(difficulty_df['difficulty'])
    model_list = responses_matrix_df.columns[1:].tolist()
    results = []

    for model in model_list:
        single_response = np.array(responses_matrix_df[['model_sha', model]].iloc[:, 1:].T.values.tolist())
        result_df = pd.DataFrame({'model_sha': model, 'ability': [calculate_ability_acc(difficulty, single_response)]})
        results.append(result_df)

    # 合并所有结果
    final_result_df = pd.concat(results, ignore_index=True)
    return final_result_df


if __name__ == '__main__':
    print(calculate_ability_binary([0.1, 0.2, 0.3], [1, 0, 1]))  # 0.0