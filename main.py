import sys
import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] ="0"
sys.path.append('./')

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np
import pandas as pd
import json, logging, argparse, random, warnings, pickle, csv
import torch
from utils.util import get_model_param, parse_json
from src.problem_assistant import Problem_Assiastant
from estimate.irt_gradient_descent import calculate_abilities_df_binary_single
from tqdm import tqdm
from utils.template_agent_classification import initialize_prompt_step1, summary_prompt_step1_multi,summary_prompt_step2_md
warnings.filterwarnings('ignore')

device = torch.device("cuda")


def parse_args():
    parser = argparse.ArgumentParser(description='Process model name.')
    parser.add_argument('--agent_model', type=str, required=False, default='Qwen2.5-VL-7B-Instruct', help='Name of the judging model to use')
    parser.add_argument('--test_model', nargs='+', required=False, default=None, help='Name of the models to test')
    parser.add_argument('--benchmark', type=str, required=False, default ="SEEDBench_IMG",  help='Name of the benchmark to use')
    parser.add_argument('--feature', type=str, required=False, default='text', help='text, image, multimean, multiconcat')
    parser.add_argument('--root_path', type=str, required=False, default='./', help='Root path of the project')
    parser.add_argument('--save_dir', type=str, required=False, default='out_folder', help='Output folder for results')
    parser.add_argument('--include_text', action='store_true', help='Whether to include text in the autojudger process')
    parser.add_argument('--include_image', action='store_true', help='Whether to include image in the autojudger process')

    return parser.parse_args()


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


def initialize_model_and_processor(model_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def load_data(benchmark, root_path, pf_data_path):
    prob_data = pd.read_csv(os.path.join(root_path, f"LMUData/{benchmark}.tsv"), sep='\t')          # problems
    data = pd.read_csv(os.path.join(root_path, f'model_performance/{benchmark}/{benchmark}.csv'))   # model-question records
    info_dict, info_df = get_model_param(os.path.join(root_path,'data/model_information_new.csv'), return_df=True) # model information
    
    train_prob_df = pd.read_json(os.path.join(pf_data_path, 'train', 'train_prob_df.json'), lines=True)
    train_prob_df[['loc_diff', 'scale_diff']] = train_prob_df[['difficulty', 'difficulty_std']]

    train_model_df = pd.read_json(os.path.join(pf_data_path, 'train', 'train_model_df.json'), lines=True)
    with open(os.path.join(pf_data_path, 'train', 'train_model_list.json'), 'r', encoding='utf-8') as file:
        train_model_list = json.load(file)
    with open(os.path.join(root_path, 'data', 'test_model_list.json'), 'r', encoding='utf-8') as file:
        test_model_list = json.load(file)
    
    full_benchmark_json_path = os.path.join(pf_data_path, f'{benchmark}.json')
    if not os.path.exists(full_benchmark_json_path):
        result_dict = process_file_to_dict(os.path.join(root_path, f'model_performance/{benchmark}/{benchmark}.csv'))
        save_dict_to_json(result_dict, full_benchmark_json_path)
    model_pf_dict = {}
    with open(full_benchmark_json_path, 'r') as file:
        for line in file:
            data_1 = json.loads(line)       # 解析每一行的 JSON 数据（data 是一个 dict）
            model_pf_dict.update(data_1)    # 将 data 中的键值对合并到 model_pf_dict 中

    #题目内容prob_data，答题记录data，题目难度train_prob_df
    return prob_data, data, info_dict, info_df, train_prob_df, train_model_df, model_pf_dict, train_model_list, test_model_list


def setup_logging(model_name, benchmark, approach, root_path, out_folder):
    if not os.path.exists(f'{root_path}/{out_folder}/{benchmark}/{approach}'):
        os.makedirs(f'{root_path}/{out_folder}/{benchmark}/{approach}')
    logging.basicConfig(
        filename=f'{root_path}/{out_folder}/{benchmark}/{approach}/logfile.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    logging.info(f"Model name: {model_name} Benchmark: {benchmark} Approach: {approach}")


def get_recommended_problems(prob_ast, ability, vec):
    # recommend_pb_df, recommend_diff_df = prob_ast.get_recomm_problem_nearest(ability, top_n = 5) #prob_ast.get_recomm_problem_quantile5(ability)
    recommend_pb_df, recommend_diff_df = prob_ast.get_recomm_problem_farthest(vec,device, ability,top_n = 10)
    return recommend_pb_df, recommend_diff_df


def process_data_list(rec_merged_data, model_pf_dict, model_under_test):
    data_list = []
    
    # 获取所有可能的选项列名（A-Z）
    all_possible_options = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    
    for index, row in rec_merged_data.iterrows():
        # 确定哪些选项列实际存在于数据中
        existing_options = [opt for opt in all_possible_options if opt in row.index]
        
        # 处理没有标准选项的情况
        if not {'A', 'B', 'C', 'D'}.issubset(set(existing_options)):
            # 从answer中提取唯一选项
            answer_options = set(row['answer'].upper().replace(' ', '').split(','))
            # 按字母顺序排序
            sorted_options = sorted(answer_options)
            # 创建选项字典，按顺序映射到A,B,C...
            options_dict = {}
            for i, opt in enumerate(sorted_options):
                options_dict[chr(ord('A') + i)] = opt
        else:
            # 有标准选项的情况，收集所有存在的选项
            options_dict = {}
            for opt in existing_options:
                if not pd.isna(row[opt]):  # 忽略NaN值
                    options_dict[opt] = row[opt]
        
        data_dict = {
            'index': row['model_sha'],
            'question': row['question'],
            'answer': row['answer'],
            'options': options_dict,
            'image': row['image'],
            'difficulty': row['difficulty'],
            'correct': model_pf_dict[model_under_test][str(row['model_sha'])]
        }
        data_list.append(data_dict)
    
    return data_list

 
def generate_model_summary(model, processor, messages, max_fail_num,question_list,ini_merged_data,response,model_under_test):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    tt = 0
    while tt < max_fail_num:
        tt=tt+1
        # try:
        generated_ids = model.generate(**inputs, max_new_tokens=12800)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        logging.info(output_text)
        try:
            return_dict = json.loads(output_text[0])
        except:
            
            json_str = output_text[0]

            # 使用正则提取 JSON 部分
            json_cleaned = re.search(r'{.*}', json_str, re.DOTALL).group()

            # 解析为字典
            return_dict = json.loads(json_cleaned)

        if not set(return_dict.keys()).issubset(set([str(x) for x in question_list])):
            logging.info('问题ID不在推荐列表中')
            return None
            
        category_df = (
            pd.DataFrame(list(return_dict.items()), columns=['model_sha', 'CLASS'])
            .explode('CLASS')  # 将多分类列表展开为独立行
          
        )
        category_df[ 'model_sha']=category_df[ 'model_sha'].astype(int)
        logging.info(category_df['CLASS'].unique())
        # 合并数据（确保ID类型一致）
        merged_df = pd.merge(
            category_df, 
            ini_merged_data[['model_sha','difficulty']], 
            on='model_sha', 
            how='left'
        )
        merged_df = pd.merge(
            merged_df, 
            response, 
            on='model_sha', 
            how='left'
        )
        # logging.info(merged_df)
        # merged_df['answer']=merged_df['answer'].astype(int)
        # 按类别分组统计
        summary = merged_df.groupby('CLASS').agg(
            question_count=('model_sha', 'count'),  # 题目数量
            max_difficulty=('difficulty', 'max'),  # 最大难度
            min_difficulty=('difficulty', 'min'),  # 最小难度
            avg_difficulty=('difficulty', 'mean'),  # 平均难度
            accuracy=(model_under_test, 'mean')  # 正确率（answer列的平均值）
        ).reset_index()
        
        # 格式化数值（保留两位小数）
        summary['avg_difficulty'] = summary['avg_difficulty'].round(2)
        summary['accuracy'] = summary['accuracy'].round(2)
        # torch.cuda.empty_cache()
        return summary
        # except:
        #     logging.info(f'失败{tt}次')
      
        
def generate_model_summary_itera(model, processor, messages, max_fail_num,summary,question_id,prob_data,data,model_under_test):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    tt = 0
    while tt < max_fail_num:
        tt=tt+1
        try:
            generated_ids = model.generate(**inputs, max_new_tokens=12800)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # logging.info(output_text)
            # try:
            return_dict = json.loads(output_text[0])
            logging.info(return_dict['category'])
            # question_id = return_dict['Question ID']
            # summary = return_dict['Summary']
            # thought = return_dict['Thought']
            # logging.info(question_list)
            def update_summary(summary, new_class, new_difficulty, new_correct):
                """
                更新分类统计表格summary，加入新问题的数据。

                参数:
                    summary (pd.DataFrame): 原始统计表格，包含列 'CLASS', 'question_count', 'max_difficulty', 'min_difficulty', 'avg_difficulty', 'accuracy'
                    new_class (str): 新问题的分类
                    new_difficulty (float): 新问题的难度值
                    new_correct (int): 新问题的回答是否正确（0或1）

                返回:
                    pd.DataFrame: 更新后的统计表格
                """
                # 检查是否已有该分类
                if new_class in summary['CLASS'].values:
                    # 获取当前统计值
                    mask = summary['CLASS'] == new_class
                    current = summary[mask].iloc[0]
                    
                    # 更新统计值
                    new_count = current['question_count'] + 1
                    new_max = max(current['max_difficulty'], new_difficulty)
                    new_min = min(current['min_difficulty'], new_difficulty)
                    new_avg = (current['avg_difficulty'] * current['question_count'] + new_difficulty) / new_count
                    new_acc = (current['accuracy'] * current['question_count'] + new_correct) / new_count
                    
                    # 更新到DataFrame
                    summary.loc[mask, 'question_count'] = new_count
                    summary.loc[mask, 'max_difficulty'] = new_max
                    summary.loc[mask, 'min_difficulty'] = new_min
                    summary.loc[mask, 'avg_difficulty'] = new_avg
                    summary.loc[mask, 'accuracy'] = new_acc
                else:
                    # 创建新分类的统计行
                    new_row = pd.DataFrame({
                        'CLASS': [new_class],
                        'question_count': [1],
                        'max_difficulty': [new_difficulty],
                        'min_difficulty': [new_difficulty],
                        'avg_difficulty': [new_difficulty],
                        'accuracy': [new_correct]
                    })
                    summary = pd.concat([summary, new_row], ignore_index=True)
                
                # 格式化数值（保留两位小数）
                summary['avg_difficulty'] = summary['avg_difficulty'].round(2)
                summary['accuracy'] = summary['accuracy'].round(2)
                
                return summary
            if isinstance(return_dict['category'], list):
                new_categories = [c for c in return_dict['category'] 
                        if c not in summary['CLASS'].unique()]
        
                if new_categories:
                    logging.info(f'出现了新的类别集合: {new_categories}')
                
                # 获取题目数据
                difficulty = prob_data[prob_data['index'] == question_id]['difficulty'].to_list()[0]
                correct = data[data['model_sha'] == question_id][model_under_test].to_list()[0]
                
                # 逐个更新每个分类
                for category in return_dict['category']:
                    summary = update_summary(summary, category, difficulty, correct)
                    
                return summary
            else:
                if return_dict['category'] not in summary['CLASS'].unique().tolist():
                    wq=return_dict['category']
                    logging.info(f'出现了新的类别{wq}')
                    summary=update_summary(summary,return_dict['category'],prob_data[prob_data['index']==question_id]['difficulty'].to_list()[0],data[data['model_sha']==question_id][model_under_test].to_list()[0])
                else:
                    logging.info(f'未出现新的类别')
                    summary=update_summary(summary,return_dict['category'],prob_data[prob_data['index']==question_id]['difficulty'].to_list()[0],data[data['model_sha']==question_id][model_under_test].to_list()[0])
            
                return summary
        except:
            logging.info(f'失败{tt}次')
        return summary


def generate_model_choose(model, processor, messages, max_fail_num, data_list, prob_ast, recommend_diff_df):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    tt = 0
    while tt < max_fail_num:
        generated_ids = model.generate(**inputs, max_new_tokens=12800)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        logging.info(output_text)
        try:
            # return_dict = parse_json(output_text[0])
            return_dict = json.loads(output_text[0])
            question_id = int(return_dict['question_index'])
            summary = return_dict['summary']
            thought = return_dict['think']
            if question_id not in recommend_diff_df['model_sha'].astype(int).values:
                logging.info('问题ID不在推荐列表中')
                tt += 1
                if tt == max_fail_num:
                    thought = ''
                    random_choice = random.choice(data_list)
                    question_id = random_choice['index']
                    difficulty = random_choice['difficulty']
                   
                    prob_ast.update_recommended_pb_diff(question_id)
                    logging.info('随机选取')
                else:
                    pre_question = int(return_dict["Question ID"])

            else:
                logging.info('成功选择题目')
                prob_ast.update_recommended_pb_diff(question_id)
                difficulty = recommend_diff_df.loc[recommend_diff_df['model_sha'] == question_id, 'difficulty'].values[0]
                break
        except:
            logging.info('解析错误')
            tt += 1
            if tt == max_fail_num:
                thought = ''
                random_choice = random.choice(data_list)
                question_id = random_choice['index']
                difficulty = random_choice['difficulty']
                try: 
                    summary+= f'We randomly select a question with ID {str(question_id)}: Difficulty {difficulty}.'
                except:
                    summary = f'We randomly select a question with ID {str(question_id)}: Difficulty {difficulty}.'
                prob_ast.update_recommended_pb_diff(question_id)
                logging.info('随机选取')
    # torch.cuda.empty_cache()

    return question_id, difficulty, None, thought


def calculate_inversions(order, true_order):
    inv_count = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if true_order.index(order[i]) > true_order.index(order[j]):
                inv_count += 1
    return inv_count


def ranking_acc(ability_dict,true_order,num,model_num):
    ability_dict
    current_orderRandom = sorted(ability_dict.keys(), key=lambda x: ability_dict[x][num], reverse=True)
    return (1-calculate_inversions(current_orderRandom, true_order)*2/model_num/(model_num-1))*100


def main():
    args = parse_args()
    print(args)
    sys.exit(0)

    root_path = args.root_path  # project root directory
    out_folder = args.save_dir  # output folder for results
    if not os.path.exists(f'{root_path}/{out_folder}/'):
        os.makedirs(f'{root_path}/{out_folder}/')

    # Initialize the judging model and processor
    judging_model_name = args.agent_model
    judging_model_path = f"./models/{judging_model_name}"
    judging_model, processor = initialize_model_and_processor(judging_model_path)

    # Load the benchmark data and question difficulty & embeddings
    benchmark = args.benchmark
    pf_data_path = os.path.join(root_path, f'data/{benchmark}')  # data directory
    prob_data, data, _, info_df, train_prob_df, _, model_pf_dict, train_model_list, test_model_list = \
        load_data(benchmark, root_path, pf_data_path)
    if args.test_model is not None: test_model_list = args.test_model
    
    feature = args.feature        # embedding features
    vec = np.load(os.path.join(root_path, f'clip_features/{benchmark}/{feature}_embeddings.npy'))
    vec = torch.from_numpy(vec).float().to(device)

    include_text, include_image = args.include_text, args.include_image
    approach = judging_model_name  + '_' + feature
    setup_logging(judging_model_name, benchmark, approach, root_path, out_folder)
    
    max_num = max(200,int(len(prob_data)/20))  # maximum number of iterations
    idxlist=[]
    for index, row in train_prob_df.iterrows():
        idxlist.append(row['model_sha'])
    prob_ast = Problem_Assiastant(data, info_df, train_model_list, idxlist=idxlist, problem_difficulty=train_prob_df)

    # concat prob_data and train_prob_df, so that we don't have to search two tables
    prob_data = pd.merge(
        prob_data,
        train_prob_df,
        left_on='index',
        right_on='model_sha',
        how='right'
    )
    
    ability_dict = {}
    question_dict = {}
    difficulty_dict = {}
    rec_dict = {}

    for model_under_test in test_model_list:
        torch.cuda.empty_cache()
        logging.info(f'[AutoJudger] Start evaluating the model: {model_under_test}')

        with open(f'./init/{benchmark}/clip_text_init10.json', 'rb') as f: rec_list = json.load(f)
        prob_ast.get_random_problem([model_under_test], 0, refresh=True)
        random_data, selected_diff = prob_ast.update_recommended_pb_diff(rec_list)
        model_result_df = calculate_abilities_df_binary_single(selected_diff, random_data)
        prob_ast.update_recommended_pb_diff(list(selected_diff['model_sha'].values))

        ability_list = [float(model_result_df['ability'].values[0])]
        question_list = [int(i) for i in list(selected_diff['model_sha'].values)]
        difficulty_list = [float(i) for i in list(selected_diff['difficulty'].values)]
        rec_dict[model_under_test] = [] # 初始化当前模型的记录

        # process the initial data
        ini_merged_data = prob_data[prob_data['index'].isin(prob_ast.recommend_diff_df['model_sha'].to_list())]
        ini_data_list = process_data_list(ini_merged_data, model_pf_dict, model_under_test)
        
        # init prompt
        ability = model_result_df['ability'].values[0]
        recommend_pb_df, recommend_diff_df = get_recommended_problems(prob_ast, ability, vec)

        # record the first round of recommended difficulties
        first_round_rec = dict(zip(recommend_diff_df['model_sha'], recommend_diff_df['difficulty'].astype(float)))
        rec_dict[model_under_test].append(first_round_rec)

        rec_merged_data =prob_data[prob_data['index'].isin(recommend_diff_df['model_sha'].to_list())]
        data_list = process_data_list(rec_merged_data, model_pf_dict, model_under_test)
        
        # based on the feature, we can decide whether to include text or image
        message1 = initialize_prompt_step1(ini_data_list,  include_text, include_image)
        summary = generate_model_summary(judging_model, processor, message1, 5,question_list,ini_merged_data,random_data[['model_sha',model_under_test]],model_under_test)
        logging.info(summary)
        message2 = summary_prompt_step2_md(summary, data_list, ability, include_text, include_image)
        question_id, difficulty, _, _=generate_model_choose(judging_model, processor, message2, 2, data_list, prob_ast, recommend_diff_df)

        model_result_df = calculate_abilities_df_binary_single(prob_ast.recommend_diff_df, prob_ast.recommend_pb_df)
        ability_list.append(model_result_df['ability'].values[0])
        question_list.append(question_id)
        difficulty_list.append(difficulty)

        for i in tqdm(range(max_num)):
            ability = model_result_df['ability'].values[0]
            recommend_pb_df, recommend_diff_df = get_recommended_problems(prob_ast, ability,vec)

            # record the current round of recommended difficulties
            current_round_rec = dict(zip(recommend_diff_df['model_sha'], recommend_diff_df['difficulty'].astype(float)))
            rec_dict[model_under_test].append(current_round_rec)

            rec_merged_data =prob_data[prob_data['index'].isin(recommend_diff_df['model_sha'].to_list())]
            data_list = process_data_list(rec_merged_data, model_pf_dict, model_under_test)

            ########################
            message1 = summary_prompt_step1_multi(summary,process_data_list(prob_data[prob_data['index']==int(question_id)], model_pf_dict, model_under_test) , include_text, include_image)
            summary = generate_model_summary_itera(judging_model, processor, message1, 5,summary, int(question_id), prob_data, data[['model_sha',model_under_test]],model_under_test)
            logging.info(summary)
            message2 = summary_prompt_step2_md(summary, data_list, ability, include_text, include_image)
            question_id, difficulty, _, _ = generate_model_choose(judging_model, processor, message2, 2, data_list, prob_ast, recommend_diff_df)
            #######################

            model_result_df = calculate_abilities_df_binary_single(prob_ast.recommend_diff_df, prob_ast.recommend_pb_df)
            ability_list.append(model_result_df['ability'].values[0])
            question_list.append(question_id)
            difficulty_list.append(difficulty)

            logging.info(ability_list)
            logging.info(difficulty_list)
            logging.info(question_list)

            ability_dict[model_under_test] = ability_list
            question_dict[model_under_test] = question_list
            difficulty_dict[model_under_test] = difficulty_list

            with open(f'{root_path}/{out_folder}/{benchmark}/{approach}/ability_dict.json', 'w') as file:
                json.dump(ability_dict, file)
            with open(f'{root_path}/{out_folder}/{benchmark}/{approach}/question_dict.json', 'w') as file:
                # json.dump(convert_numpy_int64(question_dict), file)
                json.dump(question_dict, file)
            with open(f'{root_path}/{out_folder}/{benchmark}/{approach}/difficulty_dict.json', 'w') as file:
                json.dump(difficulty_dict, file)
            with open(f'{root_path}/{out_folder}/{benchmark}/{approach}/rec_dict.json', 'w') as file:
                json.dump(rec_dict, file)
    
    # ranking accuracy
    answers = pd.read_csv(f'./model_performance/{benchmark}/{benchmark}.csv')
    acc=answers.iloc[:,1:].mean(axis=0).to_dict()
    test_model_acc={key:acc[key] for key in test_model_list}
    true_order=sorted(test_model_acc.keys(),key=lambda x:test_model_acc[x],reverse=True)
    ranks=[]
    for num in range(len(ability_list)):
        ranks.append(ranking_acc(ability_dict,true_order,num,len(test_model_list)))
    with open(f'{root_path}/{out_folder}/{benchmark}/{approach}/ranking_acc.json', 'w') as file:
        json.dump(ranks, file)


if __name__ == "__main__":
    main()