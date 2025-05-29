from utils.util import irt
import random
import pandas as pd
import numpy as np
import torch

class Problem_Assiastant:

    recommend_pb_df = None
    recommend_diff_df = None

    def __init__(self, performance_df, model_info, train_model_list,idxlist=None, problem_difficulty = None, model_ability = None):
        self.model_info = model_info
        self.train_model_list = train_model_list
        self.performance_df = performance_df
        self.train_performance_df = performance_df[['model_sha'] + train_model_list]
        
        self.idxlist=idxlist
        if problem_difficulty is None:
            self.problem_difficulty, self.model_ability = irt(self.train_performance_df, epochs=3200)
        else:
            self.problem_difficulty = problem_difficulty
            self.model_ability = model_ability


    def get_random_problem(self, model_list, ratio = 0.2, refresh = False):
        if refresh or self.recommend_pb_df is None:
            full_model_df = self.performance_df[['model_sha'] + model_list]
            #随机选取ratio比例的题目,并重新更新index的顺序
            self.recommend_pb_df = full_model_df.sample(frac=ratio)
            self.recommend_pb_df = self.recommend_pb_df.reset_index(drop=True)
            self.recommend_diff_df = self.problem_difficulty[self.problem_difficulty['model_sha'].isin(self.recommend_pb_df['model_sha'])].reset_index(drop=True)
            return self.recommend_pb_df, self.recommend_diff_df
        else:
            return self.recommend_pb_df, self.recommend_diff_df

    def get_recomm_problem(self, ability):
        """
        根据给定的 ability 找到最接近的 difficulty, 同时确保该题目没有被做过. 
        """
        # 按 difficulty 与 ability 的差值排序
        sorted_difficulties = self.problem_difficulty.copy()
        sorted_difficulties['diff'] = abs(sorted_difficulties['difficulty'] - ability)
        sorted_difficulties = sorted_difficulties.sort_values(by='diff')

        for _, row in sorted_difficulties.iterrows():
            model_sha = row['model_sha']
            if model_sha not in self.recommend_pb_df['model_sha'].values:
                # 找到未做过的题目
                self.recommend_pb_df = pd.concat([self.recommend_pb_df, self.performance_df[self.performance_df['model_sha'] == model_sha]], ignore_index=True)
                self.recommend_diff_df = pd.concat([self.recommend_diff_df, self.problem_difficulty[self.problem_difficulty['model_sha'] == model_sha]], ignore_index=True)
                return self.recommend_pb_df, self.recommend_diff_df

        # 如果没有找到, 返回 None
        return None

    def get_recomm_problem_nearest(self, ability, top_n=5):
        """
        根据给定的 ability 找到最接近的 top_n 个 difficulty, 同时确保这些题目没有被做过. 
        """
        # 计算 difficulty 与 ability 的差值
        sorted_difficulties = self.problem_difficulty.copy()
        sorted_difficulties['diff'] = abs(sorted_difficulties['difficulty'] - ability)
        sorted_difficulties = sorted_difficulties.sort_values(by='diff')

        selected_problems = []
        # 初始化结果 DataFrame
        recommended_problems = pd.DataFrame()
        recommended_difficulties = pd.DataFrame()
        for _, row in sorted_difficulties.iterrows():
            model_sha = row['model_sha']
            if model_sha not in self.recommend_pb_df['model_sha'].values:
                # 找到未做过的题目, 加入推荐列表
                selected_problems.append(model_sha)

                # 更新推荐数据
                recommended_problems = pd.concat(
                    [recommended_problems, self.performance_df[self.performance_df['model_sha'] == model_sha]],
                    ignore_index=True
                )
                recommended_difficulties = pd.concat(
                    [recommended_difficulties, self.problem_difficulty[self.problem_difficulty['model_sha'] == model_sha]],
                    ignore_index=True
                )

                # 如果已找到 top_n 个, 停止
                if len(selected_problems) == top_n:
                    break

        # 如果找到的少于 top_n 个, 可能会返回少于 5 个的结果
        return recommended_problems, recommended_difficulties
    def get_recomm_problem_farthest(self,vec,device, ability, top_n=5):
        """
        根据给定的 ability 找到最接近的 top_n 个 difficulty, 同时确保这些题目没有被做过. 
        """
        # 计算 difficulty 与 ability 的差值
        difficulties = self.problem_difficulty.copy()
        ldf=np.log(1/0.2-1)+ability
        sdf=np.log(1/0.8-1)+ability
        candidates=difficulties[(difficulties['difficulty']>sdf)&(difficulties['difficulty']<ldf)&(~(difficulties['model_sha'].isin(self.recommend_pb_df['model_sha'].values)))]['model_sha'].tolist()
        
        if len(candidates)<top_n:
            candidates=difficulties[(~(difficulties['sorted_index'].isin(self.recommend_pb_df['model_sha'].values)))]['model_sha'].tolist()
        id_to_index = {prob_id: idx for idx, prob_id in enumerate(self.idxlist)}
        history_indices = torch.tensor([id_to_index[h] for h in self.recommend_pb_df['model_sha'].values], device=device)
        candidate_indices = torch.tensor([id_to_index[c] for c in candidates], device=device)
        history_vectors = vec[history_indices]
        candidates_vectors = vec[candidate_indices]   
        dists = torch.cdist(candidates_vectors, history_vectors)
        min_dists = dists.min(dim=1).values 
        del history_indices, candidate_indices, history_vectors, candidates_vectors, dists
        
        values, indices = torch.topk(min_dists, k=top_n, largest=True)
        max_indices = indices.tolist()
        max_idx = [candidates[i] for i in max_indices]
        recommended_problems=self.performance_df[self.performance_df['model_sha'].isin(max_idx)].copy()
        recommended_problems.reset_index(inplace=True)
        recommended_difficulties=self.problem_difficulty[self.problem_difficulty['model_sha'].isin(max_idx)].copy()
        recommended_difficulties.reset_index(inplace=True)

        return recommended_problems, recommended_difficulties


    def get_recomm_problem_all(self):
        """
        根据给定的 ability 找到所有未做过的题目, 避免排序. 
        """
        # 筛选未做过的题目
        done_set = set(self.recommend_pb_df['model_sha'])
        filtered_difficulties = self.problem_difficulty[~self.problem_difficulty['model_sha'].isin(done_set)]
        
        # 获取推荐的 model_sha
        recommended_sha = filtered_difficulties['model_sha'].values

        # 直接批量筛选数据
        recommended_problems = self.performance_df[self.performance_df['model_sha'].isin(recommended_sha)]
        recommended_difficulties = self.problem_difficulty[self.problem_difficulty['model_sha'].isin(recommended_sha)]

        return recommended_problems, recommended_difficulties

    def get_recomm_problem_quantile5(self, ability): #返回每个区间（20, 40, 60, 80, 100）的题目
        """
        根据给定的 ability 找到最接近的 difficulty, 同时确保该题目没有被做过, 
        并且每个做对概率区间（0~20%, 20~40%, 40~60%, 60~80%, 80~100%）最多选择一题. 
        """
        # 计算每道题的做对概率
        self.problem_difficulty['correct_prob'] = 1 / (1 + np.exp(-(ability - self.problem_difficulty['difficulty'])))
        
        # 定义做对概率区间
        prob_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        prob_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        
        # 将题目按做对概率区间分类
        self.problem_difficulty['prob_bin'] = pd.cut(self.problem_difficulty['correct_prob'], bins=prob_bins, labels=prob_labels)
        
        # 初始化结果 DataFrame
        recommended_problems = pd.DataFrame()
        recommended_difficulties = pd.DataFrame()
        
        # 遍历每个区间, 选择最多一题
        for bin_label in prob_labels:
            # 获取当前区间的题目
            bin_problems = self.problem_difficulty[self.problem_difficulty['prob_bin'] == bin_label]
            
            # 按 difficulty 与 ability 的差值排序
            bin_problems['diff'] = abs(bin_problems['difficulty'] - ability)
            bin_problems = bin_problems.sort_values(by='diff')
            
            # 遍历当前区间的题目, 找到未做过的题目
            for _, row in bin_problems.iterrows():
                model_sha = row['model_sha']
                if model_sha not in self.recommend_pb_df['model_sha'].values:
                    # 找到未做过的题目
                    recommended_problems = pd.concat([recommended_problems, self.performance_df[self.performance_df['model_sha'] == model_sha]], ignore_index=True)
                    recommended_difficulties = pd.concat([recommended_difficulties, self.problem_difficulty[self.problem_difficulty['model_sha'] == model_sha]], ignore_index=True)
                    break  # 每个区间最多选择一题
        
        # 如果有推荐的题目, 返回结果
        if not recommended_problems.empty:
            return recommended_problems, recommended_difficulties
        
        # 如果没有找到, 返回 None
        return None
    
    def get_unattempted_problem_quantiles(self):
        """
        在未做过的题目中, 选择 difficulty 排名处于 0.1、0.3、0.5、0.7、0.9 分位数的题目. 
        """
        # 过滤出未做过的题目
        unattempted_problems = self.problem_difficulty[~self.problem_difficulty['model_sha'].isin(self.recommend_pb_df['model_sha'])]
        
        if unattempted_problems.empty:
            return None
        
        # 按 difficulty 排序
        unattempted_problems = unattempted_problems.sort_values(by='difficulty').reset_index(drop=True)
        
        # 计算分位数对应的索引
        quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
        quantile_indices = [int(q * len(unattempted_problems)) for q in quantiles]
        
        # 初始化结果 DataFrame
        recommended_problems = pd.DataFrame()
        recommended_difficulties = pd.DataFrame()
        
        # 遍历每个分位数索引, 选择对应的题目
        for idx in quantile_indices:
            if idx < len(unattempted_problems):
                selected_problem = unattempted_problems.iloc[[idx]]
                model_sha = selected_problem.iloc[0]['model_sha']
                recommended_problems = pd.concat([recommended_problems, self.performance_df[self.performance_df['model_sha'] == model_sha]], ignore_index=True)
                recommended_difficulties = pd.concat([recommended_difficulties, self.problem_difficulty[self.problem_difficulty['model_sha'] == model_sha]], ignore_index=True)
        
        # 如果有推荐的题目, 返回结果
        if not recommended_problems.empty:
            return recommended_problems, recommended_difficulties
        
        # 如果没有找到, 返回 None
        return None


    def update_recommended_pb_diff(self, model_sha):
        """
        更新已推荐题目 DataFrame, 加入制定的 model_sha（支持单个值或列表）. 

        Args:
            model_sha (str or list): 需要更新的 model_sha, 可以是单个值或列表. 

        Returns:
            tuple: 更新后的 recommend_pb_df 和 recommend_diff_df. 
        """
        # 处理单个 model_sha 或列表的情况
        if isinstance(model_sha, list):
            # 如果 model_sha 是列表, 使用 isin 方法筛选
            recommend_pb_df_to_add = self.performance_df[self.performance_df['model_sha'].isin(model_sha)]
            recommend_diff_df_to_add = self.problem_difficulty[self.problem_difficulty['model_sha'].isin(model_sha)]
        else:
            # 如果 model_sha 是单个值, 使用 == 筛选
            recommend_pb_df_to_add = self.performance_df[self.performance_df['model_sha'] == model_sha]
            recommend_diff_df_to_add = self.problem_difficulty[self.problem_difficulty['model_sha'] == model_sha]
        
        # 更新 DataFrame
        self.recommend_pb_df = pd.concat([self.recommend_pb_df, recommend_pb_df_to_add], ignore_index=True)
        self.recommend_diff_df = pd.concat([self.recommend_diff_df, recommend_diff_df_to_add], ignore_index=True)
        
        return self.recommend_pb_df, self.recommend_diff_df