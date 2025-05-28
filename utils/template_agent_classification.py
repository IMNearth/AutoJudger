def initialize_prompt_step1(initial_problems,  include_text=True, include_image=True):

    initial_problems_content = []
    for i, prob in enumerate(initial_problems):
        problem_content = []
        if include_text:
            problem_content.append({
                "type": "text",
                "text": f"""
                    Question ID: {prob['index']}
                    Difficulty: {prob['difficulty']}
                    Content: {prob['question']}
                    """
            })
        if include_image:
            problem_content.append({
                "type": "image",
                "image": f"data:image/jpeg;base64,{prob['image']}",
                "resized_height": 180,
                "resized_width": 270,
            })
        if include_text:
            problem_content.append({
                "type": "text",
                "text": "Options:\n" + "\n".join(
                    f"{opt}: {text}" 
                    for opt, text in sorted(prob['options'].items())
                ) + f"\nCorrect: {'Yes' if prob['correct'] else 'No'}"
            })

        initial_problems_content.extend(problem_content)
    
    # Prompt内容
    prompt = [
        {
            "role": "system",
            "content": [
                { 
                    "type": "text",
                    "text": """You are an expert educational AI assistant specializing in question classification. 
                    Your task is to analyze the provided questions and categorize them into meaningful subject/topic categories."""
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
                        Task Overview:
                        You will analyze a set of practice questions (including both text and images) and classify each question into a meaningful category(Expect two or more question in the same category).
                        The output should be a JSON object mapping question IDs to their respective categories.
                        """
                },
                *initial_problems_content,
                {
                    "type": "text",
                    "text": f"""
                        Output Requirements:
                        - Return a JSON object with the following format:
                          {{
                              "<Question_ID_1>": "<Category_Name_1>",
                              "<Question_ID_2>": "<Category_Name_2>",
                              ...
                          }}
                          - Keys are question IDs (index) from the input data.
                          - Values are descriptive category names that you assign.
                        - ONLY return the JSON object; do not include any other text or explanation.
                        """
                }
            ]
        }
    ]
    
    return prompt



def summary_prompt_step2_md(summary,candidate_problems, ability, include_text, include_image):
    initial_problems_content = []
    # #summary结构化
    # for i, prob in summary.iterrows():
    #     problem_content = []
        
    #     problem_content.append({
    #         "type": "text",
    #         "text": f"""
    #             Class: {prob['CLASS']}
    #             Question Number: {prob['question_count']}
    #             max_difficulty: {prob['max_difficulty']}
    #             min_difficulty: {prob['min_difficulty']}
    #             mean_difficulty: {prob['avg_difficulty']}
    #             accuracy: {prob['accuracy']}
    #             """
    #     })
        

    #     initial_problems_content.extend(problem_content)
    problem_content = []
    problem_content.append({
            "type": "text",
            "text": f"""
                {summary.to_markdown()}
                """
        })
    initial_problems_content.extend(problem_content)
    # 候选题目
    candidate_problems_content = []
    candidate_problems_ID = []
    for prob in candidate_problems:
  
        problem_content = []
        if include_text:
            problem_content.append({
                "type": "text",
                "text": f"""
                    Question ID: {prob['index']}
                    Difficulty: {prob['difficulty']}
                    Content: {prob['question']}
                """
            })
        if include_image:
            problem_content.append({
                "type": "image",
                "image": f"data:image/jpeg;base64,{prob['image']}",
                "resized_height": 280,
                "resized_width": 420,
            })
        if include_text:
            problem_content.append({
                "type": "text",
                "text": "Options:\n" + "\n".join(
                    f"{k}: {v}" 
                    for k, v in sorted(prob['options'].items())
                )
            })
        candidate_problems_content.extend(problem_content)
        candidate_problems_ID.append(prob['index'])
    # Prompt内容
    prompt = [
        {
            "role": "system",
            "content": [
                { 
                    "type": "text",
                    "text": f"""You are an expert educational AI assistant. 
                    Your task is to select the most appropriate next question from the candidate pool based on:
                    1. The student's current ability ({ability}) estimated by IRT
                    2. The diversity of question categories in the history
                    3. The match between question difficulty and student ability
                    Prioritize questions that balance category diversity and difficulty alignment."""
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Statistics in history questions
                        """
                },
                *initial_problems_content,
                {
                    "type": "text",
                    "text": '''
                            Candidate Question Pool:'''
                },
                *candidate_problems_content,  # 候选题目内容
                {
                    "type": "text",
                    "text": f"""Available IDs: {candidate_problems_ID}
                        Output JSON format:
                        {{
                            "summary": "Summary the Statistics in history questions",
                            "think": "Reasoning here",
                            "question_index": "SELECTED_ID"
                        }}
                        Only return the JSON object. DO NOT explain.
                        """
                }
            ]
        }
    ]
    
    return prompt

def summary_prompt_step1_multi(summary,last_prob , include_text, include_image):
    category=summary['CLASS'].unique().tolist()
    initial_problems_content = []
    for  prob in last_prob:
        problem_content = []
        if include_text:
            problem_content.append({
                "type": "text",
                "text": f"""
                    Question ID: {prob['index']}
                    Difficulty: {prob['difficulty']}
                    Content: {prob['question']}
                    """
            })
        if include_image:
            problem_content.append({
                "type": "image",
                "image": f"data:image/jpeg;base64,{prob['image']}",
                "resized_height": 180,
                "resized_width": 270,
            })
        if include_text:
            problem_content.append({
                "type": "text",
                "text": "Options:\n" + "\n".join(
                    f"{opt}: {text}" 
                    for opt, text in sorted(prob['options'].items())
                ) + f"\nCorrect: {'Yes' if prob['correct'] else 'No'}"
            })

        initial_problems_content.extend(problem_content)
    
    # Prompt内容
    prompt = [
        {
            "role": "system",
            "content": [
                { 
                    "type": "text",
                    "text": """You are an expert educational classifier. Analyze the question and determine its category."""
                }
            ]
        },
        {
            "role": "user",
            "content": [
                *initial_problems_content,
                {
                "type": "text",
                "text": f"""
                Task: Review the question above. Determine all applicable categories from the existing list: {category}, or include new categories if necessary.

                Output Requirements:
                - Return a JSON object with:
                {{
                "category": ["Existing or new category name(s)"]
                }}
                - List ALL relevant categories (minimum 1 item)
                - Use EXACT names for existing categories
                - Include multiple entries if needed (e.g., mixed existing/new categories)
                - Do NOT add explanations, only JSON
                """
                }
            ]
        }
    ]
    
    return prompt
