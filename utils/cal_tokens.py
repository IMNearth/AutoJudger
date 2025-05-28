import base64
from PIL import Image
from io import BytesIO
import tiktoken


def calculate_image_tokens(base64_image):
    # 将 Base64 字符串解码为二进制数据
    image_data = base64.b64decode(base64_image)
    
    # 使用 PIL 打开图像并获取分辨率
    image = Image.open(BytesIO(image_data))
    width, height = image.size
    
    # 计算 tiles
    if width > 2048 or height > 2048:
        # 如果分辨率超过 2048，先缩放到 2048
        if width > height:
            width, height = 2048, int(height * (2048 / width))
        else:
            width, height = int(width * (2048 / height)), 2048
    
    tiles = (width / 512) * (height / 512)
    
    # 计算 token 数量
    if tiles <= 1:
        return 85  # 低分辨率图像
    else:
        return 85 + (tiles * 170)


def calculate_text_tokens(text):
    # 初始化 tokenizer
    encoder = tiktoken.encoding_for_model("gpt-4")  # 根据模型选择
    tokens = encoder.encode(text)
    return len(tokens)


if __name__ == "__main__":
    # 示例数据
    data_dict = {
        'question': "What is the capital of France?",
        'answer': "B",
        'options': {
            'A': "London",
            'B': "Paris",
            'C': "Berlin",
            'D': "Madrid"
        },
        'image': "base64_encoded_image_string"  # 假设这是图像的 Base64 编码
    }

    # 构建提示文本
    prompt_text = f"""
        Question: {data_dict['question']}
        Options:
        A: {data_dict['options']['A']}
        B: {data_dict['options']['B']}
        C: {data_dict['options']['C']}
        D: {data_dict['options']['D']}
        
        Analyze the question and image to determine why the correct answer is {data_dict['answer']}. Please provide a detailed explanation.
    """

    # 计算 token 数量
    token_count = calculate_text_tokens(prompt_text)
    print(token_count)

    