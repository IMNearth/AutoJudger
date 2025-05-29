import argparse
import os
import clip
import torch
from PIL import Image
import numpy as np
import pandas as pd
from io import BytesIO
from typing import List, Dict, Any
import base64


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings using CLIP model")
    parser.add_argument('--benchmark', type=str, required=True)
    parser.add_argument('--download_root', type=str)
    parser.add_argument('--data_dir', type=str, default="LMUData/")
    parser.add_argument('--mode', type=str, choices=['text', 'image', 'multimodal'], default='multimodal')
    parser.add_argument('--fusion_method', type=str, choices=['mean', 'sum', 'concat'], default='concat')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./')
    parser.add_argument('--model_name', type=str, default='ViT-B/32')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    model, preprocess = ni(args.model_name, device=device, download_root=args.download_root)
    data_path = f"{args.data_dir}/{args.benchmark}.tsv"
    prob_data = pd.read_csv(data_path)
    all_possible_options = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    ini_data_list = []

    for index, row in prob_data.iterrows():
        existing_options = [opt for opt in all_possible_options if opt in row.index]
        options_dict = {}

        if len(existing_options) == 0:
            answer_options = set(row['answer'].upper().replace(' ', '').split(','))
            sorted_options = sorted(answer_options)
            for i, opt in enumerate(sorted_options):
                options_dict[chr(ord('A') + i)] = opt
        else:
            for opt in existing_options:
                if not pd.isna(row[opt]):
                    options_dict[opt] = row[opt]

        data_dict = {
            'index': row['index'],
            'question': row['question'],
            'options': options_dict,
            'image': row['image'],
        }
        ini_data_list.append(data_dict)

    def generate_embeddings(
        data_list: List[Dict[str, Any]],
        mode: str = 'text',
        model=None,
        preprocess=None,
        device: str = 'cuda',
        fusion_method: str = 'mean',
        normalize: bool = True
    ) -> np.ndarray:

        if mode not in ['text', 'image', 'multimodal']:
            raise ValueError("mode must be 'text', 'image', or 'multimodal'")
        if mode in ['image', 'multimodal'] and preprocess is None:
            raise ValueError("preprocess function is required for image/multimodal mode")

        text_embeddings = None
        image_embeddings = None

        if mode in ['text', 'multimodal']:
            texts = []
            for data in data_list:
                question = data['question']
                options = data['options']
                options_text = " ".join([f"{k}:{v}" for k, v in options.items()])
                text = f"Question: {question} Options: {options_text}"
                texts.append(text)
            text_tokens = clip.tokenize(texts, truncate=True).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
            text_embeddings = text_features.cpu().numpy()
            if normalize:
                text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

        if mode in ['image', 'multimodal']:
            images = []
            for data in data_list:
                try:
                    if data['image'].startswith('data:image'):
                        base64_str = data['image'].split(',', 1)[1]
                    else:
                        base64_str = data['image']
                    image_bytes = base64.b64decode(base64_str)
                    image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    image = preprocess(image)
                except Exception as e:
                    image = torch.zeros_like(preprocess(Image.new('RGB', (224, 224))))
                images.append(image)
            image_batch = torch.stack(images).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_batch)
            image_embeddings = image_features.cpu().numpy()
            if normalize:
                image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

        if mode == 'multimodal':
            if fusion_method == 'sum':
                embeddings = text_embeddings + image_embeddings
            elif fusion_method == 'mean':
                embeddings = (text_embeddings + image_embeddings) / 2
            elif fusion_method == 'concat':
                embeddings = np.concatenate([text_embeddings, image_embeddings], axis=1)
            else:
                raise ValueError("Supported fusion methods: 'sum', 'mean', or 'concat'")
            if normalize:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        else:
            embeddings = text_embeddings if mode == 'text' else image_embeddings

        return np.array(embeddings)

    embeddings = generate_embeddings(
        data_list=ini_data_list,
        mode=args.mode,
        model=model,
        preprocess=preprocess,
        device=device,
        fusion_method=args.fusion_method
    )

    save_path = f"{args.save_dir}/{args.benchmark}_{args.mode}"
    if args.mode == 'multimodal':
        save_path += f"_{args.fusion_method}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path + ".npy", embeddings)
    print(f"Embeddings saved to: {save_path}.npy")
