import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init, relation_prompt
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import os
import pandas as pd
import json
from tqdm import tqdm


# ========================================
#            Helper Functions
# ========================================
def id_exist(data_folder, id):  # have figure or not
    if os.path.isdir(os.path.join(data_folder, id)):
        return True
    else:
        return False


def id_processed(process_lst, id):
    if id in process_lst:
        return True
    else:
        return False


def prepare_csv(df, csv_path):
    exist_name = pd.read_csv(csv_path)
    for id, row in exist_name.iterrows():
        df['id_a'].append(row['id_a'])
        df['id_b'].append(row['id_b'])
        df['Product_a'].append(row['Product_a'])
        df['Product_b'].append(row['Product_b'])
        df['product_a_desc'].append(row['product_a_desc'])
        df['product_b_desc'].append(row['product_b_desc'])
        df['Intention'].append(row['Intention'])
        df['Cate'].append(row['Cate'])
        df['Relation'].append(row['Relation'])
        df['tail'].append(row['tail'])
    return df


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                           args.load_8bit, args.load_4bit,
                                                                           device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            '[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode,
                                                                                                              args.conv_mode,
                                                                                                              args.conv_mode))
    else:
        args.conv_mode = conv_mode

    portion = args.portion
    total_groups = args.total_groups
    image = args.image
    csv_path = 'intention_path'
    labelled_image_path = 'image_path'
    save_path = 'save_path'
    df = pd.read_csv(csv_path)
    unit_len = len(df) // total_groups
    df = df[(portion - 1) * unit_len:portion * unit_len] if portion != total_groups else df[(portion - 1) * unit_len:]
    new_df = {'id': [], 'Product_name': [], 'Product_desc': []}
    start_id = -100
    if start_id > 0:
        new_df = prepare_csv(new_df, save_path)
    for id, row in tqdm(df.iterrows(), desc=f'Name Clean Process {portion}', total=len(df)):
        if id < start_id:
            continue
        if id % 1000 == 0:
            temp_df = pd.DataFrame(new_df)
            temp_df.to_csv(save_path, index=False)
        id_lst = []
        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
        ans_lst = []
        product_id = row['id']
        prod_desc = row['name']
        product_dir = os.path.join(labelled_image_path, product_id)
        if not os.path.exists(product_dir) or not any(file.endswith('.jpg') for file in os.listdir(product_dir)):
            continue
        question_lst = [
            f'The image contains a product and name of it is {prod_desc}. \
                Please analyze the product image, together with the product name, \
                provide a detailed description focusing on the product\'s features, design, and apparent quality. \
                    Highlight any unique characteristics or visible elements that distinguish this product from similar items. \
                        Additionally, speculate on the potential uses and benefits of this product for a consumer, \
                            based on its appearance or any information in the image and the name.',
        ]
        image = [load_image(os.path.join(product_dir, os.listdir(product_dir)[0]))]
        image_tensor = process_images(image, image_processor, args)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        for ques_id, ques in enumerate(question_lst):
            try:
                inp = f"{roles[0]}: {ques}"
            except EOFError:
                inp = ""

            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = outputs
            ans_lst.append(outputs.replace('</s>', ''))

            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        new_df['id'].append(product_id)
        new_df['Product_name'].append(prod_desc)
        new_df['Product_desc'].append(ans_lst[0])
    new_df = pd.DataFrame(new_df)
    new_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--portion", type=int, help='which portion of data to process')
    parser.add_argument("--total_groups", type=int, help='total groups of data to process')
    parser.add_argument("--image", action='store_true', help='whether image is considered')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)
