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
def id_exist(data_folder,id): #have figure or not
    if os.path.isdir(os.path.join(data_folder,id)):
        return True
    else:
        return False
    
def id_processed(process_lst,id):
    if id in process_lst:
        return True
    else:
        return False
    
def prepare_csv(df,csv_path):
    exist_name = pd.read_csv(csv_path)
    for id,row in exist_name.iterrows():
        df['id_a'].append(row['id_a'])
        df['id_b'].append(row['id_b'])
        df['Product_a'].append(row['Product_a'])
        df['Product_b'].append(row['Product_b'])
        df['product_a_desc'].append(row['product_a_desc'])
        df['product_b_desc'].append(row['product_b_desc'])
        df['Intention'].append(row['Intention'])
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

def concatenate_images(images):
    width = max(img.width for img in images)
    total_height = sum(img.height for img in images) + 20 * (len(images) - 1)

    new_img = Image.new('RGB', (width, total_height), (0, 0, 0))

    current_height = 0
    for img in images:
        new_img.paste(img, (0, current_height))
        current_height += img.height + 20  # adding a 20px black bar

    return new_img

def intention_retrieval(intention_df,product_id):
    intention_row = intention_df.query(f'id == "{product_id}"')
    return intention_row 

def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    portion = args.portion
    total_groups = args.total_groups
    image = args.image
    resume = args.resume
    csv_path = 'product csv'
    intention_csv_path = 'feature csv'
    labelled_image_path = 'image_path'
    save_dir = 'save_directory'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f'{save_dir}/co_buy_intention_w_relation_intention_{portion}_{total_groups}.csv'
    df = pd.read_csv(csv_path)
    unit_len = len(df)//total_groups
    df = df[(portion-1)*unit_len:portion*unit_len] if portion!= total_groups else df[(portion-1)*unit_len:]
    df = df.reset_index(drop = True)
    intention_df = pd.read_csv(intention_csv_path)
    new_df = {'id_a':[],'id_b':[],'Product_a':[],'Product_b':[],'Relation':[],'product_a_desc':[],'product_b_desc':[],'Intention':[],'tail':[]}
    start_id = -100
    if start_id > 0:
        new_df = prepare_csv(new_df,save_path)
    if resume:
        new_df = prepare_csv(new_df,save_path)
        temp_df = pd.DataFrame(new_df)
        count = len(temp_df)
    for id,row in tqdm(df.iterrows(),desc = f'Name Clean Process {portion}',total=len(df)):
        if id % 1000==0:
            temp_df = pd.DataFrame(new_df)
            temp_df.to_csv(save_path,index = False)
        id_lst = []
        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
        ans_lst = []
        product_a_id = row['item_a_id']
        product_a_intent = intention_retrieval(intention_df,product_a_id)
        product_b_id = row['item_b_id']
        product_b_intent = intention_retrieval(intention_df,product_b_id)
        if product_a_intent.empty or product_b_intent.empty:
            continue
        product_a_intent = product_a_intent['Product_desc'].values[0]
        product_b_intent = product_b_intent['Product_desc'].values[0]
        prod_a_desc = row['item_a_name']
        prod_b_desc = row['item_b_name']
        product_a_dir = os.path.join(labelled_image_path,product_a_id)
        product_b_dir = os.path.join(labelled_image_path,product_b_id)
        relation = row['relation']
        tail = row['assertion']
        if not any(file.endswith('.jpg') for file in os.listdir(product_a_dir)):
            continue
        if not any(file.endswith('.jpg') for file in os.listdir(product_b_dir)):
            continue
        question_lst = [f'The two images are two different products. \
            The product name of the upper image is {prod_a_desc}. \
                The product detail and the potential purchase intention is {product_a_intent}. \
                    The product name of the lower image is {prod_b_desc}. \
                        The product detail and the potential purchase intention is {product_b_intent}. \
                            Based on information provided, together with the product images, \
                                what could be the potential intention for people buying these two products in one purchase simultanesouly based on the relation of {relation_prompt[relation]}, \
                                    take the image features into consideration, limit your word count within 120 words. \
                                        Start with the potential co-buy intention could be {relation_prompt[relation]}',
                        ]
        image_a ,image_b= load_image(os.path.join(product_a_dir,os.listdir(product_a_dir)[0])),load_image(os.path.join(product_b_dir,os.listdir(product_b_dir)[0]))  
        image = [concatenate_images([image_a,image_b])]
        image_tensor = process_images(image, image_processor, args)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            
        for ques_id,ques in enumerate(question_lst):
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

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
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
            ans_lst.append(outputs.replace('</s>',''))

            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        new_df['id_a'].append(product_a_id)
        new_df['id_b'].append(product_b_id)
        new_df['Product_a'].append(prod_a_desc)
        new_df['Product_b'].append(prod_b_desc)
        new_df['Relation'].append(relation)
        new_df['Intention'].append(ans_lst[0])
        new_df['product_a_desc'].append(product_a_intent)
        new_df['product_b_desc'].append(product_b_intent)
        new_df['tail'].append(tail)
    new_df = pd.DataFrame(new_df)
    new_df.to_csv(save_path,index = False)

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
    parser.add_argument("--portion", type = int,help = 'which portion of data to process')
    parser.add_argument("--total_groups",type = int,help = 'total groups of data to process')
    parser.add_argument("--image",action = 'store_true',help = 'whether image is considered')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)
