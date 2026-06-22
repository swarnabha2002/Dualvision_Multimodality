import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.train.train import DEGRADATION_TYPES, apply_degradation, expand2square

from PIL import Image
import math


def _aug_for_subfolder(name):
    """Map a subfolder name (e.g. 'fog092') back to the augment dict that produced it."""
    for aug in DEGRADATION_TYPES:
        if aug["_id"] == name:
            return aug
    raise ValueError(f"Unknown degradation subfolder: {name!r}")


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)

    if args.subfolder != "":
        answers_file = answers_file.replace(".jsonl", f"_{args.subfolder}.jsonl")


    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        category = line["category"]
        

        #now we dont care about ir-only questions
        if category == "ir":
            continue
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        gen_kwargs = {}
        if args.use_features:
            file_name = os.path.basename(image_file).replace('.jpg', '.pt')
            m1_feat_path = os.path.join(f"{args.data_root_dir}/HDRT/feats/ir_clip", file_name)
            if args.subfolder == "":
                img_feat_path = os.path.join(f"{args.data_root_dir}/HDRT/feats/rgb_clip", file_name)
            else:
                img_feat_path = os.path.join(f"{args.data_root_dir}/HDRT/feats/rgb_clip_aug/{args.subfolder}", file_name)

            assert os.path.exists(img_feat_path), f"Image feature file {img_feat_path} does not exist"
            assert os.path.exists(m1_feat_path), f"M1 feature file {m1_feat_path} does not exist"

            img_feat = torch.load(img_feat_path)
            m1_feat = torch.load(m1_feat_path)
            if img_feat.ndim != 3:
                img_feat = img_feat.unsqueeze(0)
            if m1_feat.ndim != 3:
                m1_feat = m1_feat.unsqueeze(0)
            gen_kwargs['img_feats'] = img_feat.cuda().half()
            gen_kwargs['m1_feats'] = m1_feat.cuda().half()
        else:
            # image mode: load raw images, apply degradation if subfolder is set
            file_name_jpg = os.path.basename(image_file)
            rgb_path = os.path.join(f"{args.data_root_dir}/HDRT/visible", file_name_jpg)
            ir_path = os.path.join(f"{args.data_root_dir}/HDRT/infrared", file_name_jpg)
            assert os.path.exists(rgb_path), f"RGB image missing: {rgb_path}"
            assert os.path.exists(ir_path), f"IR image missing: {ir_path}"
            rgb_pil = Image.open(rgb_path).convert('RGB')
            if args.subfolder != "":
                rgb_pil = apply_degradation(rgb_pil, _aug_for_subfolder(args.subfolder))
            ir_pil = Image.open(ir_path).convert('RGB')
            if args.image_aspect_ratio == 'pad':
                bg = tuple(int(x * 255) for x in image_processor.image_mean)
                rgb_pil = expand2square(rgb_pil, bg)
                ir_pil = expand2square(ir_pil, bg)
            rgb_px = image_processor.preprocess(rgb_pil, return_tensors='pt')['pixel_values'].cuda().half()
            ir_px = image_processor.preprocess(ir_pil, return_tensors='pt')['pixel_values'].cuda().half()
            gen_kwargs['rgb_pixels'] = rgb_px
            gen_kwargs['ir_pixels'] = ir_px

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=torch.zeros(2,2),
                image_sizes=None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=90,
                use_cache=True,
                **gen_kwargs)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "image": image_file,
                                   "generated_answer": outputs,
                                   "true_answer": line.get("ground_truth_answer", ""),
                                   "category": category,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--data_root_dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--answers-file", type=str)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--subfolder", type=str, default="")
    parser.add_argument("--use_features", type=lambda x: str(x).lower() == "true",
                        default=False,
                        help="True: load precomputed CLIP features (.pt). False (default): load images and run vision tower.")
    parser.add_argument("--image_aspect_ratio", type=str, default="pad",
                        choices=["pad", "square"],
                        help="Image-mode only: 'pad' = expand2square (no cropping); 'square' = processor default (resize+crop).")
    args = parser.parse_args()

    eval_model(args)
