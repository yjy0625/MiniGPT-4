import argparse
import os
import random
import time
import json
import click
from glob import glob
from PIL import Image
from attrdict import AttrDict
from tqdm import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


minigpt4_cfg_path = os.path.join(os.path.dirname(__file__), '../eval_configs/minigpt4_eval.yaml')


class Workspace(object):
    def __init__(self, cfg_path, gpu_id, num_beams=1, temperature=1.0):
        args = AttrDict(
            options=[],
            cfg_path=cfg_path
        )
        cfg = Config(args)
        model_config = cfg.model_cfg
        model_config.device_8bit = gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(gpu_id))

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.chat = Chat(model, vis_processor, device='cuda:{}'.format(gpu_id))
        self.num_beams = num_beams
        self.temperature = temperature
        print('Initialization Finished')

    def reset(self, prompt, examples=[]):
        self.chat_state = CONV_VISION.copy()
        self.img_list = []
        if len(examples) > 0:
            for im, a in examples:
                self.upload_img(im)
                self.ask(f'{prompt}###Assistant: {a}')

    def upload_img(self, img):
        if img is None:
            return
        llm_message = self.chat.upload_img(img, self.chat_state, self.img_list)

    def ask(self, user_message):
        if len(user_message) == 0:
            assert ValueError('Input should not be empty.')
        self.chat.ask(user_message, self.chat_state)

    def answer(self, prefix=None, ensemble_index=0):
        llm_message = self.chat.answer(conv=self.chat_state,
                                       img_list=self.img_list,
                                       prefix=prefix,
                                       num_beams=self.num_beams,
                                       temperature=self.temperature,
                                       max_new_tokens=300,
                                       max_length=3000,
                                       ensemble_index=ensemble_index)[0]
        return llm_message

    def compute_prob(self, queries, prefix=None, query_index=0):
        scores = self.chat.compute_prob(conv=self.chat_state,
                                        img_list=self.img_list,
                                        queries=queries,
                                        prefix=prefix,
                                        query_index=query_index,
                                        num_beams=self.num_beams,
                                        temperature=self.temperature,
                                        max_new_tokens=300,
                                        max_length=3000)
        return scores

    def run(self, img, prompt, examples=[], reset=True,
            ensemble_index=0):
        if reset:
            self.reset(prompt, examples=examples)
        self.upload_img(img)
        self.ask(prompt)
        output = self.answer(ensemble_index=ensemble_index)
        '''
        candidates = ['high', 'low'] # ['No', 'Yes']
        scores = self.compute_prob(candidates, prefix='The likelihood is', query_index=0)
        print(f'Scores: high -- {scores[0]}, low -- {scores[1]}')
        output = candidates[torch.argmax(scores).detach().cpu().item()]
        '''
        return output


@click.command()
@click.option('--input_dir', type=str)
@click.option('--example_dir', type=str, default=None)
@click.option('--prompt', type=str)
@click.option('--cfg_path', type=str, default='../eval_configs/minigpt4_eval.yaml')
@click.option('--gpu_id', type=int, default=0)
@click.option('--is_eval', is_flag=True)
def main(input_dir, example_dir, prompt, cfg_path, gpu_id, is_eval):
    workspace = Workspace(cfg_path, gpu_id)
    paths = sorted(glob(os.path.join(input_dir, '*.jpg')))
    images = [Image.open(path) for path in paths]
    examples = []
    if example_dir is not None:
        example_paths = sorted(glob(os.path.join(example_dir, '*.jpg')))
        txt_paths = [p[:-4] + '.txt' for p in example_paths]
        for p, tp in zip(paths, txt_paths):
            img = Image.open(p)
            with open(tp, 'r') as f:
                txt = f.readlines()[0].strip()
            examples.append([img, txt])
    i = 0
    P, T = 0, 0
    info = []
    pbar = tqdm(zip(paths, images))
    for p, im in pbar:
        start = time.time()
        resp = workspace.run(im, prompt, examples=examples)
        if is_eval:
            label = 1 if p[:-4].endswith('_s') else 0
            if 'Yes' in resp:
                pred = 1
            elif 'No' in resp:
                pred = 0
            else:
                pred = -1
            T += 1
            if label == pred:
                P += 1
            info.append({'index': i, 'path': p, 'label': label, 'pred': pred, 'resp': resp})
            pbar.set_description(f'Image {p.split("/")[-1]} [{time.time() - start:.2f}s]: {resp}')
        else:
            print(f'Image {p.split("/")[-1]} [{time.time() - start:.2f}s]: {resp}')
        i += 1
    print(f'Accuracy: {P} / {T} = {P / T * 100:.2f}%')
    with open('out.json', "w") as f:
        json.dump(info, f)


if __name__ == '__main__':
    main()
