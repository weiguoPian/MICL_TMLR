import warnings
warnings.filterwarnings("ignore")

import torch

from dataloader import flickr30kLoader, OKVQALoader, MSRVTTLoader, AudioCapsLoader, ClothoAQALoader, MSVDQALoader
from torch.utils.data import DataLoader


from config import Config
from lavis.common.registry import registry

from model import get_model_only_proj_from_config

from tools import VQAEval

from omegaconf import OmegaConf

import argparse

from lr_scheduler import *

from tqdm import tqdm
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import random

from pycocoevalcap.cider.cider import Cider

from torch.nn import functional as F

import copy

from datetime import datetime
import subprocess

import json

def get_optimizer(run_config, model):
    lr_scale = run_config.get("lr_layer_decay", 1)
    weight_decay = run_config.get("weight_decay", 0.05)

    optim_params = model.get_optimizer_params(weight_decay, lr_scale)

    num_parameters = 0
    for p_group in optim_params:
        for p in p_group["params"]:
            num_parameters += p.data.nelement()    
    logging.info("number of trainable parameters: {}".format(num_parameters))

    beta2 = run_config.get("beta2", 0.999)

    optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(run_config.init_lr),
        betas=(0.9, beta2),
    )

    return optimizer

def get_lr_scheduler(run_config, optimizer):
    max_epoch = run_config.max_epoch
    min_lr = run_config.min_lr
    init_lr = run_config.init_lr

    decay_rate = run_config.get("lr_decay_rate", None)
    warmup_start_lr = run_config.get("warmup_lr", -1)
    warmup_steps = run_config.get("warmup_steps", 0)

    lr_sched = run_config.lr_sched
    if lr_sched == 'linear_warmup_step_lr':
        _lr_sched = LinearWarmupStepLRScheduler(
            optimizer=optimizer,
            max_epoch=max_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            decay_rate=decay_rate,
            warmup_start_lr=warmup_start_lr,
            warmup_steps=warmup_steps,
        )
    elif lr_sched == 'linear_warmup_cosine_lr':
        _lr_sched = LinearWarmupCosineLRScheduler(
            optimizer=optimizer,
            max_epoch=max_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            decay_rate=decay_rate,
            warmup_start_lr=warmup_start_lr,
            warmup_steps=warmup_steps,
        )
    else:
        _lr_sched = ConstantLRScheduler(
            optimizer=optimizer,
            max_epoch=max_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            decay_rate=decay_rate,
            warmup_start_lr=warmup_start_lr,
            warmup_steps=warmup_steps,
        )

    return _lr_sched


def load_pseudo_qa_dict(modalities, text_processor):
    # text_processor.task = 'qa'
    # text_processor.prompt = ''
    
    # if len(modalities) != 1:
    #     text_processor.modality = 'audio-video'
    #     idx_pseudo_qa_dict = np.load('./pseudo_targets/MSRVTT_pseudo_qa_dict.npy', allow_pickle=True).item()
    # else:
    modal = modalities[0]
    # text_processor.modality = modal
    if modal == 'audio':
        idx_pseudo_qa_dict = np.load('./pseudo_targets/AudioCaps_pseudo_qa_dict.npy', allow_pickle=True).item()
    elif modal == 'image':
        idx_pseudo_qa_dict = np.load('./pseudo_targets/flickr30k_pseudo_qa_dict.npy', allow_pickle=True).item()
    elif modal == 'video':
        idx_pseudo_qa_dict = np.load('./pseudo_targets/MSRVTT_pseudo_qa_dict.npy', allow_pickle=True).item()
    else:
        raise ValueError('Unrecognized task type: {}'.format(curr_task))
    
    return idx_pseudo_qa_dict

def generate_pseudo_captions(model, dataset, modalities, text_processor, run_config):
    loader = DataLoader(dataset, batch_size=16, num_workers=8,
                        pin_memory=True, drop_last=False, shuffle=False)
    device = model.device
    modal = modalities[0]
    # text_processor.task = 'caption'
    # if len(modalities) == 1:
    #     modal = modalities[0]
    #     text_processor.prompt = 'describe the {}.'.format(modal)
    #     text_processor.modality = modal
    # else:
    #     text_processor.prompt = 'describe the audio-video.'
    #     text_processor.modality = 'audio-video'
    idx_pseudo_target_dict = {}
    for data in tqdm(loader):
        if len(modalities) == 1:
            index, data_input, text_input, text_output = data
            # print('index', index)
            # print('type(index)', type(index))
            BS = data_input.shape[0]
            data_input = data_input.to(device)
            sample = {modal: data_input, 'text_input': [text_processor("") for i in range(BS)], 'modalities': modalities}
        else:
            index, audio_input, video_input, text_input, text_output = data
            audio_input = audio_input.to(device)
            video_input = video_input.to(device)
            sample = {"audio": audio_input, 'video': video_input, 'text_input': [text_processor("")], 'modalities': modalities}
        
        with torch.cuda.amp.autocast(dtype=torch.float16):
            pred_captions, pred_ids = model.generate(
                sample,
                use_nucleus_sampling=False,
                num_beams=run_config.get("num_beams", 5),
                max_length=run_config.get("max_len", 30),
                min_length=run_config.get("min_len", 1),
                repetition_penalty=run_config.get("repetition_penalty", 1.15),
                length_penalty=run_config.get("length_penalty", 0.),
                top_p=run_config.get("top_p", 0.9),
                temperature=run_config.get("temperature", 1.),
            )
        for i in range(len(index)):
            idx_pseudo_target_dict[index[i]] = text_processor.pre_caption(pred_captions[i]).lower().strip()
        # idx_pseudo_target_dict[index[0]] = text_processor.pre_caption(pred_captions[0]).lower().strip()
    
    return idx_pseudo_target_dict


def generate_pseudo_instruct_answers(model, data_set, text_processor, run_config):
    instruct_answers = []

    logging.info('Generating pseudo instruct answers...')
    with torch.no_grad():
        for data in tqdm(data_set):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                pred_instruct_answer, pred_ids = model.generate(
                    {'text_input': [data]},
                    use_nucleus_sampling=False,
                    num_beams=run_config.get("num_beams", 5),
                    max_length=run_config.get("max_len", 30),
                    min_length=run_config.get("min_len", 1),
                    repetition_penalty=run_config.get("repetition_penalty", 1.15),
                    length_penalty=run_config.get("length_penalty", 0.),
                    top_p=run_config.get("top_p", 0.9),
                    temperature=run_config.get("temperature", 1.),
                )
            instruct_answers.append(pred_instruct_answer)

    return instruct_answers
    

def llm_fusion(old_model, curr_model, run_config, learned_modalities):
    alpha_fusion = run_config.alpha_fusion
    projection_fusion = run_config.projection_fusion

    old_llm_lora_para = {n: p.data for n, p in old_model.llm_model.named_parameters()}
    for n, p in curr_model.llm_model.named_parameters():
        if p.requires_grad:
            # p.data = alpha_fusion * old_llm_lora_para[n] + (1 - alpha_fusion) * p.data
            p.data = (1 - alpha_fusion) * old_llm_lora_para[n] + alpha_fusion * p.data
    
    
    if projection_fusion:
        modalities = curr_model.modalities
        for modal in modalities:
            if modal in learned_modalities:
                old_modal_proj = getattr(old_model, '{}_projection'.format(modal))
                old_modal_proj_para = {n: p.data for n, p in old_modal_proj.named_parameters()}
                for n, p in getattr(curr_model, '{}_projection'.format(modal)).named_parameters():
                    p.data = (1 - alpha_fusion) * old_modal_proj_para[n] + alpha_fusion * p.data

                old_modal_ln = getattr(old_model, '{}_ln'.format(modal))
                old_modal_ln_para = {n: p.data for n, p in old_modal_ln.named_parameters()}
                for n, p in getattr(curr_model, '{}_ln'.format(modal)).named_parameters():
                    p.data = (1 - alpha_fusion) * old_modal_ln_para[n] + alpha_fusion * p.data
                
                if modal == 'video' or modal == 'audio':
                    old_modal_conv_pooling = getattr(old_model, '{}_conv_pooling'.format(modal))
                    old_modal_conv_pooling_para = {n: p.data for n, p in old_modal_conv_pooling.named_parameters()}
                    for n, p in getattr(curr_model, '{}_conv_pooling'.format(modal)).named_parameters():
                        p.data = (1 - alpha_fusion) * old_modal_conv_pooling_para[n] + alpha_fusion * p.data

    
    return curr_model


def freeze_proj(model):
    # print('======================== freeze modal projection ========================')
    modal = model.modalities[0]
    for n, p in getattr(model, '{}_projection'.format(modal)).named_parameters():
        p.requires_grad = False
    for n, p in getattr(model, '{}_ln'.format(modal)).named_parameters():
        p.requires_grad = False
    if modal == 'video' or modal == 'audio':
        for n, p in getattr(model, '{}_conv_pooling'.format(modal)).named_parameters():
            p.requires_grad = False
    return model

def unfreeze_proj(model):
    # print('======================== freeze modal projection ========================')
    modal = model.modalities[0]
    for n, p in getattr(model, '{}_projection'.format(modal)).named_parameters():
        p.requires_grad = True
    for n, p in getattr(model, '{}_ln'.format(modal)).named_parameters():
        p.requires_grad = True
    if modal == 'video' or modal == 'audio':
        for n, p in getattr(model, '{}_conv_pooling'.format(modal)).named_parameters():
            p.requires_grad = True
    return model


def train(run_config, models, trainset, valset, text_instruct_set, step, task, max_epoch, batch_size, modalities, text_processor, save_dir, learned_task_type, learned_modalities):
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=8,
                                pin_memory=True, drop_last=True, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if task.endswith('cap'):
        if modalities[0] == 'video':
            val_batch_size = 8
            val_workers = 8
        else:
            val_batch_size = 8
            val_workers = 8
    else:
        if modalities[0] == 'video':
            val_batch_size = 8
            val_workers = 8
        else:
            val_batch_size = 8
            val_workers = 8

    val_loader = DataLoader(valset, batch_size=val_batch_size, num_workers=val_workers,
                            pin_memory=True, drop_last=False, shuffle=False)
    
    if step > 0:
        model, old_model = models
        old_model = old_model.to(device)
        old_model.eval()
    else:
        model = models

    model = model.to(device)

    scaler = torch.cuda.amp.GradScaler()

    optimizer = get_optimizer(run_config, model)
    lr_scheduler = get_lr_scheduler(run_config, optimizer)

    steps_each_epoch = len(train_loader)
    update_interval = 100

    prompts_list = text_instruct_set['prompts']
    all_text_inputs_list = text_instruct_set['inputs']
    # all_instruct_input = [prompts_list[ins['prompt_id']].strip() + ' ' + ins['input'].strip() for ins in text_inputs_list]
    # all_instruct_input = [text_processor.pre_caption(ins) for ins in instruct_input]
    all_instruct_input = [text_processor.pre_caption(prompts_list[ins['prompt_id']].strip() + ' ' + ins['input'].strip()) for ins in all_text_inputs_list]

    alpha_fusion = run_config.alpha_fusion
    # alpha_fusion = (1 / (step + 1)) ** 0.5

    # all_instruct_answers = generate_pseudo_instruct_answers(old_model, all_instruct_input, text_processor, run_config)

    if step > 0 and run_cfg.pseudo:
        if task.endswith('qa') and 'cap' in learned_task_type:
            logging.info('Generating pseudo captions...')
            text_processor.task = 'caption'
            text_processor.prompt = 'describe the {}.'.format(modalities[0])
            text_processor.modality = modalities[0]
            idx_pseudo_target_dict = generate_pseudo_captions(old_model, trainset, modalities, text_processor, run_config)
            generated_type = 'cap'
            logging.info('Generating Done.')
        elif task.endswith('cap') and 'qa' in learned_task_type:
            logging.info('Generating pseudo qas...')
            text_processor.task = 'qa'
            text_processor.prompt = ''
            text_processor.modality = modalities[0]
            idx_pseudo_qa_dict = load_pseudo_qa_dict(modalities, text_processor)
            generated_type = 'qa'
            logging.info('Generating Done.')
        else:
            generated_type = None
    else:
        generated_type = None
    logging.info('Generated type: {}'.format(generated_type))

    # text_processor.prompt = 'describe the {}.'.format(modalities[0])
    # text_processor.modality = modalities[0]
    # text_processor.task = 'caption' if task.endswith('cap') else text_processor.task = 'qa'

    if step > 0 and run_cfg.freeze_proj:
        model = freeze_proj(model)

    BEST_SCORE = 0
    BEST_EPOCH = 0
    patience = 5
    # patience = 10
    T = 5.0
    lambda_kl = run_cfg.lambda_kl
    for epoch in range(max_epoch):
        if epoch - BEST_EPOCH > patience:
            print('======================== early stop ========================')
            break
        model.train()
        i = 0
        train_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(total=steps_each_epoch, desc='Train Epoch {}'.format(epoch))
        # for data in tqdm(train_loader):
        update_times = 0
        for data in train_loader:
            lr_scheduler.step(cur_epoch=epoch, cur_step=i)
            # if len(modalities) == 1:
            modal = modalities[0]
            index, data_input, text_input, text_output = data
            data_input = data_input.to(device)

            if step > 0 and run_cfg.pseudo and generated_type is not None:
                if generated_type == 'cap':
                    pseudo_targets = []
                    for idx in index:
                        # pseudo_targets.append(idx_pseudo_target_dict[idx.item()])
                        pseudo_targets.append(idx_pseudo_target_dict[idx])
                    
                    pseudo_start = batch_size - run_cfg.pseudo_num
                    # pseudo_data_input = data_input[pseudo_start:]
                    # pseudo_text_input = [text_processor("") for j in range(len(pseudo_data_input))]
                    data_input = torch.cat((data_input, data_input[pseudo_start:]), dim=0)
                    text_temps = [text_processor("") for j in range(len(text_input))]
                    text_input.extend(text_temps[pseudo_start:])
                    text_output.extend(pseudo_targets[pseudo_start:])

                    # data_input = torch.cat((data_input, data_input), dim=0)
                    # text_temps = [text_processor("") for j in range(len(text_input))]
                    # text_input.extend(text_temps)
                    # text_output.extend(pseudo_targets)
                else:
                    pseudo_q, pseudo_a = [], []
                    for idx in index:
                        pseudo_q.append(idx_pseudo_qa_dict[idx]['Q'])
                        pseudo_a.append(idx_pseudo_qa_dict[idx]['A'])
                        if len(pseudo_q) == run_cfg.pseudo_num:
                            break
                    data_input = torch.cat((data_input, data_input[:run_cfg.pseudo_num]), dim=0)
                    text_input.extend([text_processor(q_) for q_ in pseudo_q])
                    text_output.extend([text_processor.pre_caption(a_) for a_ in pseudo_a])
            
            samples = {modal: data_input, 'text_input': text_input, 'text_output': text_output}

            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(samples=samples)

            loss = out['loss']

            if step > 0 and run_cfg.pseudo and lambda_kl != 0:
                pseudo_samples = {
                    modal: data_input[-run_cfg.pseudo_num:], 'text_input': text_input[-run_cfg.pseudo_num:], 'text_output': text_output[-run_cfg.pseudo_num:]}
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        # old_logits = old_model(samples=samples)['logits'].detach()
                        old_logits = old_model(samples=pseudo_samples)['logits'].detach()
                        soft_target = F.softmax(old_logits / T, dim=1)
                logits = out['logits']
                logits = logits[-old_logits.shape[0]:]
                output_log = F.log_softmax(logits / T, dim=1)
                loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean') * (T**2)
                loss += lambda_kl * loss_KD

            if step > 0 and run_config.text_instruct_num > 0:
                if run_cfg.text_instruct_constraint_type == 'KL':
                    instruct_input = random.choices(all_instruct_input, k=run_config.text_instruct_num)
                    instruct_samples = {'text_input': instruct_input}
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            old_instruct_logits = old_model(samples=instruct_samples, with_text_out=False)['logits'].detach()
                            old_instruct_soft_target = F.softmax(old_instruct_logits / T, dim=1)
                    instruct_logits = model(samples=instruct_samples, with_text_out=False)['logits']
                    instruct_logits_log = F.log_softmax(instruct_logits / T, dim=1)
                    ins_loss_KD = F.kl_div(instruct_logits_log, old_instruct_soft_target, reduction='batchmean') * (T**2)
                    loss += ins_loss_KD
                else:
                    ins_idxs = random.sample(range(len(all_instruct_input)), run_config.text_instruct_num)
                    instruct_input = [all_instruct_input[idx] for idx in ins_idxs]

                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            pred_instruct_answers, pred_ids = old_model.generate(
                                {'text_input': instruct_input},
                                use_nucleus_sampling=False,
                                num_beams=run_config.get("num_beams", 5),
                                max_length=run_config.get("max_len", 30),
                                min_length=run_config.get("min_len", 1),
                                repetition_penalty=run_config.get("repetition_penalty", 1.15),
                                length_penalty=run_config.get("length_penalty", 0.),
                                top_p=run_config.get("top_p", 0.9),
                                temperature=run_config.get("temperature", 1.),
                            )
                    
                    instruct_samples = {'text_input': instruct_input, 'text_output': pred_instruct_answers}
                    # print(instruct_input)
                    # print('===================================')
                    # print(instruct_answers)
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        instruct_loss = model(samples=instruct_samples)['loss']
                    loss += instruct_loss

            scaler.scale(loss).backward()
            if (i + 1) % run_config.accum_grad_iters == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()
            i += 1

            if i % update_interval == 0:
                pbar.update(update_interval)
                update_times += 1
        bar_rest_to_update = steps_each_epoch - update_interval * update_times
        if bar_rest_to_update > 0:
            pbar.update(bar_rest_to_update)
        pbar.close()

        train_loss /= i
        logging.info("Epoch: {}  Training Loss: {}".format(epoch, train_loss))

        optimizer.zero_grad()

        if alpha_fusion != 1.0:
            model = llm_fusion(old_model, model, run_config, learned_modalities)

        logging.info("Starting validating...")

        if task.endswith('cap'):
            metric = 'CIDEr'
            val_res = evaluate_caption(run_config, model, val_loader, device, curr_modalities, mode='val')
        elif task.endswith('qa'):
            metric = 'Accuracy'
            val_res = evaluate_qa(run_config, model, val_loader, device, curr_modalities, mode='val')
        else:
            pass
        logging.info('Epoch: {}  Validation {}: {}'.format(epoch, metric, val_res))

        if val_res > BEST_SCORE:
            BEST_SCORE = val_res
            BEST_EPOCH = epoch
            logging.info("Saving best checkpoint at epoch {}. Best {}: {}.".format(epoch, metric, val_res))
            if step > 0 and run_cfg.freeze_proj:
                model = unfreeze_proj(model)
                save_checkpoint(model, path=save_dir, step=step, save_best=True)
                model = freeze_proj(model)
            else:
                save_checkpoint(model, path=save_dir, step=step, save_best=True)
    
    if step > 0 and run_cfg.freeze_proj:
        model = unfreeze_proj(model)


def evaluate_caption(run_config, model, loader, device, modalities, mode='val'):
    total_steps= len(loader)
    update_interval = 10
    pbar = tqdm(total=total_steps)

    all_pred_caption = []
    all_pred_ids = []
    all_gt_caption = []
    preds_gts_ = []

    model.eval()

    i = 0
    update_times = 0
    for data in loader:
        if len(modalities) == 1:
            modal = modalities[0]
            index, modal_input, text_input, txt_output = data
            modal_input = modal_input.to(device)
            samples = {modal: modal_input, 'text_input': text_input, 'modalities': modalities}
        else:
            index, audio_input, video_input, text_input, txt_output = data
            audio_input = audio_input.to(device)
            video_input = video_input.to(device)
            samples = {"audio": audio_input, 'video': video_input, 'text_input': text_input, 'modalities': modalities}
        
        with torch.cuda.amp.autocast(dtype=torch.float16):
            pred_captions, pred_ids = model.generate(
                samples,
                use_nucleus_sampling=False,
                num_beams=run_config.get("num_beams", 5),
                max_length=run_config.get("max_len", 30),
                min_length=run_config.get("min_len", 1),
                repetition_penalty=run_config.get("repetition_penalty", 1.15),
                length_penalty=run_config.get("length_penalty", 0.),
                top_p=run_config.get("top_p", 0.9),
                temperature=run_config.get("temperature", 1.),
            )

        for j in range(len(pred_captions)):
            preds_gts_.append({"pred_caption": pred_captions[j], "gt_caption": txt_output[j]})

        all_pred_caption.extend(pred_captions)
        all_gt_caption.extend(txt_output)
        all_pred_ids.extend(pred_ids.detach().cpu())
        
        i += 1
        if i % update_interval == 0:
            pbar.update(update_interval)
            update_times += 1
    
    bar_rest_to_update = total_steps - update_interval * update_times
    if bar_rest_to_update > 0:
        pbar.update(bar_rest_to_update) 
    
    pbar.close()

    cider_scorer = Cider()

    all_pred_caption = {i: [all_pred_caption[i]] for i in range(len(all_pred_caption))}
    all_gt_caption = {i: [all_gt_caption[i]] for i in range(len(all_gt_caption))}
    
    cider_score, _ = cider_scorer.compute_score(all_gt_caption, all_pred_caption)

    return cider_score
    

def evaluate_qa(run_config, model, loader, device, modalities, mode='val'):
    total_steps= len(loader)
    update_interval = 10
    pbar = tqdm(total=total_steps)

    qa_eval = VQAEval()

    preds_gts_ = []

    model.eval()

    i = 0
    update_times = 0
    # total_num = 0
    acc_list = []
    for data in loader:
        if len(modalities) == 1:
            modal = modalities[0]
            index, modal_input, text_input, txt_output = data
            modal_input = modal_input.to(device)
            samples = {modal: modal_input, 'text_input': text_input, 'modalities': modalities}
        else:
            index, audio_input, video_input, text_input, txt_output = data
            audio_input = audio_input.to(device)
            video_input = video_input.to(device)
            samples = {"audio": audio_input, 'video': video_input, 'text_input': text_input, 'modalities': modalities}
    
        # img_input, text_input, txt_output = data
        # total_num += len(txt_output)
        # img_input = img_input.to(device)
        # samples = {"image": img_input, 'text_input': text_input, 'modalities': ['image']}

        with torch.cuda.amp.autocast(dtype=torch.float16):
            pred_answers, pred_ids = model.generate(
                samples,
                use_nucleus_sampling=False,
                num_beams=run_config.get("num_beams", 5),
                max_length=run_config.get("max_len", 30),
                min_length=run_config.get("min_len", 1),
                repetition_penalty=run_config.get("repetition_penalty", 1.15),
                length_penalty=run_config.get("length_penalty", 0.),
                top_p=run_config.get("top_p", 0.9),
                temperature=run_config.get("temperature", 1.),
            )
        
        # if modal == 'image':
        if modal == 'image':
            txt_output = [item.split('X') for item in txt_output]
            # if len(txt_output) == 1:
            #     txt_output = txt_output[0]

        if len(pred_answers) == 1:
            if len(txt_output) > 1:
                # [dfa, fafa, cad], [dfa, fafa, cad]
                # if isinstance(txt_output[0], list):
                #     gts = [qa_eval.processDigitArticle(item[0]) for item in txt_output]
                # [dfa, fafa, cad]
                # else:
                gts = [[qa_eval.processDigitArticle(item) for item in txt_output]]
            else:
                if isinstance(txt_output[0], list):
                    # [[as,d,d,s]]
                    gts = [[qa_eval.processDigitArticle(item) for item in txt_output[0]]]
                else:
                    gts = [qa_eval.processDigitArticle(txt_output[0])]
        else:
            if isinstance(txt_output[0], list):
                gts = [[qa_eval.processDigitArticle(item) for item in l] for l in txt_output]
            else:
                gts = [qa_eval.processDigitArticle(item) for item in txt_output]

        for j in range(len(pred_answers)):
            resAns = pred_answers[j]
            resAns = resAns.replace("\n", " ")
            resAns = resAns.replace("\t", " ")
            resAns = resAns.strip()
            resAns = qa_eval.processPunctuation(resAns)
            resAns = qa_eval.processDigitArticle(resAns)

            matching_num = 0
            gt = gts[j]
            if isinstance(gt, list):
                for item in gt:
                    if resAns == item:
                        matching_num += 1
                if len(gt) == 1:
                    sample_acc = matching_num
                else:
                    sample_acc = min(1., matching_num / 3.)
            else:
                if gt == resAns:
                    matching_num = 1
                sample_acc = matching_num
            acc_list.append(sample_acc)
        
        i += 1
        if i % update_interval == 0:
            pbar.update(update_interval)
            update_times += 1
    
    bar_rest_to_update = total_steps - update_interval * update_times
    if bar_rest_to_update > 0:
        pbar.update(bar_rest_to_update) 
    
    pbar.close()

    accuracy = sum(acc_list) / len(acc_list)

    return accuracy



def test(run_config, model, testset, step, task):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_modalities = task.split('_')[:-1]

    if task.endswith('cap'):
        if test_modalities[0] == 'video':
            batch_size = 8
            workers = 8
        else:
            batch_size = 8
            workers = 8
    else:
        if test_modalities[0] == 'video':
            batch_size = 8
            workers = 8
        else:
            batch_size = 8
            workers = 8

    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=workers,
                             pin_memory=True, drop_last=False, shuffle=False)
    if task.endswith('cap'):
        metric = 'CIDEr'
        test_res = evaluate_caption(run_config, model, test_loader, device, test_modalities, mode='test')
    # else:
    elif task.endswith('qa'):
        metric = 'Accuracy'
        test_res = evaluate_qa(run_config, model, test_loader, device, test_modalities, mode='test')
    else:
        pass
    test_res = round(test_res*100, 2)
    logging.info('Testing {}: {}'.format(metric, test_res))
    return test_res


def save_checkpoint(model, path, step, save_best=False):
    logging.info('Saving checkpoint........')
    named_parameters = model.module.named_parameters() if torch.cuda.device_count() > 1 else model.named_parameters()
    param_grad_dic = {k: v.requires_grad for (k, v) in named_parameters}
    
    state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
    encoder_ln_projection_dict = {}
    all_keys = list(state_dict.keys())
    for k in all_keys:
        if k in param_grad_dic:
            if param_grad_dic[k] is False:
                del state_dict[k]
            elif k.split('.')[0] in ['audio_encoder_projection', 'video_encoder_projection', 'image_encoder_projection', 
                                     'audio_ln', 'video_ln', 'image_ln', 'audio_projection', 'video_projection', 'image_projection', 
                                     'audio_temporal_projection', 'video_temporal_projection', 'image_temporal_projection',
                                     'audio_conv_pooling', 'video_conv_pooling', 'image_conv_pooling']:
                # setattr(self, f"{modality}_ln", modality_ln)
                encoder_ln_projection_dict[k] = state_dict[k]
                del state_dict[k]
    if len(encoder_ln_projection_dict) > 0:
        if step > 0:
            last_encoder_ln_projection_dict = torch.load(os.path.join(path, 'step_{}_encoder_ln_projection_best.pth'.format(step-1)), map_location="cpu")
            for k in last_encoder_ln_projection_dict:
                if k not in encoder_ln_projection_dict:
                    encoder_ln_projection_dict[k] = last_encoder_ln_projection_dict[k]
        encoder_ln_projection_name = 'step_{}_encoder_ln_projection_best.pth'.format(step) if save_best else 'step_{}_encoder_ln_projection_last.pth'.format(step)
        torch.save(encoder_ln_projection_dict, os.path.join(path, encoder_ln_projection_name))

    ckpt_name = 'step_{}_checkpoint_best.pth'.format(step) if save_best else 'step_{}_checkpoint_last.pth'.format(step)
    torch.save(state_dict, os.path.join(path, ckpt_name))


def load_checkpoint(model, path, step):
    logging.info('Loading checkpoint........')
    ckpt_path = os.path.join(path, 'step_{}_checkpoint_best.pth'.format(step))
    encoder_ln_proj_path = os.path.join(path, 'step_{}_encoder_ln_projection_best.pth'.format(step))
    # state_dict = torch.load(os.path.join(path, 'checkpoint_best.pth'), map_location="cpu")
    # msg = model.load_state_dict(state_dict, strict=True)
    msg = model.load_checkpoint(ckpt_path=ckpt_path, encoder_ln_proj_path=encoder_ln_proj_path, strict=False)
    # logging.info("Unloaded keys {}".format(msg.missing_keys))

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    parser.add_argument("--lambda_kl", type=float, default=0.1)
    # parser.add_argument("--lambda_f", type=float, default=0.5)
    parser.add_argument("--pseudo", action='store_true', default=False)
    parser.add_argument("--freeze_proj", action='store_true', default=False)
    # parser.add_argument("--freeze_qformer", action='store_true', default=False)
    parser.add_argument("--start_train_step", type=int, default=0)
    # parser.add_argument("--pseudo_ratio", type=str, default='all', choices=['one', 'half', 'all'])
    parser.add_argument("--pseudo_num", type=int, default=4)

    parser.add_argument("--text_instruct_constraint_type", type=str, default='KL', choices=['KL', 'pseudo'])
    parser.add_argument("--text_instruct_num", type=int, default=4)

    parser.add_argument("--alpha_fusion", type=float, default=0.0)
    parser.add_argument("--projection_fusion", action='store_true', default=False)

    parser.add_argument("--order", type=str, default='1', choices=['1', '2'])

    args = parser.parse_args()

    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




if __name__ == "__main__":

    setup_seed(1024)

    args = parse_args()
    cfg = Config(args)
    # cfg = OmegaConf.load(args.cfg_path)
    run_cfg = cfg.run_cfg

    run_cfg.lambda_kl = args.lambda_kl
    # run_cfg.lambda_f = args.lambda_f
    run_cfg.pseudo = args.pseudo
    run_cfg.freeze_proj = args.freeze_proj
    # run_cfg.freeze_qformer = args.freeze_qformer
    run_cfg.pseudo_num = args.pseudo_num
    run_cfg.text_instruct_constraint_type = args.text_instruct_constraint_type
    run_cfg.text_instruct_num = args.text_instruct_num
    run_cfg.alpha_fusion = args.alpha_fusion
    run_cfg.projection_fusion = args.projection_fusion

    logging.info('run_cfg: {}'.format(run_cfg))

    # run_cfg = cfg.run
    # model_cfg = cfg.model
    model_cfg = cfg.model_cfg

    lora = model_cfg.lora

    # config = OmegaConf.load(args.cfg_path)
    processors_cfg = OmegaConf.load(args.cfg_path).get('processors')
    txt_processor_cfg = processors_cfg.get('text_processor').get('train')
    txt_processor = registry.get_processor_class(txt_processor_cfg.get('name')).from_config(txt_processor_cfg)

    if args.order == '1':
        all_tasks_dict = {
            'audio_cap': 'AudioCaps',
            'image_cap': 'flickr30k', 
            'video_qa': 'MSVDQA', 
            'audio_qa': 'ClothoAQA', 
            'image_qa': 'OKVQA', 
            'video_cap': 'MSRVTT'}
    elif args.order == '2':
        all_tasks_dict = {
            'image_cap': 'flickr30k',
            'video_cap': 'MSRVTT',
            'video_qa': 'MSVDQA',
            'image_qa': 'OKVQA',
            'audio_cap': 'AudioCaps',
            'audio_qa': 'ClothoAQA'}
    else:
        raise ValueError

    all_tasks = list(all_tasks_dict.keys())

    logging.info('llm_model: {}'.format(model_cfg.llm_model))
    
    job_id = '{}'.format(model_cfg.llm_model.split('/')[-1])
    job_id += '-'.join(all_tasks)

    save_dir = os.path.join('./save', job_id)

    logging.info('Saving dir: {}'.format(save_dir))

    logging.info('run_cfg: {}'.format(run_cfg))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    max_epochs = [40] * len(all_tasks)

    batch_size_list = [8] * len(all_tasks)

    global_vars = globals()

    loader_classes = [global_vars['{}Loader'.format(all_tasks_dict[task])] for task in all_tasks]

    learned_task_type = []
    learned_modalities = []

    start_train_step = args.start_train_step

    natural_instruct_path = './data/Natural_Instructions/data.json'
    with open(natural_instruct_path, 'r', encoding='utf-8') as f:
        natural_instruct_data = json.load(f)

    for step in range(len(all_tasks)):

        curr_task = all_tasks[step]
        curr_modalities = curr_task.split('_')[:-1]

        if curr_task.endswith('cap'):
            curr_task_type = 'cap'
        elif curr_task.endswith('qa'):
            curr_task_type = 'qa'
        else:
            raise ValueError('Unrecognized task type: {}'.format(curr_task))

        if step <= start_train_step - 1:
            # if step < start_train_step:
            if curr_task_type not in learned_task_type:
                learned_task_type.append(curr_task_type)
            learned_modalities.extend(curr_modalities)
            learned_modalities = np.unique(learned_modalities).tolist()
            continue
        if step == start_train_step + 1:
            break

        model_cfg.modalities = curr_modalities
        model = get_model_only_proj_from_config(model_cfg, load_pretrained=False)

        if len(curr_modalities) != 1:
            model.joint_video_audio = True
        
        # task = all_tasks[step]
        logging.info("Start training at step {}: {}. Modality: {}".format(step, all_tasks_dict[curr_task], curr_modalities))

        trainset = loader_classes[step](mode='train', processors_cfg=processors_cfg)
        valset = loader_classes[step](mode='val', processors_cfg=processors_cfg)

        if step >= start_train_step:
            model = load_checkpoint(model, save_dir, step-1)
            
            old_model = get_model_only_proj_from_config(model_cfg, load_pretrained=False)
            # old_model.prompt = model.prompt
            old_model.joint_video_audio = model.joint_video_audio
            old_model = load_checkpoint(old_model, save_dir, step-1)
            old_model.requires_grad_ = False

            train(run_cfg, (model, old_model), trainset, valset, natural_instruct_data, step, curr_task, max_epochs[step], batch_size_list[step], curr_modalities, txt_processor, save_dir, learned_task_type, learned_modalities)
        
        if curr_task_type not in learned_task_type:
            learned_task_type.append(curr_task_type)
        learned_modalities.extend(curr_modalities)
        learned_modalities = np.unique(learned_modalities).tolist()

        del model
        try:
            del old_model
        except:
            pass

        logging.info("Start testing after training of step {}".format(step))
        test_res_list = []
        for i in range(step+1):
            curr_test_task = all_tasks[i]
            logging.info("Testing on task {}: {}".format(i, all_tasks_dict[curr_test_task]))
            testset = loader_classes[i](mode='test', processors_cfg=processors_cfg)
            # testset = loader_classes[i](mode='train', processors_cfg=processors_cfg)

            curr_modalities = curr_test_task.split('_')[:-1]
            # curr_modalities = ['audio', 'image']
            # cfg.model_cfg.modalities = curr_modalities
            model_cfg.modalities = curr_modalities
            model = get_model_only_proj_from_config(model_cfg, load_pretrained=False)
            model = load_checkpoint(model, path=save_dir, step=step)

            if len(curr_modalities) != 1:
                model.joint_video_audio = True
            
            if not (curr_task.endswith('cap') or curr_task.endswith('qa')):
                raise ValueError('Unrecognized task type: {}'.format(curr_task))

            test_res = test(run_cfg, model, testset, step, curr_test_task)
            test_res_list.append(test_res)

            del model
        
        test_res_dict = dict(zip(list(all_tasks_dict.values()), test_res_list))
        logging.info('Overall testing results at step {}: {}'.format(step, test_res_dict))

