from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf

# import lavis
# from lavis.datasets.builders import load_dataset, dataset_zoo
from lavis.common.registry import registry

import os
import h5py
import numpy as np
import random

from PIL import Image
import torchaudio

class flickr30kLoader(Dataset):
    def __init__(self, processors_cfg, mode='train'):
        self.mode = mode

        flickr30k_data_root = '/mnt/data0/wpian/dataset/flickr30k'

        self.img_root = os.path.join(flickr30k_data_root, 'flickr30k-images')
        # self.all_imgs = h5py.File(os.path.join(flickr30k_data_root, 'imgs.h5'), 'r')

        # img_id_caption_dict_path = os.path.join(flickr30k_data_root, 'all_img_id_caption_dict.npy')
        img_id_caption_dict_path = './dataset/all_img_id_caption_dict.npy'
        self.img_id_caption_dict = np.load(img_id_caption_dict_path, allow_pickle=True).item()

        # processors_cfg = config.get('processors')

        if self.mode == 'train':
            # self.flickr30k = self.flickr30k["train"]
            self.img_id_caption_dict = self.img_id_caption_dict['train']
            img_processor_cfg = processors_cfg.get('img_processor').get('train')
            txt_processor_cfg = processors_cfg.get('text_processor').get('train')
        elif self.mode == 'val':
            # self.flickr30k = self.flickr30k["test"]
            self.img_id_caption_dict = self.img_id_caption_dict['val']
            img_processor_cfg = processors_cfg.get('img_processor').get('eval')
            txt_processor_cfg = processors_cfg.get('text_processor').get('eval')
        elif self.mode == 'test':
            # self.flickr30k = self.flickr30k["test"]
            self.img_id_caption_dict = self.img_id_caption_dict['test']
            img_processor_cfg = processors_cfg.get('img_processor').get('eval')
            txt_processor_cfg = processors_cfg.get('text_processor').get('eval')
        else:
            raise ValueError('mode must be \'train\', \'val\' or \'test\'')
        
        self.img_id_list = list(self.img_id_caption_dict.keys())

        self.img_processor = registry.get_processor_class(img_processor_cfg.get('name')).from_config(img_processor_cfg)
        self.txt_processor = registry.get_processor_class(txt_processor_cfg.get('name')).from_config(txt_processor_cfg)
        
        if self.mode == 'train':
            self.txt_processor.task = 'caption'
        else:
            self.txt_processor.task = 'eval'
        self.txt_processor.prompt = 'describe the image.'
        self.txt_processor.modality = 'image'

    
    def __getitem__(self, index):
        img_id = self.img_id_list[index]

        img_path = os.path.join(self.img_root, img_id + '.jpg')
        img_input = Image.open(img_path).convert('RGB')

        img_input = self.img_processor(img_input)
        # txt_input = 'describe the image.'
        txt_input = self.txt_processor("")

        txt_output = self.img_id_caption_dict[img_id]
        txt_output = self.txt_processor(txt_output)

        return str(img_id), img_input, txt_input, txt_output

    def __len__(self):
        # return len(self.flickr30k)
        return len(self.img_id_list)
    

class flickr30kLoader4pseudoQA(Dataset):
    def __init__(self, processors_cfg, mode='train'):
        self.mode = mode

        # self.flickr30k = load_dataset("flickr30k")

        flickr30k_data_root = '/project/home/p200686/dataset/flickr30k'

        self.img_root = os.path.join(flickr30k_data_root, 'flickr30k-images')
        # self.all_imgs = h5py.File(os.path.join(flickr30k_data_root, 'imgs.h5'), 'r')

        img_id_caption_dict_path = './dataset/all_img_id_caption_dict.npy'
        self.img_id_caption_dict = np.load(img_id_caption_dict_path, allow_pickle=True).item()

        # processors_cfg = config.get('processors')

        if self.mode == 'train':
            # self.flickr30k = self.flickr30k["train"]
            self.img_id_caption_dict = self.img_id_caption_dict['train']
            img_processor_cfg = processors_cfg.get('img_processor').get('train')
            txt_processor_cfg = processors_cfg.get('text_processor').get('train')
        elif self.mode == 'val':
            # self.flickr30k = self.flickr30k["test"]
            self.img_id_caption_dict = self.img_id_caption_dict['val']
            img_processor_cfg = processors_cfg.get('img_processor').get('eval')
            txt_processor_cfg = processors_cfg.get('text_processor').get('eval')
        elif self.mode == 'test':
            # self.flickr30k = self.flickr30k["test"]
            self.img_id_caption_dict = self.img_id_caption_dict['test']
            img_processor_cfg = processors_cfg.get('img_processor').get('eval')
            txt_processor_cfg = processors_cfg.get('text_processor').get('eval')
        else:
            raise ValueError('mode must be \'train\', \'val\' or \'test\'')
        
        self.img_id_list = list(self.img_id_caption_dict.keys())

        self.img_processor = registry.get_processor_class(img_processor_cfg.get('name')).from_config(img_processor_cfg)
        self.txt_processor = registry.get_processor_class(txt_processor_cfg.get('name')).from_config(txt_processor_cfg)
        
        if self.mode == 'train':
            self.txt_processor.task = 'caption'
        else:
            self.txt_processor.task = 'eval'
        self.txt_processor.prompt = 'describe the image.'
        self.txt_processor.modality = 'image'

    
    def __getitem__(self, index):
        img_id = self.img_id_list[index]

        # img_path = os.path.join(self.img_root, img_id + '.jpg')
        # img_input = Image.open(img_path).convert('RGB')

        # img_input = self.img_processor(img_input)
        # txt_input = 'describe the image.'
        txt_input = self.txt_processor("")

        txt_output = self.img_id_caption_dict[img_id]
        txt_output = self.txt_processor(txt_output)

        # return str(img_id), img_input, txt_input, txt_output
        return str(img_id), txt_input, txt_output

    def __len__(self):
        # return len(self.flickr30k)
        return len(self.img_id_list)




class OKVQALoader(Dataset):
    def __init__(self, processors_cfg, mode='train'):
        self.mode = mode

        if mode not in ['train', 'val', 'test']:
            raise ValueError('mode must be \'train\', \'val\' or \'test\'')

        OKVQA_data_root = '/mnt/data0/wpian/dataset/coco/OK-VQA'

        anno_path = './dataset/question_id_img_id_quesion_answer_dict.npy'
        self.question_id_dict = np.load(anno_path, allow_pickle=True).item()

        # processors_cfg = config.get('processors')

        self.question_id_dict = self.question_id_dict[mode]

        if self.mode == 'train':
            self.img_root = os.path.join(OKVQA_data_root, 'train2014')
            self.img_prefix = 'COCO_train2014_'
            img_processor_cfg = processors_cfg.get('img_processor').get('train')
            txt_processor_cfg = processors_cfg.get('text_processor').get('train')
        elif self.mode == 'val':
            self.img_root = os.path.join(OKVQA_data_root, 'train2014')
            self.img_prefix = 'COCO_train2014_'
            img_processor_cfg = processors_cfg.get('img_processor').get('eval')
            txt_processor_cfg = processors_cfg.get('text_processor').get('eval')
        elif self.mode == 'test':
            self.img_root = os.path.join(OKVQA_data_root, 'val2014')
            self.img_prefix = 'COCO_val2014_'
            img_processor_cfg = processors_cfg.get('img_processor').get('eval')
            txt_processor_cfg = processors_cfg.get('text_processor').get('eval')
        
        self.question_ids_list = list(self.question_id_dict.keys())

        self.img_processor = registry.get_processor_class(img_processor_cfg.get('name')).from_config(img_processor_cfg)
        self.txt_processor = registry.get_processor_class(txt_processor_cfg.get('name')).from_config(txt_processor_cfg)

        if self.mode == 'train':
            self.txt_processor.task = 'qa'
        else:
            self.txt_processor.task = 'eval'
        self.txt_processor.modality = 'image'
        self.txt_processor.prompt = ''

    
    def __getitem__(self, index):
        question_id = self.question_ids_list[index]

        img_id = self.question_id_dict[question_id]['img_id']
        question = self.question_id_dict[question_id]['question']
        answers = self.question_id_dict[question_id]['answers']

        img_name = self.img_prefix + str(img_id).zfill(12) + '.jpg'

        img_path = os.path.join(self.img_root, img_name)

        img_input = Image.open(img_path).convert('RGB')
        img_input = self.img_processor(img_input)

        txt_input = question
        txt_input = self.txt_processor(txt_input)

        # print('answers', answers)
        if self.mode == 'train':
            txt_output = random.choice(answers)
            txt_output = self.txt_processor.pre_caption(txt_output)
        else:
            # txt_output = [self.txt_processor(answer) for answer in answers]
            txt_output = [self.txt_processor(answer) for answer in answers]
            txt_output = 'X'.join(txt_output)

        return str(question_id), img_input, txt_input, txt_output

    def __len__(self):
        return len(self.question_ids_list)
    
class MSRVTTLoader(Dataset):
    def __init__(self, processors_cfg, mode='train'):
        self.mode = mode

        msrvtt_data_root = '/mnt/data0/wpian/dataset/MSRVTT'

        self.video_root = os.path.join(msrvtt_data_root, 'videos', 'all')
        # self.audio_root = os.path.join(msrvtt_data_root, 'audios')

        vid_caption_dict_path = './dataset/all_vid_caption_dict.npy'
        # vid_caption_dict_path = os.path.join(msrvtt_data_root, 'all_avid_caption_dict.npy')
        self.vid_caption_dict = np.load(vid_caption_dict_path, allow_pickle=True).item()

        # avid_caption_dict_path = os.path.join(msrvtt_data_root, 'all_avid_caption_dict.npy')
        # self.avid_caption_dict = np.load(avid_caption_dict_path, allow_pickle=True).item()

        if self.mode == 'train':
            # self.avid_caption_dict = self.avid_caption_dict['train']
            self.vid_caption_dict = self.vid_caption_dict['train']
            video_processor_cfg = processors_cfg.get('video_processor').get('train')
            # audio_processor_cfg = processors_cfg.get('audio_processor').get('train')
            txt_processor_cfg = processors_cfg.get('text_processor').get('train')
        elif self.mode == 'val':
            # self.avid_caption_dict = self.avid_caption_dict['val']
            self.vid_caption_dict = self.vid_caption_dict['val']
            video_processor_cfg = processors_cfg.get('video_processor').get('eval')
            # audio_processor_cfg = processors_cfg.get('audio_processor').get('eval')
            txt_processor_cfg = processors_cfg.get('text_processor').get('eval')
        elif self.mode == 'test':
            # self.avid_caption_dict = self.avid_caption_dict['test']
            self.vid_caption_dict = self.vid_caption_dict['test']
            video_processor_cfg = processors_cfg.get('video_processor').get('eval')
            # audio_processor_cfg = processors_cfg.get('audio_processor').get('eval')
            txt_processor_cfg = processors_cfg.get('text_processor').get('eval')
        else:
            raise ValueError('mode must be \'train\', \'val\' or \'test\'')
        
        # self.avid_list = list(self.avid_caption_dict.keys())
        self.vid_list = list(self.vid_caption_dict.keys())

        self.video_processor = registry.get_processor_class(video_processor_cfg.get('name')).from_config(video_processor_cfg)
        # self.audio_processor = registry.get_processor_class(audio_processor_cfg.get('name')).from_config(audio_processor_cfg)
        self.txt_processor = registry.get_processor_class(txt_processor_cfg.get('name')).from_config(txt_processor_cfg)
        
        if self.mode == 'train':
            self.txt_processor.task = 'caption'
        else:
            self.txt_processor.task = 'eval'
        self.txt_processor.prompt = 'describe the video.'
        self.txt_processor.modality = 'video'
        # self.txt_processor.prompt = 'describe the audio-video.'
        # self.txt_processor.modality = 'audio-video'

    
    def __getitem__(self, index):
        # vid = self.avid_list[index]
        vid = self.vid_list[index]

        video_path = os.path.join(self.video_root, vid + '.mp4')
        # audio_path = os.path.join(self.audio_root, vid + '.wav')

        video = self.video_processor(video_path)
        # audio = self.audio_processor(audio_path)

        txt_input = self.txt_processor("")

        # txt_output = self.avid_caption_dict[vid]
        txt_output = self.vid_caption_dict[vid]
        txt_output = self.txt_processor(txt_output)

        # return str(vid), audio, video, txt_input, txt_output
        return str(vid), video, txt_input, txt_output

    def __len__(self):
        # return len(self.avid_list)
        return len(self.vid_list)



class AudioCapsLoader(Dataset):
    def __init__(self, processors_cfg, mode='train'):
        self.mode = mode

        audiocaps_data_root = '/mnt/data0/wpian/dataset/AudioCaps'

        self.audio_root = os.path.join(audiocaps_data_root, 'all_audios')
        # self.audio_root = os.path.join(audiocaps_data_root, 'all_audios_new')
        # print('audio_root: {}'.format(self.audio_root))

        aid_caption_dict_path = './dataset/all_audio_id_cap_dict.npy'
        self.aid_caption_dict = np.load(aid_caption_dict_path, allow_pickle=True).item()

        if self.mode == 'train':
            self.aid_caption_dict = self.aid_caption_dict['train']
            audio_processor_cfg = processors_cfg.get('audio_processor').get('train')
            txt_processor_cfg = processors_cfg.get('text_processor').get('train')
        elif self.mode == 'val':
            self.aid_caption_dict = self.aid_caption_dict['val']
            audio_processor_cfg = processors_cfg.get('audio_processor').get('eval')
            txt_processor_cfg = processors_cfg.get('text_processor').get('eval')
        elif self.mode == 'test':
            self.aid_caption_dict = self.aid_caption_dict['test']
            audio_processor_cfg = processors_cfg.get('audio_processor').get('eval')
            txt_processor_cfg = processors_cfg.get('text_processor').get('eval')
        else:
            raise ValueError('mode must be \'train\', \'val\' or \'test\'')
        
        self.aid_list = list(self.aid_caption_dict.keys())

        self.audio_processor = registry.get_processor_class(audio_processor_cfg.get('name')).from_config(audio_processor_cfg)
        self.txt_processor = registry.get_processor_class(txt_processor_cfg.get('name')).from_config(txt_processor_cfg)
        
        if self.mode == 'train':
            self.txt_processor.task = 'caption'
        else:
            self.txt_processor.task = 'eval'
        self.txt_processor.prompt = 'describe the audio.'
        self.txt_processor.modality = 'audio'

    
    def __getitem__(self, index):
        aid = self.aid_list[index]

        audio_path = os.path.join(self.audio_root, aid + '.wav')

        try:
            # waveform, sr = torchaudio.load(audio_path)
            # print(waveform.shape)
            audio = self.audio_processor(audio_path)
        except:
            print('loading audio failed: {}'.format(audio_path))
            exit(0)

        txt_input = self.txt_processor("")

        txt_output = self.aid_caption_dict[aid]
        txt_output = self.txt_processor(txt_output)

        return str(aid), audio, txt_input, txt_output

    def __len__(self):
        return len(self.aid_list)


class ClothoAQALoader(Dataset):
    def __init__(self, processors_cfg, mode='train'):
        self.mode = mode

        clotho_data_root = '/mnt/data0/wpian/dataset/Clotho-AQA'

        self.audio_root = os.path.join(clotho_data_root, 'audio_files')

        anno_path = './dataset/question_id_audio_id_quesion_answer_dict.npy'
        self.question_id_dict = np.load(anno_path, allow_pickle=True).item()

        self.question_id_dict = self.question_id_dict[mode]

        if self.mode == 'train':
            audio_processor_cfg = processors_cfg.get('audio_processor').get('train')
            txt_processor_cfg = processors_cfg.get('text_processor').get('train')
        elif self.mode == 'val':
            audio_processor_cfg = processors_cfg.get('audio_processor').get('eval')
            txt_processor_cfg = processors_cfg.get('text_processor').get('eval')
        elif self.mode == 'test':
            audio_processor_cfg = processors_cfg.get('audio_processor').get('eval')
            txt_processor_cfg = processors_cfg.get('text_processor').get('eval')
        
        self.question_ids_list = list(self.question_id_dict.keys())

        self.audio_processor = registry.get_processor_class(audio_processor_cfg.get('name')).from_config(audio_processor_cfg)
        self.txt_processor = registry.get_processor_class(txt_processor_cfg.get('name')).from_config(txt_processor_cfg)

        if self.mode == 'train':
            self.txt_processor.task = 'qa'
        else:
            self.txt_processor.task = 'eval'
        self.txt_processor.modality = 'audio'
        self.txt_processor.prompt = ''

    
    def __getitem__(self, index):
        question_id = self.question_ids_list[index]

        audio_id = self.question_id_dict[question_id]['audio_id']
        question = self.question_id_dict[question_id]['question']
        answer = self.question_id_dict[question_id]['answer']

        audio_path = os.path.join(self.audio_root, audio_id)

        try:
            audio = self.audio_processor(audio_path)
        except:
            print('loading audio failed: {}'.format(audio_path))
            exit(0)

        txt_input = self.txt_processor(question)

        if self.mode == 'train':
            # txt_output = random.choice(answers)
            txt_output = self.txt_processor.pre_caption(answer)
        else:
            # txt_output = [self.txt_processor(answer) for answer in answers]
            # txt_output = [self.txt_processor(answer) for answer in answers]
            # txt_output = [self.txt_processor(answer)]
            txt_output = self.txt_processor(answer)

        return str(question_id), audio, txt_input, txt_output

    def __len__(self):
        return len(self.question_ids_list)


class MSVDQALoader(Dataset):
    def __init__(self, processors_cfg, mode='train'):
        self.mode = mode

        MSVD_data_root = '/mnt/data0/wpian/dataset/MSVD/MSVD-QA'

        self.video_root = os.path.join(MSVD_data_root, 'video')

        anno_path = './dataset/question_id_video_id_quesion_answer_dict.npy'
        self.question_id_dict = np.load(anno_path, allow_pickle=True).item()

        self.question_id_dict = self.question_id_dict[mode]

        if self.mode == 'train':
            video_processor_cfg = processors_cfg.get('video_processor').get('train')
            txt_processor_cfg = processors_cfg.get('text_processor').get('train')
        elif self.mode == 'val':
            video_processor_cfg = processors_cfg.get('video_processor').get('eval')
            txt_processor_cfg = processors_cfg.get('text_processor').get('eval')
        elif self.mode == 'test':
            video_processor_cfg = processors_cfg.get('video_processor').get('eval')
            txt_processor_cfg = processors_cfg.get('text_processor').get('eval')
        
        self.question_ids_list = list(self.question_id_dict.keys())

        self.video_processor = registry.get_processor_class(video_processor_cfg.get('name')).from_config(video_processor_cfg)
        self.txt_processor = registry.get_processor_class(txt_processor_cfg.get('name')).from_config(txt_processor_cfg)

        if self.mode == 'train':
            self.txt_processor.task = 'qa'
        else:
            self.txt_processor.task = 'eval'
        self.txt_processor.modality = 'video'
        self.txt_processor.prompt = ''

    
    def __getitem__(self, index):
        question_id = self.question_ids_list[index]

        video_id = self.question_id_dict[question_id]['video_id']
        question = self.question_id_dict[question_id]['question']
        answer = self.question_id_dict[question_id]['answer']

        video_path = os.path.join(self.video_root, video_id)

        video = self.video_processor(video_path)

        txt_input = self.txt_processor(question)

        if self.mode == 'train':
            txt_output = self.txt_processor.pre_caption(answer)
            # txt_output = [self.txt_processor(answer)]
        else:
            txt_output = self.txt_processor(answer)
            # txt_output = [self.txt_processor(answer).lower(), self.txt_processor(answer).lower(), self.txt_processor(answer).lower()]
            # txt_output = 'X'.join(txt_output)

        return str(question_id), video, txt_input, txt_output

    def __len__(self):
        return len(self.question_ids_list)

