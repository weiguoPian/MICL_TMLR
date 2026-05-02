
from .blip2_vicuna_xinstruct_only_proj_with_temporal import Blip2VicunaXInstructOnlyProjwithTemporal


import logging
from omegaconf import OmegaConf


def get_model_only_proj_from_config(cfg, load_pretrained=True):
    image_model = cfg.get("image_model","eva_clip_g")
    pc_model = cfg.get("pc_model","ulip2_pointbert")
    video_model = cfg.get("video_model","eva_clip_g")
    audio_model = cfg.get("audio_model","beats")

    pretrained_image_qformer = cfg.get("pretrained_image_qformer",None)
    pretrained_pc_qformer = cfg.get("pretrained_pc_qformer",None)
    pretrained_video_qformer = cfg.get("pretrained_video_qformer",None)
    pretrained_audio_qformer = cfg.get("pretrained_audio_qformer",None)

    load_attention_image_qformer = cfg.get("load_attention_image_qformer",False)
    load_attention_pc_qformer = cfg.get("load_attention_pc_qformer",False)
    load_attention_video_qformer = cfg.get("load_attention_video_qformer",False)
    load_attention_audio_qformer = cfg.get("load_attention_audio_qformer",False)

    load_qformer_type_image=cfg.get('load_qformer_type_image', "")
    load_qformer_type_pc=cfg.get('load_qformer_type_pc', "")
    load_qformer_type_video=cfg.get('load_qformer_type_video', "")
    load_qformer_type_audio=cfg.get('load_qformer_type_audio',"")

    load_projection_image=cfg.get('load_projection_image', True)
    load_projection_pc=cfg.get('load_projection_pc', True)
    load_projection_video=cfg.get('load_projection_video', True)
    load_projection_audio=cfg.get('load_projection_audio', True)

    load_projection_type_image=cfg.get('load_projection_type_image', "")
    load_projection_type_pc=cfg.get('load_projection_type_pc', "")
    load_projection_type_video=cfg.get('load_projection_type_video', "")
    load_projection_type_audio=cfg.get('load_projection_type_audio', "")

    load_ln_type_image=cfg.get('load_ln_type_image', "")
    load_ln_type_pc=cfg.get('load_ln_type_pc', "")
    load_ln_type_video=cfg.get('load_ln_type_video', "")
    load_ln_type_audio=cfg.get('load_ln_type_audio', "")

    image_encoder_kwargs = cfg.get("image_encoder_kwargs", {"image_size": 224, "drop_path_rate": 0, "use_grad_checkpoint": False})
    pc_encoder_kwargs = cfg.get("pc_encoder_kwargs",{})
    video_encoder_kwargs = cfg.get("video_encoder_kwargs",{})
    audio_encoder_kwargs = cfg.get("audio_encoder_kwargs",{})

    image_precision = cfg.get("image_precision","fp16")
    pc_precision = cfg.get("pc_precision","fp16")
    video_precision = cfg.get("video_precision","fp16")
    audio_precision = cfg.get("audio_precision","fp16")

    freeze_image = cfg.get("freeze_image",True)
    freeze_pc = cfg.get("freeze_pc",True)
    freeze_video = cfg.get("freeze_video",True)
    freeze_audio = cfg.get("freeze_audio",True)
    num_query_token = cfg.get("num_query_token")
    
    llm_model = cfg.get("llm_model")
    freeze_pc = cfg.get("freeze_pc", True)
    freeze_video = cfg.get("freeze_video", True)
    freeze_audio = cfg.get("freeze_audio", True)

    prompt = cfg.get("prompt", "")
    max_txt_len = cfg.get("max_txt_len", 128)
    max_output_txt_len = cfg.get("max_output_txt_len", 256)

    apply_lemmatizer = cfg.get("apply_lemmatizer", False)

    qformer_text_input = cfg.get("qformer_text_input", True)
    modalities = cfg.get("modalities", ["image"])
    use_cues = cfg.get("use_cues", True)
    shared_qformer = cfg.get("shared_qformer",False)
    pretrained_shared_qformer = cfg.get("pretrained_shared_qformer", None)
    load_attention_shared_qformer = cfg.get("load_attention_shared_qformer", None)
    load_qformer_type_shared= cfg.get('load_qformer_type_shared',"")
    load_projection_shared= cfg.get('load_projection_shared',False)
    load_projection_type_shared= cfg.get('load_projection_type_shared',"")
    shared_qformer_num_features=cfg.get("shared_qformer_num_features", 512)
    encoder_projection_type_image=cfg.get("encoder_projection_type_image","")
    encoder_projection_type_video=cfg.get("encoder_projection_type_video","")
    encoder_projection_type_audio=cfg.get("encoder_projection_type_audio","")
    encoder_projection_type_pc=cfg.get("encoder_projection_type_pc","")

    llm_text_input = cfg.get("llm_text_input", True)
    lora = cfg.get("lora", False)
    prefix = cfg.get("prefix", "")
    postfix = cfg.get("postfix", "")

    cached_audio= cfg.get("cached_audio", False)
    cached_image= cfg.get("cached_image", False)
    cached_video= cfg.get("cached_video", False)
    cached_pc= cfg.get("cached_pc", False)

    num_features_audio=cfg.get('num_features_audio', 768)
    num_features_image=cfg.get('num_features_image', 1408)
    num_features_video=cfg.get('num_features_video', 14080)
    num_features_pc=cfg.get('num_features_depth', 512)

    joint_video_audio=cfg.get('joint_video_audio', False)
    use_caption=cfg.get('use_caption', False)
    use_describe=cfg.get('use_describe', False)
    predict_with_gen = cfg.get('predict_with_gen', False)
    format_candidates_prompt = cfg.get('format_candidates_prompt', "{}")
    special_qformer_input_prompt = cfg.get('special_qformer_input_prompt', False)
    enumerate_inputs = cfg.get('enumerate_inputs', False)
    add_space = cfg.get('add_space', True)
    projection_only = cfg.get('projection_only', False)

    lora_model = cfg.get('lora_model', '')

    projection_only_audio= cfg.get('projection_only_audio', False)
    projection_only_pc=  cfg.get('projection_only_pc', False)
    projection_only_video=  cfg.get('projection_only_video', False)
    projection_only_image=  cfg.get('projection_only_image', False)

    projection_path_audio=cfg.get('projection_path_audio', False)
    projection_path_pc=cfg.get('projection_path_pc', False)
    projection_path_video=cfg.get('projection_path_video', False)
    projection_path_image=cfg.get('projection_path_image', False)
    remove_start=cfg.get('remove_start', False)
    proj_dim=cfg.get('proj_dim', 1)
    clean_tokenization=cfg.get('clean_tokenization', False)


    model = Blip2VicunaXInstructOnlyProjwithTemporal(
        image_model=image_model,
        pc_model=pc_model,
        video_model=video_model,
        audio_model=audio_model,

        pretrained_image_qformer=pretrained_image_qformer,
        pretrained_pc_qformer=pretrained_pc_qformer,
        pretrained_video_qformer=pretrained_video_qformer,
        pretrained_audio_qformer=pretrained_audio_qformer,

        load_attention_image_qformer=load_attention_image_qformer,
        load_attention_pc_qformer=load_attention_pc_qformer,
        load_attention_video_qformer=load_attention_video_qformer,
        load_attention_audio_qformer=load_attention_audio_qformer,

        load_qformer_type_image=load_qformer_type_image,
        load_qformer_type_pc=load_qformer_type_pc,
        load_qformer_type_video=load_qformer_type_video,
        load_qformer_type_audio=load_qformer_type_audio,

        load_projection_image=load_projection_image,
        load_projection_pc=load_projection_pc,
        load_projection_video=load_projection_video,
        load_projection_audio=load_projection_audio,

        load_projection_type_image=load_projection_type_image,
        load_projection_type_pc=load_projection_type_pc,
        load_projection_type_video=load_projection_type_video,
        load_projection_type_audio=load_projection_type_audio,

        load_ln_type_image=load_ln_type_image,
        load_ln_type_pc=load_ln_type_pc,
        load_ln_type_video=load_ln_type_video,
        load_ln_type_audio=load_ln_type_audio,

        image_encoder_kwargs = image_encoder_kwargs,
        pc_encoder_kwargs = pc_encoder_kwargs,
        video_encoder_kwargs = video_encoder_kwargs,
        audio_encoder_kwargs = audio_encoder_kwargs,

        image_precision=image_precision,
        pc_precision=pc_precision,
        video_precision=video_precision,
        audio_precision=audio_precision,

        freeze_image=freeze_image,
        freeze_pc=freeze_pc,
        freeze_video=freeze_video,
        freeze_audio=freeze_audio,

        num_query_token=num_query_token,
        llm_model=llm_model,
        lora_model=lora_model,
        lora = lora,
        prompt=prompt,
        max_txt_len=max_txt_len,
        max_output_txt_len=max_output_txt_len,
        apply_lemmatizer=apply_lemmatizer,
        qformer_text_input=qformer_text_input,
        modalities=modalities,
        use_cues=use_cues,
        llm_text_input=llm_text_input,
        shared_qformer=shared_qformer,
        pretrained_shared_qformer = pretrained_shared_qformer,
        load_attention_shared_qformer = load_attention_shared_qformer,
        shared_qformer_num_features=shared_qformer_num_features,
        load_qformer_type_shared= load_qformer_type_shared,
        load_projection_shared= load_projection_shared,

        encoder_projection_type_image=encoder_projection_type_image, 
        encoder_projection_type_video=encoder_projection_type_video, 
        encoder_projection_type_audio=encoder_projection_type_audio, 
        encoder_projection_type_pc=encoder_projection_type_pc,

        projection_path_audio=projection_path_audio,
        projection_path_pc=projection_path_pc,
        projection_path_video=projection_path_video,
        projection_path_image=projection_path_image,

        load_projection_type_shared= load_projection_type_shared,

        prefix=prefix,
        postfix=postfix,

        cached_audio=cached_audio,
        cached_image=cached_image,
        cached_video=cached_video,
        cached_pc=cached_pc,

        num_features_audio=num_features_audio,
        num_features_image=num_features_image,
        num_features_video=num_features_video,
        num_features_pc=num_features_pc,

        joint_video_audio=joint_video_audio,
        use_caption=use_caption,
        use_describe=use_describe,
        predict_with_gen=predict_with_gen,
        format_candidates_prompt=format_candidates_prompt,
        special_qformer_input_prompt=special_qformer_input_prompt,
        enumerate_inputs=enumerate_inputs,
        add_space=add_space,
        projection_only=projection_only,

        projection_only_audio= projection_only_audio,
        projection_only_pc=  projection_only_pc,
        projection_only_video= projection_only_video,
        projection_only_image=  projection_only_image,
        remove_start= remove_start,
        proj_dim=proj_dim,
        clean_tokenization=clean_tokenization
    )

    if load_pretrained:
        logging.info("Load from pretrained...")
        model.load_checkpoint_from_config(cfg)
    return model




