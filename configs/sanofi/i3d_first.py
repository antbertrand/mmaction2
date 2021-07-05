_base_ = [
    '../_base_/models/i3d_r50.py', '../_base_/schedules/sgd_100e.py',
    '../_base_/default_runtime.py'
]

model = dict(
    cls_head=dict(
        num_classes=7,
        multi_class=True))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/home/deepo/sanofius/TRAINING/dataset/rawframes_rgb'
ann_file_train = '/home/deepo/sanofius/TRAINING/train_file_DEBUG.txt'
ann_file_val = '/home/deepo/sanofius/TRAINING/val_file_DEBUG.txt'
ann_file_test = '/home/deepo/sanofius/TRAINING/test_file_DEBUG.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=1),
    #clip_len (int): Frames of each sampled output clip.
    #frame_interval (int): Temporal interval of adjacent sampled frames.
    #num_clips (int): Number of clips to be sampled. Default: 1.
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        multi_class=True,
        num_classes=7,
        modality='RGB'),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root,
        pipeline=val_pipeline,
        multi_class=True,
        num_classes=7,
        modality='RGB'),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root,
        pipeline=test_pipeline,
        multi_class=True,
        num_classes=7,
        modality='RGB'))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = '/home/deepo/sanofius/TRAINING/workdir/'
log_level = 'DEBUG'

# # learning policy
# lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
#     policy='step',  # Policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
#     step=[40, 80])  # Steps to decay the learning rate
# total_epochs = 100  # Total epochs to train the model
# checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation
#     interval=5)  # Interval to save checkpoint
# evaluation = dict(  # Config of evaluation during training
#     interval=5,  # Interval to perform evaluation
#     metrics=['top_k_accuracy', 'mean_class_accuracy'],  # Metrics to be performed
#     metric_options=dict(top_k_accuracy=dict(topk=(1, 3))), # Set top-k accuracy to 1 and 3 during validation
#     save_best='top_k_accuracy')  # set `top_k_accuracy` as key indicator to save best checkpoint