checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook', interval=1000,
            init_kwargs=dict(
                project='CRVL_DetectoRS',
                entity='chaeyoon_kim',
                name = 'exp00'
            )
        )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = None
load_from = 'checkpoints/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth'
# load_from = 'checkpoints/16class_detectoRS_epoch20.pth'
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
