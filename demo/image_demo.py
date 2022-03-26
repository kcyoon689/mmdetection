# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

import os
import numpy as np
import cv2
from matplotlib import pyplot
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.18, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    parser.add_argument('--directory', help='Image directory')
    args = parser.parse_args()
    return args


def main(args):
    image_list = []
    if args.directory is None:
        image_list.append(args.img)
    else:
        for filename in os.listdir(args.directory):
            image_list.append(args.directory + "/" + filename)

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    for image in tqdm(image_list):
        # test a single image
        #############################
        ###########################
        result = inference_detector(model, image)
        # show the results
        show_result_pyplot(
            model,
            image,
            result,
            palette=args.palette,
            score_thr=args.score_thr)
        ##############################
        ###############################
        # result, x_np = inference_detector(model, image) # x_np: 200, 304, 256
        # # print(image)

        # np.save('/home/crvl-yoon/mmdetection/demo/feature_map.npy', x_np)
        # # featureMaps = np.load('/home/crvl-yoon/mmdetection/demo/feature_map.npy')
        # # featureMaps_np = np.asarray(featureMaps)

        # featureMaps_np = x_np

        # os.makedirs('/home/crvl-yoon/mmdetection/demo/feature_map', exist_ok=True)
        # head, tail = os.path.split(image)

        # images = 4
        # square = 8
        # for kdx in range(images):
        #     for idx in range(square):
        #         for jdx in range(square):
        #             pyplot.subplot(square, square, idx * square + jdx + 1)
        #             pyplot.imshow(featureMaps_np[:,:,idx * square + jdx], cmap='gray')
        #     # pyplot.savefig('/home/crvl-yoon/mmdetection/demo/feature_map_' + str(kdx) + '.png', dpi=1000)
        #     pyplot.savefig('/home/crvl-yoon/mmdetection/demo/feature_map/' + tail.split('.')[0] + '_' + str(kdx) + '.png', dpi=1000)
        #     # pyplot.show()

        # img_input = cv2.imread(image)
        # cv2.imwrite('/home/crvl-yoon/mmdetection/demo/feature_map/' + tail, img_input)
        # # show_result_pyplot(
        # #     model,
        # #     image,
        # #     result,
        # #     palette=args.palette,
        # #     score_thr=args.score_thr)
        ######################################
        ###################################
        


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
