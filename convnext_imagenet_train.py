# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#:
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" ConvNext training script. """

import argparse

import mindspore.nn as nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.communication import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from mindvision.classification.dataset import ImageNet
from mindvision.engine.lr_schedule.lr_schedule import warmup_cosine_annealing_lr_v1
from mindvision.engine.callback import LossMonitor

set_seed(1)


def convnext_train(args_opt):
    """ ConvNext train."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data Pipeline.
    if args_opt.run_distribute:
        init("nccl")
        rank_id = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        dataset = ImageNet(args_opt.data_url,
                           split="train",
                           num_parallel_workers=args_opt.num_parallel_workers,
                           shuffle=True,
                           resize=args_opt.resize,
                           num_shards=device_num,
                           shard_id=rank_id,
                           batch_size=args_opt.batch_size,
                           repeat_num=args_opt.repeat_num)
        ckpt_save_dir = args_opt.ckpt_save_dir + "_ckpt_" + str(rank_id) + "/"
    else:
        dataset = ImageNet(args_opt.data_url,
                           split="train",
                           num_parallel_workers=args_opt.num_parallel_workers,
                           shuffle=True,
                           resize=args_opt.resize,
                           batch_size=args_opt.batch_size,
                           repeat_num=args_opt.repeat_num)
        ckpt_save_dir = args_opt.ckpt_save_dir

    dataset_train = dataset.run()
    step_size = dataset_train.get_dataset_size()

    # Create_model.
    if args_opt.model == 'convnext_tiny':
        from mindvision.classification.models import convnext_tiny
        network = convnext_tiny(pretrained=args_opt.pretrained, num_classes=args_opt.num_classes)
    if args_opt.model == 'convnext_small':
        from mindvision.classification.models import convnext_small
        network = convnext_small(pretrained=args_opt.pretrained, num_classes=args_opt.num_classes)
    if args_opt.model == 'convnext_base':
        from mindvision.classification.models import convnext_base
        network = convnext_base(pretrained=args_opt.pretrained, num_classes=args_opt.num_classes)
    if args_opt.model == 'convnext_large':
        from mindvision.classification.models import convnext_large
        network = convnext_large(pretrained=args_opt.pretrained, num_classes=args_opt.num_classes)
    if args_opt.model == 'convnext_xlarge':
        from mindvision.classification.models import convnext_xlarge
        network = convnext_xlarge(pretrained=args_opt.pretrained, num_classes=args_opt.num_classes)

    # Set lr scheduler.
    if args_opt.lr_decay_mode == 'cosine_annealing_lr_v1':
        lr = warmup_cosine_annealing_lr_v1(lr=args_opt.lr, steps_per_epoch=step_size,
                                           warmup_epochs=20, max_epoch=args_opt.epoch_size,
                                           t_max=150, eta_min=0)

    # Define optimizer
    network_opt = nn.AdamWeightDecay(params=network.trainable_params(),
                                     learning_rate=lr,
                                     beta1=0.9,
                                     beta2=0.999,
                                     eps=1e-6,
                                     weight_decay=0.05)
    # Define loss function
    network_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Define metrics
    metrics = {'acc'}

    # Set the checkpoint config for the network.
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=step_size,
        keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix=args_opt.model,
                                    directory=ckpt_save_dir,
                                    config=ckpt_config)

    # Init the model.
    model = Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics)

    # Begin to train.
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor(lr)],
                dataset_sink_mode=args_opt.dataset_sink_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ConvNext train.")
    parser.add_argument('--model', required=True, default=None,
                        choices=["convnext_tiny",
                                 "convnext_small",
                                 "convnext_base",
                                 "convnext_large",
                                 "convnext_xlarge"])
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--epoch_size', type=int, default=90, help='Train epoch size.')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model.')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10, help='Max number of checkpoint files.')
    parser.add_argument('--ckpt_save_dir', type=str, default="./resnet", help='Location of training outputs.')
    parser.add_argument('--num_parallel_workers', type=int, default=8, help='Number of parallel workers.')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of batch size.')
    parser.add_argument('--repeat_num', type=int, default=1, help='Number of repeat.')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classification.')
    parser.add_argument('--lr_decay_mode', type=str, default="cosine_annealing_lr_v1", help='Learning rate decay mode.')
    parser.add_argument('--lr', type=str, default=4e-3, metavar='LR', help='learning rate.')
    parser.add_argument('--dataset_sink_mode', type=bool, default=True, help='The dataset sink mode.')
    parser.add_argument('--run_distribute', type=bool, default=True, help='Run distribute.')
    parser.add_argument('--resize', type=int, default=224, help='Resize the image.')

    args = parser.parse_known_args()[0]
    convnext_train(args)
