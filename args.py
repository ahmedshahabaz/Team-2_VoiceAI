# Deep learning course

import os, argparse

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./data/bids_with_sensitive_recordings', help='Dataset directory')

    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--mode', type=str, default='train')

    #parser.add_argument('--crop_size', type=int, default=224, help='size for randomly or center cropping images')

    #parser.add_argument('--image_size', type=int, default=256, help='size to rescale images')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='base learning rate')

    parser.add_argument('--num_epochs', type=int, default=100, help='maximum number of epochs')

    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')

    parser.add_argument('--patience', type=int, default=15,help='maximum number of epochs to allow before early stopping')

    parser.add_argument('--comment', required=False, type=str, default = 'test', help='name for TensorboardX')

    args = parser.parse_args()

    return args
