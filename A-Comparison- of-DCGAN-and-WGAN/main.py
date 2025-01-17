import tensorflow as tf
import os
import argparse

## GAN Variants
from DCGAN import DCGAN
from WGAN import WGAN
from utils import show_all_variables
from utils import check_folder

def parse_args():
    desc = "GAN implementation"
    parser = argparse.ArgumentParser(description=desc)

    # Choose GAN type
    parser.add_argument('--gan_type', type=str, default='DCGAN',
                        choices=['DCGAN', 'WGAN'],
                        help='The type of GAN')

    # Set dataset as cifar10                    
    parser.add_argument('--dataset', type=str, default='cifar-10', choices=['cifar-10'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=60, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=62, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())


def check_args(args):
    check_folder(args.checkpoint_dir)
    check_folder(args.result_dir)
    check_folder(args.log_dir)

    assert args.epoch >= 1
    assert args.batch_size >= 1
    assert args.z_dim >= 1

    return args

def main():
    args = parse_args()
    if args is None:
      exit()

    models = [DCGAN, WGAN]
    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        gan = None
        for model in models:
            if args.gan_type == model.model_name:
                gan = model(sess,
                            epoch=args.epoch,
                            batch_size=args.batch_size,
                            z_dim=args.z_dim,
                            dataset_name=args.dataset,
                            checkpoint_dir=args.checkpoint_dir,
                            result_dir=args.result_dir,
                            log_dir=args.log_dir)
        if gan is None:
            raise Exception("No option")

        # Process Visualization
        gan.build_model()
        show_all_variables()
        gan.train()
        print("Finish Training")
        gan.visualize_results(args.epoch-1)
        print("Finish Testing")
        
        if args.dataset == 'cifar-10':
            gan.calculate_is()

if __name__ == '__main__':
    main()
