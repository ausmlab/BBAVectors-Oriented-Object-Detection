import argparse
import train
import test
import eval
from datasets.dataset_dota import DOTA
from datasets.dataset_hrsc import HRSC
from models import ctrbox_net
import decoder
import os
import func_utils

# from reclassification.MACNN import MACNN


def parse_args():
    parser = argparse.ArgumentParser(description='BBAVectors Implementation')

    parser.add_argument('--num_epoch', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-4, help='Initial learning rate')
    parser.add_argument('--input_h', type=int, default=608, help='Resized image height')
    parser.add_argument('--input_w', type=int, default=608, help='Resized image width')
    parser.add_argument('--conf_thresh', type=float, default=0.18, help='Confidence threshold, 0.1 for general evaluation')
    parser.add_argument('--K', type=int, default=500, help='Maximum of objects')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of gpus, ngpus>1 for multigpu')
    parser.add_argument('--resume_train', type=str, default='', help='Weights resumed in training')
    parser.add_argument('--resume', type=str, default='model_50.pth', help='Weights resumed in testing and evaluation')
    parser.add_argument('--dataset', type=str, default='dota', help='Name of dataset')
    parser.add_argument('--data_dir', type=str, default='../Datasets/dota', help='Data directory')
    parser.add_argument('--phase', type=str, default='test', help='Phase choice= {train, test, eval}')
    parser.add_argument('--wh_channels', type=int, default=8, help='Number of channels for the vectors (4x2)')

    parser.add_argument('--classnames', type=str, default=None, help='path of a file with the name of all categories')

    parser.add_argument('--test_mode', type=str, default=None, help='either <show> or <save> the results of test phase')
    parser.add_argument('--test_save_dir', type=str, default=None, help='where to save results for test phase, applicable in save mode')

    parser.add_argument('--eval_w_train', action='store_true', help='Whether to perform evaluation while training (to check overfitting)')
    parser.add_argument('--eval_data_dir', type=str, default='/BBAV/DS/val1024', help='Directory pointing to evaluation data')
    parser.add_argument('--log_dir', type=str, default='/BBAV/logs', help='where to save tensorboard logs')

    parser.add_argument('--rcm', action='store_true', help='whether or not the objects should be re-classified by a separate model')
    parser.add_argument('--rcm_path', type=str, default='', help='model weights for classifying objects separately')
    parser.add_argument('--src_path', type=str, default='')
    parser.add_argument('--dst_path', type=str, default='')

    args = parser.parse_args()
    return args


def read_classnames(filepath):
    classname_file = open(filepath, 'r')
    classnames = [class_name.strip() for class_name in classname_file.readlines() if class_name.strip()]
    return classnames

if __name__ == '__main__':
    args = parse_args()
    dataset = {'dota': DOTA, 'hrsc': HRSC}

    # We'll set num_classes for both datasets to the number of classes in the
    #   current dataset, this is ok since the other value will not be used.
    if args.classnames is not None:
        classnames = read_classnames(args.classnames)
        num_classes = {'dota': len(classnames), 'hrsc': len(classnames)}
    else:
        classnames = ['car','small_truck','large_truck','bus','container','LCV']
        num_classes = {'dota': 6, 'hrsc': 1}

    heads = {'hm': num_classes[args.dataset],
             'wh': 10,
             'reg': 2,
             'cls_theta': 1
             }
    down_ratio = 4
    model = ctrbox_net.CTRBOX(heads=heads,
                              pretrained=True,
                              down_ratio=down_ratio,
                              final_kernel=1,
                              head_conv=256)

    decoder = decoder.DecDecoder(K=args.K,
                                 conf_thresh=args.conf_thresh,
                                 num_classes=num_classes[args.dataset])
    if args.rcm:
        rcm = MACNN()
    else:
        rcm = None

    if args.phase == 'train':
        ctrbox_obj = train.TrainModule(dataset=dataset,
                                       num_classes=num_classes,
                                       model=model,
                                       decoder=decoder,
                                       down_ratio=down_ratio,
                                       classnames=classnames)

        ctrbox_obj.train_network(args)
    elif args.phase == 'test':
        ctrbox_obj = test.TestModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder, save_dir=args.test_save_dir, mode=args.test_mode, classnames=classnames)
        ctrbox_obj.test(args, down_ratio=down_ratio)
    elif args.phase == 'eval':
        ctrbox_obj = eval.EvalModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder, classnames=classnames, rcm=rcm)
        ctrbox_obj.evaluation(args, down_ratio=down_ratio)
    elif args.phase == 'reclassify':
        func_utils.separate_results_by_file(args.src_path, args.dst_path)
