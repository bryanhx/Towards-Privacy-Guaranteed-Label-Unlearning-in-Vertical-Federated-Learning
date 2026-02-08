import argparse

def config():
    parser = argparse.ArgumentParser(description="Vertical federated Unlearning Experiments")

    parser.add_argument('--print_decimal_digits', default=6, type=int,
                        help='How many decimal places print out in logger.')
    parser.add_argument('--verbose', default=1, type=int,
                        help='Whether to print verbose logging info')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_gpu', default=True,
                        help='Whether to use GPU or not')
    ################################### Experiment ###############################################
    parser.add_argument('--outdir', default='exp', type=str,
                        help='output directory')
    parser.add_argument('--expname', default='', type=str,
                        help='detailed exp name to distinguish different sub-exp')
    # parser.add_argument('--expname_tag', default='', type=str,
    #                     help=' detailed exp tag to distinguish different sub-exp with the same expname')

    ##################################### Dataset #################################################
    parser.add_argument('--data', default='cifar10', type=str,
                        help='name of dataset',
                        choices=['cifar10', 'mnist', 'cifar100', 'modelnet', 'yahoo', 'mri', 'ct'])
    parser.add_argument('--data_path', default='data', type=str,
                        help='path of dataset')
    parser.add_argument('--batch_size', default=32, type=int,
                        help = 'Number of data per batch')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--num_classes', default=10, type=int,
                        help = 'Number of classes in data')

    #################################### Training ################################################
    parser.add_argument('--epochs', default=100, type=int,
                        help = 'Number of epochs in training')
    parser.add_argument('--mode', default='full', type=str,
                        help = 'Train with full dataset or retrain with remain dataset',
                        choices=['full', 'retrain'])
    parser.add_argument('--std', default=0.1, type=float)
    parser.add_argument('--percent', default=0.999, type=float)

    ##################################### Unlearn #################################################
    parser.add_argument('--unlearn_method', default='LUV', type=str,
                        help='Unlearning method')
    parser.add_argument('--unlearn_client_method', default='ours', type=str,
                        help='Unlearning client method')
    parser.add_argument('--unlearn_class', default=0, type=int,
                        help='Unlearning class')
    parser.add_argument('--unlearn_class_num', default=1, type=int,
                        help='Number of unlearn class, eg. 1 class, 2 classes, 4 classes unlearning')
    parser.add_argument('--unlearn_lr', default=0.0000002, type=float,
                        help='Learning rate in unlearning')
    parser.add_argument('--unlearn_samples', default=40, type=int,
                        help='Samples of data use in EE unlearning')
    parser.add_argument('--unlearn_epochs', default=16, type=int,
                        help='Unlearning epochs')

    ###################################### Model ##################################################
    parser.add_argument('--model_type', default='resnet18', type=str,
                        help='Type of model used')
    parser.add_argument('--optimizer_lr', default=1e-3, type=float,
                        help='Type of model used')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)

    return parser.parse_args()
