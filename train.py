import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, MaskedKLDivLoss, Transformer_Based_Model, KLDivLoss, FocalLoss, SimilarityAwareFocalLoss, SimilarityAwareFocalLoss_2
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pickle as pk
import datetime
import torch.nn as nn
from vision import confuPLT, sampleCount


def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):
    """Create samplers for training and validation sets.
    This function is especially useful for model validation during deep learning training.
    """
    size = len(trainset)
    idx = list(range(size))
    # Calculate validation set size based on the given ratio 'valid'
    split = int(valid * size)
    # Return two SubsetRandomSampler objects for training and validation
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])



def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader



if __name__ == '__main__':
    # Handle command line arguments
    parser = argparse.ArgumentParser()
    # Add arguments to the parser. Each argument has properties such as action, default value, type, help info, etc.
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')  # 16->2
    parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')  # Number of heads in multi-head attention
    parser.add_argument('--epochs', type=int, default=80, metavar='E', help='number of epochs')
    parser.add_argument('--temp', type=int, default=1.0, metavar='temp', help='temp')  # Temperature parameter, usually used to adjust softmax output
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')  # IEMOCAP MELD

    # Parse command line arguments and store them in args variable
    args = parser.parse_args()
    today = datetime.datetime.now()
    print(args)

    # Set cuda flag based on system CUDA availability and user specification to disable CUDA
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    # Set feature dimensions for audio, visual, and text based on dataset
    feat2dim = {'IS10': 1582, 'denseface': 342, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    D_m = D_audio + D_visual + D_text

    # Set number of speakers and classes based on dataset
    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 1
    # Print temperature parameter
    print('temp {}'.format(args.temp))

   
    if cuda:
        # Move model to GPU
        model.cuda()
        
    # kl_loss = MaskedKLDivLoss()
    kl_loss = KLDivLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    # Set loss function and data loaders based on dataset
    if args.Dataset == 'MELD':
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=0)
    elif args.Dataset == 'IEMOCAP':
      
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.001, batch_size=batch_size,
                                                                      num_workers=0)
    else:
        print("There is no such dataset")

    # Track best performance during validation
    best_fscore, best_loss, best_label, best_pred, best_mask, best_label2 = None, None, None, None, None, None
    # Store performance logs during training
    all_fscore, all_acc, all_loss = [], [], []

    # 3. Training loop: train and evaluate a Transformer-based model over epochs; record performance metrics.
    for e in range(n_epochs):
        start_time = time.time()

        # Training
        train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(model, loss_function, kl_loss, train_loader, e, optimizer, True)
        # Validation
        valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(model, loss_function, kl_loss, valid_loader, e)
        # Testing
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function, kl_loss, test_loader, e)
        all_fscore.append(test_fscore)
        all_acc.append(test_acc)

        # Update best F-Score and corresponding labels/predictions if improved
        if best_fscore == None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred, best_mask = test_label, test_pred, test_mask

        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)

        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
        if (e+1) % 10 == 0:
            # Print classification report and confusion matrix for best labels and predictions
            # print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
            # print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
            print(classification_report(best_label, best_pred, digits=4))
            print(confusion_matrix(best_label, best_pred))


    if args.tensorboard:
        writer.close()

    print('Best performance..')
    print('F1-Score: {}'.format(max(all_fscore)))
    # print('ACC: {}'.format(max(all_acc)))
    print('index: {}'.format(all_fscore.index(max(all_fscore)) + 1))

