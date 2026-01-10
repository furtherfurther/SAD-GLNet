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
    This function is particularly useful for model validation during deep learning model training.
    """
    size = len(trainset)
    idx = list(range(size))
    # Calculate validation set size based on the provided valid ratio
    split = int(valid*size)
    # Return two SubsetRandomSampler objects for training and validation sets respectively
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    """
    Create data loaders for MELD dataset training, validation, and testing.
    These loaders use the DataLoader class and can specify batch size, whether to use multi-threaded data loading, 
    whether to pin memory, and other parameters.
    :param num_workers: Number of subprocesses to use for data loading, default is 0
    :param pin_memory: Whether to load data into pinned memory to accelerate data transfer to CUDA devices, default is False
    :return: Returns a tuple containing training, validation, and testing loaders
    """
    trainset = MELDDataset('data/meld_multimodal_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    # Create DataLoader for training set with specified batch size, sampler, collate function, number of workers, and pin_memory option
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

    testset = MELDDataset('data/meld_multimodal_features.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    # Return a tuple containing training, validation, and testing loaders
    return train_loader, valid_loader, test_loader


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


def train_or_eval_model(model, loss_function, kl_loss, dataloader, epoch, optimizer=None, train=False, gamma_1=1.0, gamma_2=1.0, gamma_3=1.0):
    """
    Train or evaluate a model, depending on the value of the train parameter. 
    It handles data loading, forward propagation, loss calculation, backpropagation (if in training mode), and performance evaluation.
    :param kl_loss: Function for calculating KL divergence loss
    :param train: Boolean value indicating whether to train the model or evaluate the model
    :param gamma_1: Coefficient for weighting different loss terms
    """
    losses, preds, labels, masks = [], [], [], []
    labels_g = []
    # Ensure optimizer is provided in training mode
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    # Iterate through data in dataloader
    for data in dataloader:
        # If in training mode, clear optimizer gradients
        if train:
            optimizer.zero_grad()
        # Move data to device (CUDA device if using GPU) and prepare data
        # Can guess dataset features are textf, visuf, acouf, qmask, umask, label, with the last dimension being label
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        # qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        # Call model for forward propagation, get log probabilities and probabilities for different modalities and multimodal fusion
        log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
        kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob = model(textf, visuf, acouf, umask, qmask, lengths)

        # Calculate main loss and KL divergence loss, and combine them into total loss
        lp_1 = log_prob1.view(-1, log_prob1.size()[2])
        lp_2 = log_prob2.view(-1, log_prob2.size()[2])
        lp_3 = log_prob3.view(-1, log_prob3.size()[2])
        lp_all = all_log_prob.view(-1, all_log_prob.size()[1])
        labels_ = label.view(-1)

        kl_lp_1 = kl_log_prob1.view(-1, kl_log_prob1.size()[1])
        kl_lp_2 = kl_log_prob2.view(-1, kl_log_prob2.size()[1])
        kl_lp_3 = kl_log_prob3.view(-1, kl_log_prob3.size()[1])
        # print("kl_all_prob", kl_all_prob.shape)  # ([726, 6])
        kl_p_all = kl_all_prob.view(-1, kl_all_prob.size()[1])
        # print("kl_p_all", kl_p_all.shape)  # ([726, 6])

        loss_weights = torch.FloatTensor([1 / 0.086747,
                                          1 / 0.144406,
                                          1 / 0.227883,
                                          1 / 0.160585,
                                          1 / 0.127711,
                                          1 / 0.252668])
        
        loss_function_1 = nn.NLLLoss(loss_weights.to(torch.device("cuda:0")) if cuda else loss_weights)  # IEMOCAP
        loss_function_g = SimilarityAwareFocalLoss()  # MELD
        loss_function_3 = SimilarityAwareFocalLoss_2()
        # print("label.shape", label.shape)  # ([16, 74])
        # print("labels_.shape", labels_.shape)  # ([1184])

        label_g = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        # print("lp_all", all_log_prob.shape)  # torch.Size([838, 6])
        # print("label_g.shape", label_g.shape)  # torch.Size([838])
        # print("kl_p_all", kl_p_all.shape)  # ([838, 6])  ([851, 6]) ([759, 6])
        # print("kl_lp_1", kl_lp_1.shape)  # ([1760, 6])  ([759, 6])
        # print("kl_lp_2", kl_lp_2.shape)  #
        # print("kl_lp_3", kl_lp_3.shape)  #
        # print("umask", umask.shape)  # ([16, 110])

        # print("kl_lp_1", kl_lp_1.shape)  # ([1152, 6])

        loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
                gamma_2 * (loss_function_g(lp_1, labels_) + loss_function_g(lp_2, labels_) + loss_function_g(lp_3, labels_)) + \
                gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3, kl_p_all))

        lp_ = all_prob.view(-1, all_prob.size()[1])
        # print("lp_", lp_.shape)  # [718, 6])
        # Convert predicted probabilities to predicted labels
        pred_ = torch.argmax(lp_, 1)
        # print("pred_", pred_.shape)  # ([718])
        # print("labels_", labels_.shape)  # ([1760])
        # print("label_g", label_g.shape)  # ([718])
        # print("labels", labels)

        # Add loss, predictions, labels, and masks to respective lists
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        labels_g.append(label_g.data.cpu().numpy())

        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())

        # If in training mode, perform backpropagation and update model parameters
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        labels_g = np.concatenate(labels_g)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    # Calculate average loss, accuracy, and F1 score
    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    # print("labels", labels.shape)  # (9512,)
    # print("preds", preds.shape)  # (5810,) Dimension mismatch for prediction, dimensions need to be consistent
    # print("label_g", label_g.shape)  # ([718])
    # print("masks", masks.shape)  # (9512,)
    # avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)
    # avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
    avg_accuracy = round(accuracy_score(labels_g, preds) * 100, 2)
    avg_fscore = round(f1_score(labels_g, preds, average='weighted') * 100, 2)
    # Return calculated average loss, accuracy, labels, predictions, masks, and F1 score
    # return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore
    return avg_loss, avg_accuracy, labels_g, preds, masks, avg_fscore


if __name__ == '__main__':
    """Python script that uses the argparse library to parse command line arguments.
    This way, users can easily adjust model hyperparameters without modifying the code itself.
    """
    # For handling command line arguments
    parser = argparse.ArgumentParser()
    # Add parameters to the parser. Each parameter has some attributes like action, default value, type, help information, etc.
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')  # 16->2
    parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')  # Number of heads in multi-head attention mechanism
    parser.add_argument('--epochs', type=int, default=80, metavar='E', help='number of epochs')
    parser.add_argument('--temp', type=int, default=1.0, metavar='temp', help='temp')  # Temperature parameter, typically used to adjust softmax output
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')  # IEMOCAP MELD

    # Parse command line arguments and store them in the args variable
    args = parser.parse_args()
    today = datetime.datetime.now()
    print(args)

    # Set cuda flag based on whether the system supports CUDA and whether the user specifies not to use CUDA
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
    # Set audio, visual, and text feature dimensions based on the dataset
    feat2dim = {'IS10': 1582, 'denseface': 342, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    D_m = D_audio + D_visual + D_text

    # Set number of speakers and number of classes based on the dataset
    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 1
    # Print temperature parameter
    print('temp {}'.format(args.temp))

    # 2. This code is the preparation stage for training and evaluation process, ensuring that the model, loss function, and optimizer are correctly set, and data loaders are ready to provide training and validation data.
    model = Transformer_Based_Model(args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,
                                        n_classes=n_classes,
                                        hidden_dim=args.hidden_dim,
                                        n_speakers=n_speakers,
                                        dropout=args.dropout)


    def get_model_size(model):
        dtype_size_map = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int8: 1
        }

        total_params = sum(p.numel() for p in model.parameters())
        # Get dtype of model parameters (default is to take dtype of first parameter)
        param_dtype = next(model.parameters()).dtype
        bytes_per_param = dtype_size_map.get(param_dtype, 4)  # Default is float32

        param_size_bytes = total_params * bytes_per_param
        param_size_mb = param_size_bytes / (1024 * 1024)

        print(f"total parameters: {total_params}")
        print(f"parameter dtype: {param_dtype}")
        print(f"model size: {param_size_mb:.2f} MB")
        return total_params, param_size_mb


    # Usage example
    get_model_size(model)

    if cuda:
        # Move model to GPU
        model.cuda()
        
    # kl_loss = MaskedKLDivLoss()
    kl_loss = KLDivLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    # Set loss function and data loaders based on the dataset
    if args.Dataset == 'MELD':
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=0)
    elif args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.001, batch_size=batch_size,
                                                                      num_workers=0)
    else:
        print("There is no such dataset")

    # For tracking best performance during validation
    best_fscore, best_loss, best_label, best_pred, best_mask, best_label2 = None, None, None, None, None, None
    # For storing performance records during training
    all_fscore, all_acc, all_loss = [], [], []

    # 3. This code is a training loop for training and evaluating a Transformer-based model. It iterates through the specified number of epochs, performing training, validation, and testing in each epoch, and records performance metrics.
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

        # If this is the first iteration or the current test F-Score is higher than the previous best, update best F-Score and corresponding labels and predictions
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



  

    print(classification_report(best_label, best_pred, digits=4))
    print(confusion_matrix(best_label, best_pred))
