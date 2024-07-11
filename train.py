from args import get_parser
import torch
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, vgg16, vgg19, inception_v3
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import os, random, math

import models

from dataset import get_loader, get_test_loader
from torch.optim.lr_scheduler import StepLR

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------

def train(model, epoch, mode, data_loader, device, loss_func, optimizer, scheduler):
    
    total_correct_preds = 0.0
    total_elements = 1e-10
    loss = 0.0

    model_preds = [[]]* len(data_loader)

    with tqdm(data_loader, unit="batch") as tepoch:

        for step, (image_input, class_idxs) in (enumerate(tepoch)):
            
            tepoch.set_description(f"Epoch {epoch},{mode}")

            class_idxs = class_idxs.to(device)
            image_input = image_input.to(device)
            
            if (mode == "Train" or mode == "train"):
                model.train()
                optimizer.zero_grad()
            
            output = model(image_input)
            total_elements += output.size(0)
            batch_loss = loss_func(output, class_idxs) # --> batch_size * 1
            # aggregate loss for logging
            loss += torch.sum(batch_loss).item()
        
            # back-propagate the loss in the model & optimize
            if mode == "train" or mode == "Train":
                torch.mean(batch_loss).backward()
                optimizer.step()

            # accuracy computation
            _, pred_idx = torch.max(F.softmax(output, dim=1), dim=1)

            # Aggregating the predictions from model for later comparison
            model_preds[step] = pred_idx.tolist()

            correct_preds_batch = torch.sum(pred_idx==class_idxs).item()
            total_correct_preds += correct_preds_batch
            
            tepoch.set_postfix({"Acc":round(total_correct_preds/total_elements,2), 
                "Lss":round(loss/total_elements,2)})

    #print('\rEpoch: {}, {} accuracy: {}, loss: {}'.format(epoch,mode, accuracy, loss)) 

    return loss/total_elements, total_correct_preds/total_elements, model_preds

# -------------------------------------------

def main(args):
    
    writer = SummaryWriter('./runs/%s'%args.comment)

    writer.add_text('Args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))

    num_classes = 19

    dataset, train_loader, val_loader = get_loader(args.data_dir, batch_size=args.batch_size, shuffle=True, 
                                           num_workers=args.num_workers, drop_last=False, args=args)
    args.mode = "test"

    test_dataset, test_loader = get_test_loader(args.data_dir, batch_size=args.batch_size, shuffle=False, 
                                           num_workers=args.num_workers, drop_last=False, args=args)

    args.mode = "train"

    device = torch.device("cuda:"+ str(args.device) if torch.cuda.is_available() else 'cpu')

    #model = resnet18(pretrained = True)
    #model.fc = nn.Linear(512*1*1,num_classes)

    #model = resnet50(pretrained = True)
    #model.fc = nn.Linear(2048,num_classes)

    model = resnet101(pretrained = True)
    model.fc = nn.Linear(2048,num_classes)

    '''
    Line 110-114 is needed only if you want to train only the last layer.
    o/w comment those if want to train full model.
    '''

    for param in model.parameters():
        param.requires_grad = False
    
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

    #model = vgg16(pretrained = True)
    #model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

    print(model)

    # create optimizer

    #params = list(model.parameters())

    params = filter(lambda p: p.requires_grad, model.parameters())
    
    #optimizer = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9, lr=args.learning_rate, weight_decay = 0.000001)

    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.000001)


    learning_rate_scheduler = None
    learning_rate_scheduler = StepLR(optimizer, step_size = 15, gamma = 0.1)
    
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

    print()

    writer.add_text("Optim", optimizer.__class__.__name__)

    model = model.to(device)

    print ("model created & starting training on", device, "...\n\n", )
    min_val_ls = 100000000000000000000
    no_improve = 0
    max_val_acc = 0

    # Training script

    _train_ = [list(), list()]
    _val_ = [list(), list()]

    for epoch in (range(args.num_epochs)):

        #model.train()
        train_ls , train_acc, _ = train(model = model, epoch = epoch, mode = "Train", data_loader = train_loader, device = device,
            loss_func = cross_entropy_loss, optimizer = optimizer, scheduler = learning_rate_scheduler)

        model.eval()
        with torch.no_grad():

            val_ls, val_acc, _ = train(model = model, epoch = epoch, mode = "Validation", data_loader = val_loader, device = device,
                loss_func = cross_entropy_loss, optimizer = optimizer, scheduler = learning_rate_scheduler)


        print()
        print("Epoch ", epoch)
        print("\rTrain Accuracy:", train_acc, " -- Loss:", train_ls)
        print("\rValid Accuracy:", val_acc, " -- Loss:", val_ls)

        # Early stopping

        if val_ls <= min_val_ls  or max_val_acc <= val_acc:
            no_improve = 0

            if val_ls <= min_val_ls:
                min_val_ls = val_ls

            if val_acc >= max_val_acc:
                max_val_acc = val_acc

            directory = "./saved_models/" + args.comment
            if os.path.exists(directory) is False:
                os.mkdir(directory)
            torch.save(model.state_dict(), directory + "/dl_project_" + str(epoch) + ".pth")
        else:
            no_improve +=1

        writer.add_scalars(f'',{
            'Acc_trn': train_acc,
            'Acc_val': val_acc,
            'Ls_trn': train_ls,
            'Ls_val': val_ls,}, epoch)

        _train_[0].append(train_acc)
        _train_[1].append(train_ls)

        _val_[0].append(val_acc)
        _val_[1].append(val_ls)


        if no_improve == args.patience:
            print()
            print("Early Stopping!!!")
            break
        
        if learning_rate_scheduler is not None:
            learning_rate_scheduler.step()

        print()

    print()

    print("*** Computing Performance on Test Data *** ")
    print()

    args.mode = 'test'

    model.eval()
    with torch.no_grad():
        test_ls , test_acc , _ = train(model = model, epoch = 0, mode = "Test", data_loader = test_loader, device = device,
            loss_func = cross_entropy_loss, optimizer = None, scheduler = None)

        print()

        print("Test Accuracy : ", test_acc, "Test Loss: ", test_ls)

    np.save(directory+"/train_logs.npy", np.array(_train_))
    np.save(directory+"/val_logs.npy", np.array(_val_))

if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)
    random.seed(12345)
    np.random.seed(12345)
    main(args)
