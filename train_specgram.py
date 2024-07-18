from args import get_parser
import torch
#from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, vgg16, vgg19, inception_v3
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import os, random, math

from models import *
from dataset import *
from fairness_metrics import *
from torch.optim.lr_scheduler import StepLR

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchsummary import summary


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------

def train_epoch(model, epoch, mode, data_loader, device, loss_func, optimizer, scheduler, writer=None):

    """
    Train or evaluate the model based on the mode.
    Arguments:
    - model: The model to train or evaluate
    - epoch: The current epoch number
    - mode: 'Train' or 'Validation'
    - data_loader: DataLoader for the data
    - device: Device to run the model on
    - loss_func: Loss function
    - optimizer: Optimizer for training (None for evaluation)
    - scheduler: Learning rate scheduler (None if not used)
    - writer: TensorBoard SummaryWriter object (None if not used)
    - log_batch_metrics: Function for logging metrics (None if not used)
    Returns:
    - Average loss and accuracy
    """
    
    total_correct_preds = 0.0
    total_elements = 1e-10
    total_loss = 0.0

    model_preds = []
    ground_truths = []

    with tqdm(data_loader, unit="batch") as iteration:

        for step, data_batch in (enumerate(iteration)):
            
            iteration.set_description(f"Epoch {epoch},{mode}")

            #mel_specgram_batch, yamnet_embedding_batch, opensmile_batch, age_batch, gender_batch, site_batch, binned_age_batch , diagnosis_batch = data_batch
            mel_specgram_batch, yamnet_embedding_batch, age_batch, gender_batch, site_batch, binned_age_batch , diagnosis_batch = data_batch
            label_batch = diagnosis_batch
            #bs, h, w = mel_specgram_batch.shape
            #mel_specgram_batch = mel_specgram_batch.unsqueeze(1).expand(bs,3,h,w)
            mel_specgram_batch = mel_specgram_batch.to(device)
            #yamnet_embedding_batch = yamnet_embedding_batch.to(device)
            label_batch = label_batch.to(device, dtype = torch.float32)
            
            #print(mel_specgram_batch.shape, label_batch.shape)

            if mode.upper() == "TRAIN":
                model.train()
                optimizer.zero_grad()
            
            output = model(mel_specgram_batch)
            #output = model(yamnet_embedding_batch)
            output = torch.sigmoid(output)
            total_elements += output.size(0)
            if output.dim() == 1:
                output = output.unsqueeze(1)
            label_batch = label_batch.unsqueeze(1)

            # Loss calculation
            batch_loss = loss_func(output, label_batch) # --> batch_size * 1
            total_loss += torch.sum(batch_loss).item()
        
            # Back-propagate the loss in the model & optimize
            if mode.upper() == "TRAIN":
                torch.mean(batch_loss).backward()
                optimizer.step()

            # Accuracy computation
            #_, pred_idx = torch.max(F.softmax(output, dim=1), dim=1)
            predictions = (output >= 0.5).float()
            batch_acc = torch.mean((predictions == label_batch).float()).item()
            total_correct_preds += torch.sum(predictions==label_batch).item()
            
            # Aggregating the predictions from model for later comparison
            if mode.upper()=="TEST":
                model_preds.extend(predictions.tolist())
                ground_truths.extend(label_batch.tolist())

            iteration.set_postfix({"Acc":round(total_correct_preds/total_elements,2), "Lss":round(total_loss/total_elements,2)})

            if writer is not None:
                writer.add_scalar(f'{mode}/Batch_Loss', torch.mean(batch_loss).item(), epoch * len(data_loader) + step)
                writer.add_scalar(f'{mode}/Batch_Accuracy', batch_acc, epoch * len(data_loader) + step)

    #print('\rEpoch: {}, {} accuracy: {}, loss: {}'.format(epoch,mode, accuracy, loss)) 
    if mode.upper()=="TEST":
        model_preds = np.array(model_preds).flatten()
        ground_truths = np.array(ground_truths).flatten()

    return total_loss/total_elements, total_correct_preds/total_elements, model_preds, ground_truths

# -------------------------------------------

def main(args):
    
    writer = SummaryWriter('./runs/%s'%args.comment)
    writer.add_text('Args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))

    dataset_dict = get_dataset(data_dir=args.data_dir,target_diagnosis=args.label,algo='DL',spec_gram=True)
    train_dataset, train_identities = dataset_dict['train_dataset']
    val_dataset, val_identities = dataset_dict['val_dataset']
    test_dataset, test_identities, test_dataset_DT = dataset_dict['test_dataset']
    
    #DT_test_dataset, DT_test_identities = dataset_dict['DT_test_dataset']
    #full_dataset, all_identities = dataset_dict['full_dataset']

    #train_dataset = torch.utils.data.Subset(train_dataset, range(500))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model = get_models(args)

    '''
    Line 110-114 is needed only if you want to train only the last layer.
    o/w comment those if want to train full model.
    '''

    '''
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True
    '''
    #model = vgg16(pretrained = True)
    #model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

    #print(summary(model, (1,192)))

    # create optimizer

    params = list(model.parameters())
    #params = filter(lambda p: p.requires_grad, model.parameters())
    
    #optimizer = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9, lr=args.learning_rate, weight_decay = 0.000001)

    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.000001)
    #optimizer = torch.optim.Adam(params, lr=0.0001)

    learning_rate_scheduler = None
    #learning_rate_scheduler = StepLR(optimizer, step_size = 7, gamma = 0.1)
    
    #cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    binary_crs_entropy_ls = torch.nn.BCELoss(reduction='none')
    print()
    writer.add_text("Optim", optimizer.__class__.__name__)

    model = model.to(device)

    print ("model created & starting training on", device, "...\n\n", )
    min_val_ls = float('inf')
    no_improve = 0
    max_val_acc = 0


    # Training logs
    _train_ = [[], []]
    _val_ = [[], []]

    '''
    Training script
    '''

    print()
    #model_summary_str = summary(model, (3, 257, 301))
    #print()

    for epoch in (range(args.num_epochs)):

        #model.train()
        train_ls,train_acc,_,_ = train_epoch(model = model, epoch = epoch, mode = "Train", data_loader = train_loader, device = device,
            loss_func = binary_crs_entropy_ls, optimizer = optimizer, scheduler = learning_rate_scheduler, writer = writer)

        model.eval()
        with torch.no_grad():
            val_ls,val_acc,_,_ = train_epoch(model = model, epoch = epoch, mode = "Validation", data_loader = val_loader, device = device,
                loss_func = binary_crs_entropy_ls, optimizer = optimizer, scheduler = learning_rate_scheduler, writer = writer)


        print()
        print("Epoch ", epoch)
        print(f"Train Accuracy: {train_acc:.4f} -- Loss: {train_ls:.4f}")
        print(f"Valid Accuracy: {val_acc:.4f} -- Loss: {val_ls:.4f}")

        # Early stopping and model saving
        is_best = False
        if val_ls < min_val_ls or val_acc > max_val_acc:
            no_improve = 0
            if val_ls < min_val_ls:
                min_val_ls = val_ls
            if val_acc > max_val_acc:
                max_val_acc = val_acc
            is_best = True
        else:
            no_improve += 1

        if is_best:
            directory = f"./saved_models/{args.comment}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(model.state_dict(), os.path.join(directory, f"best_model_epoch_{epoch}.pth"))
            #torch.save(model, os.path.join(directory, f"best_model_epoch_{epoch}.pth"))

        # Logging to TensorBoard
        writer.add_scalars('Performance', {
            'Train Accuracy': train_acc,
            'Validation Accuracy': val_acc,
            'Train Loss': train_ls,
            'Validation Loss': val_ls,
        }, epoch)

        _train_[0].append(train_acc)
        _train_[1].append(train_ls)
        _val_[0].append(val_acc)
        _val_[1].append(val_ls)


        # Check for early stopping
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
        test_ls,test_acc,model_preds,ground_truths = train_epoch(model = model, epoch = 0, mode = "Test", data_loader = test_loader, device = device,
            loss_func = binary_crs_entropy_ls, optimizer = None, scheduler = None, writer = writer)

        print()

    print(f"Test Accuracy : {test_acc:.4f} -- Test Loss: {test_ls:.4f}")

    # opensmile_df_test,feature_cols,label_cols = create_open_smile_df(test_dataset_DT,diagnosis_column=args.label)
    # _, _, _print_string_ = chi_DIR_plot(test_dataset_DT,opensmile_df_test,ground_truths,model_preds,attribute='gender',writer=None)
    # print(_print_string_)
    # equalized_metrics(opensmile_df_test,ground_truths,model_preds,attribute='gender',writer=None)


    # Save training and validation logs
    np.save(os.path.join(directory, "train_logs.npy"), np.array(_train_))
    np.save(os.path.join(directory, "val_logs.npy"), np.array(_val_))

    stacked_test_array = np.vstack((ground_truths, model_preds))
    np.save(os.path.join(directory, "test_gt_pred.npy"), stacked_test_array)
    np.save(os.path.join(directory, "test_ids.npy"), np.array(test_identities))

    writer.close()



if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    main(args)
