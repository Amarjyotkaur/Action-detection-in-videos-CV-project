import os
import keras.callbacks
from utils.UCF_utils import image_from_sequence_generator, sequence_generator, get_data_list
from models.finetuned_resnet import finetuned_resnet
from models.temporal_CNN import temporal_CNN
import time 
from torch import nn
from torch import optim
from torch.pytorch_widedeep.callbacks import ModelCheckpoint
from torch.pytorch_widedeep.callbacks import EarlyStopping
from utils.UCF_preprocessing import regenerate_data

def train(model, n_epoch, lr, input_shape, device, data_dir, model_dir, optical_flow):
    N_CLASSES = 101
    BatchSize = 32

    if optical_flow:
            trainloader = sequence_generator(train_data, BatchSize, input_shape, N_CLASSES)
            testloader = sequence_generator(test_data, BatchSize, input_shape, N_CLASSES)
    else:
        trainloader = image_from_sequence_generator(train_data, BatchSize, input_shape, N_CLASSES)
        testloader = image_from_sequence_generator(test_data, BatchSize, input_shape, N_CLASSES)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-6, momentum=0.9, nesterov=True)

    model.to(device)
    #start = time.time()
    
    epochs = n_epoch
    steps = 0 
    running_loss = 0
    train_accuracy = 0
    #train_recall_score = 0
    #train_precision_score = 0 
    print_every = 1

    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            steps += 1

            optimizer.zero_grad()
            output = model.forward(images)
            pred = torch.flatten(torch.round(output)).int()

            labels = labels[:,1]
            output = torch.flatten(output)
            #print("output", output)
            #print("label", labels)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            """ if steps % print_every == 0:
                    # Eval mode for predictions
                    model.eval()

                    # Turn off gradients for validation
                    with torch.no_grad():
                        val_loss, val_accuracy, val_recall_score, val_precision_score = validation(model, validloader, criterion, device)

                    print("Epoch: {}/{} - ".format(e+1, epochs),
                        "Training Loss: {:.3f} - ".format(running_loss/print_every),
                        "Validation Loss: {:.3f} - ".format(val_loss/len(validloader)),
                        "Validation Accuracy: {:.3f}".format(val_accuracy/len(validloader)),
                        "Validation Recall-score: {:.3f}".format(val_recall_score/len(validloader)),
                        "Validation Precision-score: {:.3f}".format(val_precision_score/len(validloader))
                    )
                    
                    model.train() 
        if (e % 10 == 0) or (e == epochs-1): 
            save_checkpoint(model, optimizer, epochs, train_loss_ls, val_loss_ls, os.path.join(model_dir, 'epoch-{}.pt'.format(e)))
    """
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    UCF_dir = os.path.join(data_dir, 'UCF-101')
    regenerate_data(data_dir, list_dir, UCF_dir)
    return model



if __name__ == '__main__':
    data_dir = '/home/changan/ActionRecognition/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    weights_dir = '/home/changan/ActionRecognition/models'
    video_dir = os.path.join(data_dir, 'UCF-Preprocessed-OF')
    train_data, test_data, class_index = get_data_list(list_dir, video_dir)

    n_epoch = 1000
    lr = 0.001
    input_shape = (10, 216, 216, 3)
    device = "gpu"
    data_dir = ""
    model_dir = ""
    optical_flow= False
    model = finetuned_resnet(include_top=True, weights_dir=weights_dir)
    model = train(model, n_epoch, lr, input_shape, device, data_dir, model_dir, optical_flow)

    # train CNN using optical flow as input
    # weights_dir = os.path.join(weights_dir, 'temporal_cnn_42.h5')
    # train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    # video_dir = os.path.join(data_dir, 'OF_data')
    # input_shape = (216, 216, 18)
    # model = temporal_CNN(input_shape, N_CLASSES, weights_dir, include_top=True)
    # fit_model(model, train_data, test_data, weights_dir, input_shape, optical_flow=True)
