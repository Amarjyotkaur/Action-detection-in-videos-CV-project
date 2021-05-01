import os
from UCF_utils import image_from_sequence_generator, sequence_generator, get_data_list
from models.finetuned_resnet import finetuned_resnet
from models.temporal_cnn import temporal_CNN
import time 
from torch import nn
from torch import optim
from UCF_preprocessing import regenerate_data

def train(model, n_epoch, learningrate, train_data, test_data, input_shape, device, data_dir, model_dir, optical_flow):
    N_CLASSES = 174
    BatchSize = 32

    #Get the corresponding Data Generator
    if optical_flow:
        trainloader = sequence_generator(train_data, BatchSize, input_shape, N_CLASSES)
        testloader = sequence_generator(test_data, BatchSize, input_shape, N_CLASSES)
    else:
        trainloader = image_from_sequence_generator(train_data, BatchSize, input_shape, N_CLASSES)
        testloader = image_from_sequence_generator(test_data, BatchSize, input_shape, N_CLASSES)

    #Set-up Loss function and Optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learningrate, weight_decay=1e-6, momentum=0.9, nesterov=True)

    #Connect to GPU
    model.to(device)
    #start = time.time()
    
    #Instanciate Parameters
    epochs = n_epoch
    steps = 0 
    running_loss = 0
    train_accuracy = 0
    #train_recall_score = 0
    #train_precision_score = 0 
    print_every = 1

    #TRAINING PHASE
    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            steps += 1

            #Forward Propagate and obtain Prediction
            optimizer.zero_grad()
            output = model.forward(images)
            pred = torch.flatten(torch.round(output)).int()

            #Obtain Labels
            labels = labels[:,1]
            #output = torch.flatten(output)

            #Backpropagate
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
                    """
        #Saving Checkpoints
        if (e % 10 == 0) or (e == epochs-1): 
            save_checkpoint(model, optimizer, e, os.path.join(model_dir, 'epoch-{}.pt'.format(e)))
    return model

def save_checkpoint(model, optimizer, n_epoch, path):
    checkpoint = {'state_dict': model.state_dict(),
                  'opti_state_dict': optimizer.state_dict(),
                  }
    torch.save(checkpoint, path)


if __name__ == '__main__':
    
    #extract frames from videos as npy files
    sequence_length = 10
    image_size = (216, 216, 3)
    data_dir = 'D:/SUTD/Term-7/Deep_Learning/BigProject/Action_Detection_In_Videos/data/' 
    list_dir = os.path.join(data_dir, 'TrainTestlist')
    smtsmt_dir = os.path.join(data_dir, 'smtsmt')
    #frames_dir = os.path.join(data_dir, 'frames/mean.npy')

    regenerate_data(data_dir, list_dir, smtsmt_dir)

    #Train CNN
    data_dir = 'D:/SUTD/Term-7/Deep_Learning/BigProject/Action_Detection_In_Videos/data'
    list_dir = os.path.join(data_dir, 'TrainTestList')
    weights_dir = 'D:/SUTD/Term-7/Deep_Learning/BigProject/Action_Detection_In_Videos/models/weights'
    video_dir = os.path.join(data_dir, 'smtsmt-Preprocessed-OF')
    n_epoch = 1000
    lr = 0.001
    input_shape = (10, 216, 216, 3)
    device = "gpu"
    model_dir = 'D:/SUTD/Term-7/Deep_Learning/BigProject/Action_Detection_In_Videos/models'
    optical_flow= False

    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    
    model = finetuned_resnet(include_top=True, weights_dir=weights_dir)
    model = train(model, n_epoch, lr, train_data, test_data, input_shape, device, data_dir, model_dir, optical_flow)

    # train CNN using optical flow as input
    # weights_dir = os.path.join(weights_dir, 'temporal_cnn_42.h5')
    # train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    # video_dir = os.path.join(data_dir, 'OF_data')
    # input_shape = (216, 216, 18)
    # model = temporal_CNN(input_shape, N_CLASSES, weights_dir, include_top=True)
    # fit_model(model, train_data, test_data, weights_dir, input_shape, optical_flow=True)
