import os
import torch
from torch import nn
from torch import optim

from models import RNN
from utils.UCF_utils import sequence_generator, get_data_list

N_CLASSES = 101
BatchSize = 30

def train(model, train_data, test_data, epochs, steps_per_epoch, test_steps, weights_dir, input_shape, device):
    
    try:
        if os.path.exists(weights_dir):
            model.load_state_dict(torch.load(weights_dir))
            print('Load weights')
        train_generator = sequence_generator(train_data, BatchSize, input_shape, N_CLASSES)
        test_generator = sequence_generator(test_data, BatchSize, input_shape, N_CLASSES)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6, momentum=0.9, nesterov=True)
        model.to(device)
        
        print('Start fitting model')
        epochs = n_epoch
        steps = 0 
        train_loss = 0
        train_accuracy = 0
        print_every = 1

        for e in range(epochs):
            for i in range(steps_per_epoch):
                model.train()
                images, labels = next(train_generator)
                images, labels = images.to(device), labels.to(device)
                steps += 1

                optimizer.zero_grad()
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                train_loss = 1/(i+1)*loss.item() + i/(i+1)*train_loss
                correct = (labels == output.argmax(-1)).float().detach().numpy()
                train_accuracy = 1/(i+1)*correct.mean() + i/(i+1)*train_accuracy
                
                if steps % print_every == 0:
                    model.eval()
                    with torch.no_grad():
                        test_loss = 0
                        test_accuracy = 0
                        for j in range(test_steps):
                            test_images, test_labels = next(test_generator)
                            test_images, test_labels = test_images.to(device), test_labels.to(device)
                            test_output = model.forward(test_images)
                            loss = criterion(test_output, test_labels)
                            test_loss = 1/(j+1)*loss.item() + j/(j+1)*test_loss
                            test_correct = (test_labels == test_output.argmax(-1)).float().detach().numpy()
                            test_accuracy = 1/(j+1)*test_correct.mean() + j/(j+1)*test_accuracy
                    
                    print(f'Epoch: {e+1}/{epochs} - ',
                          f'Training Loss: {train_loss:.3f} - ',
                          f'Training Accuracy: {train_accuracy:.3f} - ',
                          f'Test Loss: {test_loss:.3f} - ',
                          f'Test Accuracy: {test_accuracy:.3f}')
                    
                    running_loss = 0
            
            if (e % 10 == 0) or (e == epochs-1):
                checkpoint = {
                    'epoch': e,
                    'state_dict': model.state_dict(),
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                }
                torch.save(checkpoint, os.path.join(weights_dir, f'epoch-{e}.pt'))
    
    except KeyboardInterrupt:
        print('Training time:')


if __name__ == '__main__':
    data_dir = 'D:/SUTD/Term-7/Deep_Learning/BigProject/Action_Detection_In_Videos/data/'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    video_dir = os.path.join(data_dir, 'CNN_Predicted')
    weights_dir = 'D:/SUTD/Term-7/Deep_Learning/BigProject/Action_Detection_In_Videos/models'

    rnn_weights_dir = os.path.join(weights_dir, 'rnn.h5')
    RNN_model = RNN.RNN(rnn_weights_dir, CNN_output)
    
    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    print('Train data size: ', len(train_data))
    print('Test data size: ', len(test_data))
    
    epochs = 200
    steps_per_epoch = 300
    test_steps = 100
    
    CNN_output = 1024
    input_shape = (10, CNN_output)

    device = 'cuda'
    
    train(RNN_model, train_data, test_data, epochs, steps_per_epoch, test_steps, rnn_weights_dir, input_shape, device)