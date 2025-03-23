import sys
import timeit
import keras # type: ignore

import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torch.optim as optim # type: ignore

from sklearn.metrics import r2_score # type: ignore

from keras.layers import Layer
import keras.backend as K
from keras.saving import register_keras_serializable

import preprocess as pp


# CNN model
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 5)
        self.conv2 = nn.Conv1d(32, 128, 3)
        self.conv3 = nn.Conv1d(128, 64, 3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2048, 1024) # Adjust input dimension based on the output of conv3
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.output_cnn = nn.Linear(256, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.output_cnn(x))
        return x

# Define attention layer
@register_keras_serializable(package="AttLayer")
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()

# Attention mechanism
class Attention(nn.Module):
    def __init__(self, attention_input_dim, attention_dim, attention_output_dim):
        super(Attention, self).__init__()
        self.attention = nn.Conv1d(attention_input_dim, attention_dim, 3)
        self.lstm = nn.LSTM(attention_dim, attention_output_dim, batch_first=True, dropout=0.2)

    def forward(self, x):
        x = F.relu(self.attention(x))
        x, _ = self.lstm(x)
        x = attention()(x)
        return x

# Combined model
class CombinedModel(nn.Module):
    def __init__(self, cnn, mlp_hidden_dim, mlp_output_dim, attention_dim, attention_output_dim, dropout_rate):
        super(CombinedModel, self).__init__()
        self.cnn = cnn
        self.mlp = nn.Sequential(
            nn.Linear(3, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_output_dim),
            nn.ReLU()
        )
        self.attention = Attention(attention_input_dim=mlp_output_dim+128, attention_dim=attention_dim, 
                                   attention_output_dim=attention_output_dim) # Adjust input dimension based on the concatenated output
        self.fc = nn.Linear(attention_output_dim, 1) # Adjust input dimension based on LSTM output

    def forward(self, dataset, train=True):
        # Check if gnn_data[-1] is a tensor
        if isinstance(dataset[-1], torch.Tensor):
            correct_values = dataset[-1].view(-1, 1)  # Adjust size to match predicted values
        else:
            correct_values = torch.cat(dataset[-1]).view(-1, 1)  # Adjust size to match predicted values

        cnn_input = dataset[0]
        mlp_input = dataset[1]

        cnn_output = self.cnn(cnn_input)
        mlp_output = self.mlp(mlp_input)

        merged = torch.cat((cnn_output, mlp_output), dim=1).unsqueeze(2)
        merged = F.batch_norm(merged)
        attention_output = self.attention(merged)
        predicted_values = F.relu(self.output(attention_output))

        if train:
            loss = F.mse_loss(predicted_values, correct_values)
            return loss
        else:
            with torch.no_grad():
                predicted_values = predicted_values.to('cpu').data.numpy()
                correct_values = correct_values.to('cpu').data.numpy()
                predicted_values = np.concatenate(predicted_values)
                correct_values = np.concatenate(correct_values)
            return predicted_values, correct_values
    

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            fingerprints_batch = torch.stack([torch.tensor(fp, dtype=torch.float32).to(device) for fp in data_batch[0]])
            homo_lumo_batch = torch.stack([hl.clone().detach().to(device) for hl in data_batch[1]]).to(device)
            correct_values_batch = torch.tensor(data_batch[2], dtype=torch.float32).to(device)
            loss = self.model((fingerprints_batch, correct_values_batch), homo_lumo_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total
    

class Tester(object):
    def __init__(self, model):
        self.model = model

    def test_regressor(self, dataset):
        N = len(dataset)
        all_predicted_values = []
        all_correct_values = []
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            fingerprints_batch = torch.stack([torch.tensor(fp, dtype=torch.float32).to(device) for fp in data_batch[0]])
            homo_lumo_batch = torch.stack([hl.clone().detach().to(device) for hl in data_batch[1]]).to(device)
            correct_values_batch = torch.tensor(data_batch[2], dtype=torch.float32).to(device)
            predicted_values, correct_values = self.model((fingerprints_batch, correct_values_batch), homo_lumo_batch, train=False)
            all_predicted_values.append(predicted_values)
            all_correct_values.append(correct_values)

        all_predicted_values = np.concatenate(all_predicted_values)
        all_correct_values = np.concatenate(all_correct_values)
        r2 = r2_score(all_predicted_values, all_correct_values)
        return r2

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')
            

if __name__ == "__main__":

    (dataset, mlp_hidden_dim, mlp_output_dim, attention_dim, attention_output_dim,
     batch_train, batch_test, lr, lr_decay, dropout_rate, decay_interval, weight_decay,
     iteration, setting) = sys.argv[1:]
    
    (mlp_hidden_dim, mlp_output_dim, attention_dim, 
     attention_output_dim, batch_train, 
     batch_test, decay_interval, iteration) = map(int, [mlp_hidden_dim, mlp_output_dim, attention_dim, 
                                                        attention_output_dim, batch_train, batch_test, 
                                                        decay_interval, iteration])
    lr, lr_decay, dropout_rate, weight_decay = map(float, [lr, lr_decay, dropout_rate, weight_decay])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    print('-'*100)

    print('Preprocess the dataset.')
    print('Just a moment......')
    (dataset_train, dataset_test, dataset_val) = pp.create_datasets(dataset, device)
    print('-'*100)


    print('Creating a model.')
    torch.manual_seed(1234)
    cnn = ConvolutionalNeuralNetwork().to(device)
    model = CombinedModel(cnn, mlp_hidden_dim, mlp_output_dim, attention_dim, attention_output_dim, dropout_rate).to(device)

    trainer = Trainer(model)
    tester = Tester(model)
    print('# of model parameters:', sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*100)

    file_result = '../output/r2_result/result_.txt'
    result = 'Epoch\tTime(sec)\tLoss_val\tLoss_train\tR2_val\tR2_train'

    with open(file_result, 'w') as f:
        f.write(result + '\n')

    print('Start training.')
    print('The result is saved in the output directory every epoch!')

    np.random.seed(1234)

    start = timeit.default_timer()

    for epoch in range(iteration):

        epoch += 1
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_val = trainer.train(dataset_val)
        loss_train = trainer.train(dataset_train)

        prediction_val = tester.test_regressor(dataset_val)
        prediction_train = tester.test_regressor(dataset_train)

        time = timeit.default_timer() - start

        if epoch == 1:
            minutes = time * iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about', hours, 'hours', minutes, 'minutes.')
            print('-'*100)
            print(result)

        result = '\t'.join(map(str, [epoch, time, loss_val, loss_train, prediction_val, prediction_train]))
        tester.save_result(result, file_result)

        print(result)

    def graph_result(file):

        df = pd.read_csv(file, delimiter='\t')

        # Plot training and validation loss
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(df['Loss_train'], label='Training Loss')
        plt.plot(df['Loss_dev'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(df['R2_train'], label='Training Accuracy')
        plt.plot(df['R2_dev'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.savefig('../output/graph_result/graph_result_.png')

    # Test with dataset_test
    r2_test = tester.test_regressor(dataset_test)
    with open(file_result, 'w') as f:
        f.write('\n' + r2_test + '\n')
    print(f'R^2 score on test dataset: {r2_test}')

    # Print and save graph result
    graph_result(file_result)

    # Save model
    torch.save(model.state_dict(), '../model/trained_model_.pth')