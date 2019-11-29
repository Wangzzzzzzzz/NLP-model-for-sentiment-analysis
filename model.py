from allennlp.modules.elmo import Elmo, batch_to_ids
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os, nltk, random
from nltk.tokenize import word_tokenize
nltk.download('punkt')
random.seed(12345)

# initialize torch device
device = "cuda" if torch.cuda.is_available() else "cpu"

# for elmo, import from the pre-trained setting
option_file = './setup/elmo_2x1024_128_2048cnn_1xhighway_options.json'
weight_file = './setup/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

# get the data
positive_train = [filename for filename in os.listdir('./train/pos') if filename != '.DS_Store']
negative_train = [filename for filename in os.listdir('./train/neg') if filename != '.DS_Store']
positive_test = [filename for filename in os.listdir('./test/pos') if filename != '.DS_Store']
negative_test = [filename for filename in os.listdir('./test/neg') if filename != '.DS_Store'] 

train_set = []
for item in positive_train:
    file_path = './train/pos/' + item
    with open(file_path, encoding='utf-8', mode='r') as f:
        content = f.read()
    content = word_tokenize(content.replace('\n', ''))
    train_set.append((1,content))

for item in negative_train:
    file_path = './train/neg/' + item
    with open(file_path,encoding='utf-8', mode='r') as f:
        content = f.read()
    content = word_tokenize(content.replace('\n', ''))
    train_set.append((0,content))

train_text = []
train_label = []
for label, text in random.sample(train_set,5000):
    train_text.append(text)
    train_label.append(label)


test_set = []
for item in positive_test:
    file_path = './test/pos/' + item
    with open(file_path, encoding='utf-8', mode='r') as f:
        content = f.read()
    content = word_tokenize(content.replace('\n', ''))
    test_set.append((1,content))

for item in negative_test:
    file_path = './test/neg/' + item
    with open(file_path, encoding='utf-8', mode='r') as f:
        content = f.read()
    content = word_tokenize(content.replace('\n', ''))
    test_set.append((1,content))

test_text = []
test_label = []
for label, text in random.sample(test_set,500):
    test_text.append(text)
    test_label.append(label)

class TextDataset(Dataset):
    def __init__(self, x_list, y_list):
        super().__init__()
        self.x = batch_to_ids(x_list)
        self.y = torch.tensor(y_list)
        self.length = torch.tensor([len(t) for t in x_list])
    
    def __getitem__(self, index):
        return ((self.x[index], self.length[index]), self.y[index])
    
    def __len__(self):
        return len(self.x)

class BiGRU(nn.Module):
    def __init__(self, option_file, weight_file, require_grad, hidden_dim, output_dim, n_layer, bidirectional, dropout):
        assert isinstance(dropout, dict), 'dropout must be dictionry.'
        super().__init__()
        self.embedding = Elmo(option_file,weight_file,1, requires_grad=require_grad,dropout=dropout['ELMo'])
        self.bidir = bidirectional
        self.rnn = nn.GRU(256,hidden_size=hidden_dim,
                          num_layers = n_layer, batch_first=True,
                          dropout=dropout['GRU'],
                          bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_dim*(2 if bidirectional else 1), 64)
        self.acti_linear = nn.ELU()
        self.out = nn.Linear(64,output_dim)
        self.acti_out = nn.Softmax(dim=1) # softmax for each "row" of the 2d tensor
        self.dropout = nn.Dropout(dropout['Fully_connected'])

    def forward(self, input_text, input_text_len):
        # calculate embedding
        elmo_rep = self.embedding(input_text)['elmo_representations'][0]
        packed_seq = nn.utils.rnn.pack_padded_sequence(elmo_rep, input_text_len, 
                                                       batch_first=True, 
                                                       enforce_sorted=False)

        # run RNN
        _, hidden = self.rnn(packed_seq)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)) if self.bidir \
                    else self.dropout(hidden[-1, :, :])
        x = self.linear(hidden)
        x = self.acti_linear(x)
        x = self.out(x)
        x = self.acti_out(x)

        return x

def binary_accuracy(preds, y):
    _, prediction = torch.max(preds,1)
    correct = (prediction == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def train(model, train_iter, optimizer, loss_function):
    total_epoch_loss, total_epoch_acc = 0,0
    model.train()
    print('Start training')
    for x_batch, y_batch in train_iter:
        optimizer.zero_grad()
        text, text_len = x_batch
        text = text.to(device)
        y_batch = y_batch.to(device)
        pred = model(text,text_len).squeeze(1)
        loss = loss_function(pred,y_batch)
        accuracy = binary_accuracy(pred, y_batch)
        loss.backward() 
        optimizer.step()

        total_epoch_loss += loss.item()
        total_epoch_acc += accuracy.item()
    print('done epoch')

    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def evaluate(model, test_set_loader, loss_function):
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = next(iter(test_set_loader))
        text, text_len = x_batch
        text = text.to(device)
        predictions = model(text, text_len).squeeze(1)
        y_batch =  y_batch.to(device)
        loss = loss_function(predictions, y_batch).item()
        acc = binary_accuracy(predictions, y_batch).item()

    return loss, acc

# initialize and train model
def main():
    model = BiGRU(option_file=option_file,weight_file=weight_file,
                        require_grad=True, hidden_dim=512,output_dim=2,
                        n_layer=2,bidirectional=True,
                        dropout={'ELMo':0.25,'GRU':0.5,'Fully_connected':0.33})
    optimizer = optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()

    model = model.to(device)
    loss_function = loss_function.to(device)

    training_dataset = TextDataset(train_text, train_label)
    testing_dataset = TextDataset(test_text,test_label)

    N_EPOCHS = 5
    opt_loss = float('inf')
    for epoch in range(N_EPOCHS):
        training_loader = DataLoader(dataset=training_dataset,batch_size=4,shuffle=True)
        testing_loader = DataLoader(dataset=testing_dataset,batch_size=250,shuffle=True)
        train_loss, train_acc = train(model, training_loader, optimizer, loss_function)
        valid_loss, valid_acc = evaluate(model, testing_loader, loss_function)

        if valid_loss < opt_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut2-model.pt')

        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f},  Val. Acc: {valid_acc*100:.2f}%')


    model.load_state_dict(torch.load('tut2-model.pt'))
    test_loss, test_acc = evaluate(model, test_data, loss_function)
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc*100:.2f}%')

if __name__ == "__main__":
    main()
