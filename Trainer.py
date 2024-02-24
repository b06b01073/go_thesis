import torch
from tqdm import tqdm


from config.training_config import training_config
import matplotlib.pyplot as plt 


from seed import set_seed
set_seed()


class Trainer:
    def __init__(self, net, optim, loss_func, file_path):
        '''
            Args:
                net (nn.Module): the neural network to be trained
                optim (nn.Module): optimizer
                loss_func (nn.Module): loss function
                file_path (str): the path to save the model
        '''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Training on {self.device}')

        self.net = net.to(self.device)
        self.optim = optim
        self.loss_func = loss_func
        self.file_path = file_path


    def fit(self, train_set, test_set):
        '''
            Fits the model on `train_set` and saves the best performing model on `test_set`

            Args:
                
                train_set (DataLoader)

                test_set (DataLoader)
        '''
        best_acc = 0
        
        top_1_train_accs = []
        top_5_train_accs = []
        
        top_1_test_accs = []
        top_5_test_accs = []


        for e in range(training_config['num_epochs']):
            top_1_train, top_5_train = self.train(train_set) 
            top_1_test, top_5_test = self.test(test_set)
            
            self.log_accs(
                e,
                top_1_train,
                top_5_train,
                top_1_test,
                top_5_test,
            )

            if top_1_test > best_acc:
                best_acc = top_1_test
                torch.save(self.net, self.file_path)
                print(f'model saved, acc = {best_acc:.4f}')

            
            top_1_train_accs.append(top_1_train)
            top_5_train_accs.append(top_5_train)
            top_1_test_accs.append(top_1_test)
            top_5_test_accs.append(top_5_test)


        self.plot(top_1_train_accs, top_5_train_accs, top_1_test_accs, top_5_test_accs)


        
    def plot(self, top_1_train_accs, top_5_train_accs, top_1_test_accs, top_5_test_accs):
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        plt.ylim(0, 1) # accuracy must be in [0, 1]

        plt.plot(top_1_train_accs, label='Top 1 Training')
        plt.plot(top_5_train_accs, label='Top 5 Training')
        plt.plot(top_1_test_accs, label='Top 1 Testing')
        plt.plot(top_5_test_accs, label='Top 5 Testing')

        plt.legend()

        plt.savefig('result.png')


    def log_accs(self, e, top_1_train, top_5_train, top_1_test, top_5_test):
        print(f'Epoch: {e}')
        print(f'Training Acc: top1: {top_1_train:.4f}, top5: {top_5_train:.4f}')
        print(f'Testing Acc: top1: {top_1_test:.4f}, top5: {top_5_test:.4f}')
        


    def train(self, train_set):
        '''
            Args:
                train_set (DataLoader): training set
                log_interval (float): the accuracy will be logged at progress bar for every `len(train_set) * log_interval` iterations

            Returns: 
                top 1 accuracy (float) and top 5 accuracy (float)
        '''

        self.net.train()

        correct_preds = 0
        total_preds = 0
        top5_hit = 0 
            

        pbar = tqdm(enumerate(train_set), total=len(train_set), dynamic_ncols=True)
        for iter, (states, target) in pbar:
            states = states.squeeze(dim=0)
            target = target.squeeze(dim=0)

            states = states.to(self.device)
            target = target.to(self.device)

    
            preds = self.net(states) 

            self.optim.zero_grad()

            loss = self.loss_func(preds, target)
            loss.backward()
            self.optim.step()

            predicted_classes = torch.argmax(torch.softmax(preds, dim=1), dim=1)
            # Compare the predicted classes to the target labels
            correct_preds += torch.sum(predicted_classes == target).item()
            top5_hit += self.batch_topk_hit(preds, target)

            total_preds += target.shape[0]

            if iter % 100 == 0 and iter != 0:
                top1_accuracy = correct_preds / total_preds
                top5_accuracy = top5_hit / total_preds
                pbar.set_description(f'Training Acc: top1: {top1_accuracy:.4f}, top5: {top5_accuracy:.4f}')

        return correct_preds / total_preds, top5_hit / total_preds
    

    def test(self, test_set):
        self.net.eval()
        correct_preds = 0
        total_preds = 0
        top5_hit = 0
        
        pbar = tqdm(test_set, total=len(test_set), dynamic_ncols=True)

        with torch.no_grad():
            for states, target in pbar:
                states = states.squeeze(dim=0)
                target = target.squeeze(dim=0)

                states = states.to(self.device)
                target = target.to(self.device)

                preds = self.net(states) 

                predicted_classes = torch.argmax(torch.softmax(preds, dim=1), dim=1)
                # Compare the predicted classes to the target labels
                correct_preds += torch.sum(predicted_classes == target).item()
                top5_hit += self.batch_topk_hit(preds, target)

                total_preds += target.shape[0]

                pbar.set_description(f'Testing Acc: top1: {correct_preds / total_preds:.4f}, top5: {top5_hit / total_preds:.4f}')

        pbar.close()

        return correct_preds / total_preds, top5_hit / total_preds



    def batch_topk_hit(self, preds, label_index, k=5):
        preds = torch.softmax(preds, dim=1)
        _, topk_indices = preds.topk(k, dim=-1) # output (batch, k)

        # Check if the true label_index is in the top-k predicted labels for each example
        batch_size, pred_size = preds.shape

        correct = 0

        for i in range(batch_size):
            if label_index[i] in topk_indices[i]:
                correct += 1

        return correct