import socket
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import pickle
import threading
from tqdm.auto import tqdm
import sys
import sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#HOST = '127.0.0.1'
HOST = '192.168.1.8'
PORT = 9998
EPOCHS = 10

train_event = threading.Event()
send_data_event = threading.Event()
server_received_event = threading.Event()
weights_event = threading.Event()
cursor_lock = threading.Lock()
storing_event = threading.Event()
test_event = threading.Event()

class Client:
    def __init__(self, host_ip, host_port, id):
        self.host_ip = host_ip
        self.host_port = host_port
        self.id = id
        self.client_db = f'client_data/client_{self.id}/database_{self.id}.db'
        #self.device = self._get_default_device()
        self.device = torch.device('cpu')
        print('Device', self.device)
        self.train_losses = []
        self.val_losses = []
        self.val_acc = []
        self.test_acc = 0
        self.data_columns = {'global_loss': 'global_loss', 'aggr_model': 'model_weights', 
                       'aggr_classifier': 'classifier_weights', 'model_weights': 'model_weights'}
        self.event_dict = {b'<RECVD>': server_received_event, b'<SEND>': send_data_event, b'<TRAIN>': train_event, b'<TEST>': test_event}
        threading.Thread(target=self._create_database_schema, args=()).start()

    def _create_database_schema(self):
        query_data = """
            CREATE TABLE data(
            id INT PRIMARY KEY,
            global_loss BLOB,
            client_grads BLOB,
            model_weights BLOB,
            classifier_weights BLOB
            )
        """  
        query_means = """
            CREATE TABLE means(
            class VARCHAR(50) PRIMARY KEY,
            mean BLOB
            )
        """ 
        query_cov = """
            CREATE TABLE covariance(
            gaussian VARCHAR(50) PRIMARY KEY,
            inverse_cov_matrix BLOB
            )
        """
        query_statistics="""
            CREATE TABLE statistics(
            epoch INT PRIMARY KEY,
            training_loss REAL,
            validation_loss REAL,
            validation_acc REAL,
            testing_acc REAL,
            gamma REAL
            )
        """
        # If any tables do not exists create them
        self._execute_query(query_data, None) if not self._check_table_existence('data') else None
        self._execute_query(query_means, None) if not self._check_table_existence('means',) else None
        self._execute_query(query_cov, None) if not self._check_table_existence('covariance',) else None
        self._execute_query(query_statistics, None) if not self._check_table_existence('statistics',) else None
        threading.Thread(target=self._create_socket, args=()).start()
    
    def _check_table_existence(self, target_table: str):
        """
        Checks if a specific table exists in the db

        Args:
            table: Table to look for

        Returns:
            True or False depending on existense
        """
        query = "SELECT name FROM sqlite_master WHERE type ='table'"
        tables = self._execute_query(query=query, values=None, fetch_data_flag=True, fetch_all_flag=True)
        exists = any(table[0] == target_table for table in tables)
        return exists


    def _create_socket(self):
        """
        Creates a socket object and binds it to the specified address

        Creates:
            A thread responsible for server communication
            A thread responsible for training
        """
        try:
            serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(socket.gethostbyname(socket.gethostname()))
            serversocket.connect((self.host_ip, self.host_port))
            print(f'[+] Connection with server {self.host_ip} : {self.host_port} established')
            threading.Thread(target=self._listen_for_data, args=(serversocket,)).start()
            threading.Thread(target=self._training_procedure, args=(serversocket,)).start()
        except socket.error as error:
            print(f'[+] Socket creation failed: {error}')

    def _listen_for_data(self, serversocket: socket.socket):
        """
        Responsible for client communication.

        Communication Format:
        <MESSAGE><SEPERATOR><DATA><DONE>
            <SEPERATOR>: Constant to help us split the message from the data
            Data: Data sent from client
            <DONE>: Constant indicating end of trasmission
    
        Creates:
        A thread responsible for storing the data received from the server

        Args:
        serversocket: Server's socket
        """
        data = b''
        while True:
            data_chunk = serversocket.recv(4096)
            data += data_chunk
            data = self.check_for_message(data)
            if b'<END>' in data:
                payload = self.get_payload(data)
                threading.Thread(target=self._store_data, args=(pickle.loads(payload[0]), None, 'data', self.id, serversocket)).start() if len(payload) > 0 else None
                serversocket.send(b'<RECVD>')
                data = b''

    def check_for_message(self, data):
        keywords = [b'<RECVD>', b'<SEND>', b'<TRAIN>', b'<TEST>']
        for keyword in keywords:
            if keyword in data:
                self.event_dict[keyword].set()
                data = data.replace(keyword, b'')
        return data    
    
    def get_payload(self, data):
        payload = data.split(b'<START>')
        payload.remove(b'')
        payload = [el.split(b'<END>') for el in payload]
        payload = [element for sub in payload for element in sub]
        payload = [el for el in payload if el != b'']
        return payload

    
    def _store_data(self, data, column: str, table: str, primary_key: str = None, server: socket = None):
        
        """
        Stores data to a pickle (.pkl) file

        Args:
            data: Data to be stored
            header: keyword to use for filename
        """
        if table == 'data':
            for key, value in data.items():
                query = f"""
                    INSERT INTO {table} (id, {key}) VALUES (?, ?)
                    ON CONFLICT(id) DO
                    UPDATE SET {key} = ?"""
                self._execute_query(query=query, values=(primary_key, pickle.dumps(value), pickle.dumps(value)))
                weights_event.set() if 'weights' in key else None
        else:
            query = f"""
                INSERT INTO {table} ({primary_key}, {column}) VALUES (?, ?)
                ON CONFLICT({primary_key}) DO
                UPDATE SET {column} = ?"""
            
            self._execute_query(query=query, values=data)
        storing_event.set()
        return
    
    
    def _execute_query(self, query: str, values=None, fetch_data_flag=False, fetch_all_flag=False):
        """
        Executes given query

        Args:
            cursor: Cursor Object
            connection: Connection Object
            query: Query to be executed
            values: Query values
        """
        with cursor_lock:
            connection = sqlite3.Connection(self.client_db)
            cursor = connection.cursor()
            cursor.execute(query, values) if values is not None else cursor.execute(query)
            fetched_data = (cursor.fetchall() if fetch_all_flag else cursor.fetchone()[0]) if fetch_data_flag else None
            connection.commit()
            connection.close()        
            return fetched_data

    def _training_procedure(self, serversocket: socket.socket):
        client_model = ClientModel()
        client_classifier = ClientClassifier()
        client_model.to(device=self.device)
        client_classifier.to(device=self.device)
        storing_event.wait()
        default_model_weights = pickle.loads(self._execute_query('SELECT model_weights FROM data', None, True))
        default_classifier_weights = pickle.loads(self._execute_query('SELECT classifier_weights FROM data', None, True))
        weights_event.wait()
        weights_event.clear()
        storing_event.clear()
        print('\n[+] Recieved Default Weights\n')
        client_model.load_state_dict(default_model_weights)
        client_classifier.load_state_dict(default_classifier_weights)
        train_data, test_data = self._get_dataset(False)
        train_dl, val_dl, test_dl = self._get_dataloaders(training_data=train_data, testing_data=test_data)
        loss_fn = torch.nn.CrossEntropyLoss()
        # Optimizers specified in the torch.optim package
        model_optimizer = torch.optim.SGD(client_model.parameters(), lr=0.001, momentum=0.9)
        classifier_optimizer = torch.optim.SGD(client_classifier.parameters(), lr=0.001, momentum=0.9)
        global EPOCHS
        all_labels = self.train(EPOCHS, client_model, client_classifier, train_dl, val_dl, model_optimizer, classifier_optimizer, loss_fn, serversocket)
        self.test_acc = self.test(client_model=client_model, client_classifier=client_classifier, test_dataloader=test_dl, serversocket=serversocket)
        self._store_statistics()
        #self._fit_gaussians_to_distributions_(client_model.penultimate_feature_map, all_labels)
        sys.exit(1)

    def _send_data(self, message: str, data, serversocket: socket.socket):
        #message += '<SEPERATOR>'
        #serialized_data = struct.pack('!i', data) if isinstance(data, int) else pickle.dumps(data)
        serialized_data = data if data.__class__ is bytes else pickle.dumps(data)
        #serversocket.sendall(bytes(message, 'utf-8') + serialized_data + b'<DONE>')
        serversocket.sendall(b'<START>' + serialized_data + b'<END>')
        #time.sleep(0.0001)
        return
    
    def _get_dataset(self, cifar_flag: bool=False):
        """
        Get the training and testing data

        Returns:
            training_data
            testing_data
        """
        subset_labels = {0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]}
        #subset_labels = {1: [0, 1, 2], 0: [3, 4, 5]}
        train_indices = []
        test_indices = []
        transform = transforms.Compose([transforms.RandomRotation(degrees=(-30, 30)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        if cifar_flag:
            print('Cifar10')
            training_data = datasets.CIFAR10(root='/data', train=True, transform=transform, download=True)
            testing_data = datasets.CIFAR10(root='/data', train=False, transform=transform, download=True)
        else:
            training_data = datasets.FashionMNIST(root='/data', train=True, transform=transform, download=True)
            testing_data = datasets.FashionMNIST(root='/data', train=False, transform=transform, download=True)
        
        for i in range(len(training_data.targets)):
            if training_data.targets[i] in subset_labels[self.id]:
                train_indices.append(i)
    
        for i in range(len(testing_data.targets)):
            if testing_data.targets[i] in subset_labels[self.id]:
                test_indices.append(i)

        train_subset = torch.utils.data.Subset(training_data, train_indices)
        test_subset = torch.utils.data.Subset(testing_data, test_indices)
        print('Labels:', subset_labels[self.id])
        return train_subset, test_subset
    
    def _get_dataloaders(self, training_data: torchvision.datasets, testing_data: torchvision.datasets):
        """
        Split the training data into training and validation data.

        Returns:
            Training, validation and testing dataloaders
        """
        batch_size = 32
        val_ratio = 0.2
        training_data, validation_data = random_split(training_data, [int((1 - val_ratio) * len(training_data)), int(val_ratio * len(training_data))])
        train_dl = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
        val_dl = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
        test_dl = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)
        return train_dl, val_dl, test_dl
    
    def _fit_gaussians_to_distributions_(self, feature_maps: list, all_labels: list):
        """
        Fits the class-conditional and class-naive gaussians to client's data distributions.
        Stores the means and inverse matrices needed for Relative Mahalanobis Distance

        Args:
            feature_maps: List of model's penultimate layer features for all data.
            all_labels: List of batches and their respective labels .
    
        """
        in_distribution, all_labels, class_distributions = self._get_distributions_(all_labels, feature_maps)
        class_means = self._get_means_(class_distributions, in_distribution)
        inv_class_cov_matrix, inv_distr_cov_matrix = self._get_inverse_covariance_matrices_(class_distributions, in_distribution, class_means)
        print(f'Class Covariance Matrix: {inv_class_cov_matrix.shape} Symmetrical: {np.allclose(inv_class_cov_matrix, inv_class_cov_matrix.T)}')
        print(f'Distribution Covariance Matrix: {inv_distr_cov_matrix.shape} Symmetrical: {np.allclose(inv_distr_cov_matrix, inv_distr_cov_matrix.T)}')
        print(f"Mean Vectors shape: {class_means['global'].shape}")
        # Save on db to have em handy
        for key, value in class_means.items():
            self._store_data(data=(str(key), pickle.dumps(value), pickle.dumps(value)), column='mean', table='means', primary_key='class')
        
        self._store_data(data=('class_conditional', pickle.dumps(inv_class_cov_matrix), pickle.dumps(inv_class_cov_matrix)), column='inverse_cov_matrix', table='covariance', primary_key='gaussian')    
        self._store_data(data=('in_distribution', pickle.dumps(inv_distr_cov_matrix), pickle.dumps(inv_distr_cov_matrix)), column='inverse_cov_matrix', table='covariance', primary_key='gaussian') 
    

    def _get_distributions_(self, all_labels: list, all_feature_maps: list):
        """
        Creates the class and in-distribution matrices of the model's penultimate layer output.

        Args:
            all_labels: List of batches and their respective labels
            all_feature_maps: List of all the penultimate layer's outputs

        Returns:
            in_distribution: in-distribution data matrix of shape DATA x FEATURES
            all_labels: a list containing the labels of the in-distribution data
            class_distributions: dict where keys are unique labels and values are matrices of the class's distribution of shape DATA x FEATURES. 
        """
        # Make one vector of all labels
        all_labels = [int(item) for sublist in all_labels for item in sublist]
        # Make one matrix of the in-distribution data
        in_distribution = np.vstack([map.detach().numpy() for map in all_feature_maps])
        # Create a dict where the keys are the unique labels and the values are matrices of the class's distribution
        class_distributions = {}
        for i in range(len(all_labels)):
            if all_labels[i] not in class_distributions:
                class_distributions[all_labels[i]] = in_distribution[i]
            else:
                class_distributions[all_labels[i]] = np.vstack((class_distributions[all_labels[i]], in_distribution[i]))
        print('IN: ', in_distribution.shape)
        [print(f"{key} -> {value.shape}")for key, value in class_distributions.items()]
        return in_distribution, all_labels, class_distributions

    def _get_means_(self, class_distributions, in_distribution):
        """
        Calculates the in-distribution's and class distributions' means for each feature.

        Args:
            class_distributions: Dictionary of (class, matrix) pairs. Each matrix has a shape of FEATURES x CLASS DATA.
            in_distribution: Matrix of the in-distribution data of shape FEATURES x DATA.
        
        Returns:
            class_means: Dictionary containing (class, mean) pairs. Each mean is a vector of length FEATURES.
        """
        class_means = {}
        # For each class distribution calculate the mean.
        for key, value in class_distributions.items():
            class_means[key] = np.mean(value, axis=0)
        distr_mean = np.mean(in_distribution, axis=0)
        print('eeeee', distr_mean.shape)
        class_means['global'] = distr_mean
        #[print(f'Key: {key} ->', class_means[key].shape) for key in class_means.keys()]
        return class_means

    def _get_inverse_covariance_matrices_(self, class_distributions, in_distribution, means):
        """
        Calculates the covariance matrices Σ and Σ(0).
            Σ = (1/N) * Σ Σ (z(i) - μk)
            Σ(0) = (1/N) * Σ (z(i) - μ0) * (z(i) - μ0)^T

        Args:
            class_distributions: Dictionary of (class, matrix) pairs. Each matrix has a shape of FEATURES x CLASS DATA.
            in_distribution: Matrix of the in-distribution data of shape FEATURES x DATA.

        Returns:
            class_cov_matrix: Inverse of covariance matrix used for the class gaussians with shape FEATURES x FEATURES.
            distr_cov_matrix: Inverse of covariance matrix for the in-distribution data, class-naive gaussian with shape FEATURES x FEATURES.
        """
        class_matrices = {}
        for key, value in class_distributions.items():
            diff = value - means[key]
            print(f'diff : {diff.shape}')
            matrix = diff.T @ diff
            class_matrices[key] = matrix
            print(f'Matrix: {matrix.shape}')

        cov_matrix = np.zeros((84, 84))
        for key, value in class_matrices.items():
            cov_matrix += value 

        N = in_distribution.shape[0]
        class_cov_matrix = (1 / N) * cov_matrix 
        distr_cov_matrix = (1 / N) * ((in_distribution - means['global']).T @ (in_distribution - means['global']))
        # Regularize the diagonal elements to avoid singular matrix errors
        alpha = 1e-6
        class_cov_matrix = class_cov_matrix + alpha * np.eye(class_cov_matrix.shape[0])
        distr_cov_matrix = distr_cov_matrix + alpha * np.eye(distr_cov_matrix.shape[0])
        return np.linalg.inv(class_cov_matrix), np.linalg.inv(distr_cov_matrix)

    def _mahalanobis_distance():
        pass

    def _get_default_device(self):
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def train(self, epochs: int, client_model: nn.Module, client_classifier: nn.Module, train_dataloader: DataLoader, valid_dataloader: DataLoader, model_optimizer:torch.optim, classifier_optimizer:torch.optim, criterion: torch.nn, serversocket: socket.socket):
        best_vloss = 1_000_000.
        for epoch in tqdm(range(epochs), desc="Epoch"):
            print('\n[+] Waiting for Server')
            train_event.wait()
            train_event.clear()
            print('\n[+] Training Initiated')
            self._send_data('', {'training_batches': len(train_dataloader)}, serversocket)
            #server_received_event.wait()
            #server_received_event.clear()
            avg_loss, avg_gloss, all_labels = self.train_one_epoch(epoch, train_dataloader, model_optimizer, classifier_optimizer, criterion, client_model, client_classifier, serversocket)
            self.train_losses.append(avg_loss)
            avg_vloss, accuracy = self._validate(client_model, client_classifier, valid_dataloader, criterion, serversocket)
            self.val_losses.append(avg_vloss)
            self.val_acc.append(accuracy)
            print(f'\nAverage Training Loss: {avg_loss: .3f}')
            print(f'Average Global Loss: {avg_gloss: .3f}')
            print(f'Average Validation Loss: {avg_vloss: .3f}')
            print(f'Validation Accuracy: {accuracy: .2f} %')

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
            print(f'\n[+] Waiting for aggregated weights')
            weights_event.wait()
            weights_event.clear()
            storing_event.wait()
            query = 'SELECT model_weights FROM data'
            client_model.load_state_dict(pickle.loads(self._execute_query(query, None, True)))
            query = 'SELECT classifier_weights FROM data'
            client_classifier.load_state_dict(pickle.loads(self._execute_query(query, None, True)))
            print('[+] Received Aggregated Weights\n')
            storing_event.clear()
        return all_labels

    def train_one_epoch(self, epoch: int, train_dl: DataLoader, model_optimizer: torch.optim, classifier_optimizer:torch.optim, criterion: torch.nn, client_model: nn.Module, client_classifier: nn.Module, serversocket: socket.socket):
        client_model.train()
        client_classifier.train()
        running_loss = 0.
        running_global_loss = 0
        all_labels = []
        for i, data in enumerate(train_dl):
            inputs, labels = data
            inputs, labels = inputs.to(device=self.device), labels.to(device=self.device)
            # Clear gradient accumulation
            model_optimizer.zero_grad()
            classifier_optimizer.zero_grad()

            # If last epoch save penultimate layer's feature maps and get all labels
            #outputs1 = client_model(inputs) if epoch != EPOCHS - 1 else client_model(inputs, True)
            outputs1 = client_model(inputs)
            all_labels.append(labels) if epoch != EPOCHS - 1 else None
            outputs2 = client_classifier(outputs1)
            # Compute the client side loss
            client_loss = criterion(outputs2, labels)
            #print(f'{inputs.device}, {labels.device}')
            #data_packet = {'training_outputs': outputs1.requires_grad_(True), 'training_labels': labels, 'training_loss': client_loss}
            data_packet = {'training_outputs': outputs1, 'training_labels': labels, 'training_loss': client_loss}
            send_data_event.wait()
            self._send_data(message='', data=data_packet, serversocket=serversocket)
            send_data_event.clear()
            storing_event.wait()
            query = 'SELECT global_loss FROM data'
            server_loss = pickle.loads(self._execute_query(query, None, True))
            query = 'SELECT client_grads FROM data'
            client_grads = pickle.loads(self._execute_query(query, None, True))
            storing_event.clear()
            global_loss = (0.5 * client_loss) + (0.5 * server_loss)
            global_loss.backward()
            classifier_optimizer.step()
            model_optimizer.step()
            # Gather data and report
            running_loss += client_loss.item()
            running_global_loss += global_loss.item()
            del inputs, labels
            torch.cuda.empty_cache()
        data_packet = {'model_updated_weights': client_model.state_dict(), 'classifier_updated_weights': client_classifier.state_dict()}
        self._send_data(message='', data=data_packet, serversocket=serversocket)
        #print('Sent train pack, waiting for recvd')
        #server_received_event.wait()
        #server_received_event.clear()
        return (running_loss / len(train_dl)), (running_global_loss / len(train_dl)), all_labels
    
    def _validate(self, client_model, client_classifier, valid_dataloader, criterion, serversocket):
        # We don't need gradients on to do reporting
        client_model.eval()
        client_classifier.eval()
        total = 0
        correct = 0
        running_vloss = 0.0
        send_data_event.wait()
        self._send_data('',{'validation_batches': len(valid_dataloader)}, serversocket)
        send_data_event.clear()
        print('\n[+] Validating with Server')
        for i, vdata in enumerate(valid_dataloader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device=self.device), vlabels.to(device=self.device)
            voutputs1 = client_model(vinputs)
            data_packet = {'validation_outputs': voutputs1, 'validation_labels': vlabels}
            send_data_event.wait()
            self._send_data('', data=data_packet, serversocket=serversocket)
            send_data_event.clear()
            voutputs2 = client_classifier(voutputs1)
            _, predicted = torch.max(voutputs2.data, 1)
            total += vlabels.size(0)
            correct += (predicted == vlabels).sum().item()
            vloss = criterion(voutputs2, vlabels)
            running_vloss += vloss.item()
            del vinputs, vlabels
            torch.cuda.empty_cache()
        avg_vloss = running_vloss / len(valid_dataloader)
        accuracy = 100 * correct / total
        return avg_vloss, accuracy
    
    def test(self, client_model: nn.Module, client_classifier: nn.Module, test_dataloader: DataLoader, serversocket: socket.socket):
        client_model.eval()
        client_classifier.eval()
        total = 0
        correct = 0
        print('\n[+] Waiting For Server.')
        test_event.wait()
        test_event.clear()
        print('\n[+] Testing Initiated With Server.')
        self._send_data('', data={'testing_batches': len(test_dataloader)}, serversocket=serversocket)
        with torch.inference_mode():
            for i, data in enumerate(test_dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                model_outputs = client_model(inputs)
                class_outputs = client_classifier(model_outputs)
                data_packet = {'testing_outputs': model_outputs, 'testing_labels': labels}
                send_data_event.wait()
                self._send_data('', data=data_packet, serversocket=serversocket)
                send_data_event.clear()
                _, predictions = torch.max(class_outputs, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
                del model_outputs, class_outputs
                torch.cuda.empty_cache()
        print(f'[+] Test Accuracy: {correct / total : .2%}')
        return 100 * (correct / total)

    def _store_statistics(self):
        query = f"""
                INSERT INTO statistics (epoch, training_loss, validation_loss, validation_acc) VALUES (?, ?, ?, ?)
                ON CONFLICT(epoch) DO UPDATE SET training_loss = ?, validation_loss = ?, validation_acc = ?
            """
        for idx in range(len(self.train_losses)):
            self._execute_query(query=query, values=(idx, self.train_losses[idx], self.val_losses[idx], self.val_acc[idx], self.train_losses[idx], self.val_losses[idx], self.val_acc[idx]), fetch_data_flag=False)
        
        query = 'UPDATE statistics SET testing_acc = ? WHERE epoch = ?'
        self._execute_query(query=query, values=(self.test_acc, EPOCHS - 1))
        print('[+] Successfully stored data in the database.')
        #self.produce_graph(self.train_losses, 'Epochs', 'Training loss', 'Learning Curve', f'client_data/client_{self.id}/figures/train_loss')
        #self.produce_graph(self.val_losses, 'Epochs', 'Validation loss', 'Evaluation Curve', f'client_data/client_{self.id}/figures/val_loss')
        #self.produce_graph(self.val_acc, 'Epochs', 'Accuracy Percentage', 'Validation Accuracy', f'client_data/client_{self.id}/figures/val_acc')

    def produce_graph(self, y, xlab, ylab, title, path):
        x_ticks = list(range(len(y)))
        fig, ax = plt.subplots()
        ax.plot(x_ticks, y)
        ax.set_xlabel(xlab)
        ax.set_xticklabels(x_ticks)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        plt.savefig(path)
        return

# FMNIST MODELS
class ClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))
        return x

class ClientClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=4*4*256, out_features=10)
        
    def forward(self, x):
        x = self.fc1(torch.flatten(x, 1))
        return x
"""
# CIFAR 10 MODELS

class ClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))
        return x

class ClientClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=5*5*256, out_features=10)
        
    def forward(self, x):
        x = self.fc1(torch.flatten(x, 1))
        return x

class ServerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3)
        self.fc1 = nn.Linear(3*3*512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv5(x))
        x = self.fc1(torch.flatten(x, 1))
        x = self.fc2(x)
        x = self.fc3(x)
        return x          
"""

if __name__ == '__main__':
    arg = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    Client(HOST, PORT, arg)