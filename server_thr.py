import socket
import pickle
import threading
import torch
from torch import nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm
import sys
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Events to enasure data is saved in the database before trying to access it
# Received and stored data from client
training_packet_event = threading.Event()
validation_packet_event = threading.Event()
testing_packet_event = threading.Event()
# Weight for client model and classifier after each epoch
updated_weights_event = threading.Event()
aggregation_event = threading.Event()
storing_event = threading.Event()
testing_lock = threading.Lock()
aggr_lock = threading.Lock()
# λ: Controls Generalization / Personalization 
GAMMA = 0.5
LAMBDA = 0
EPOCHS = 10
# Flags to ensure we compute the weighted averages and store the server model only once per global round
IS_WEIGHTED_AVERAGE_COMPUTED = False
IS_SERVER_AGGR_SAVED = False
client_recvd_event = threading.Event()
training_event = threading.Event()
training_lock = threading.Lock()
cursor_lock = threading.Lock()
send_data_lock = threading.Lock()

class Server:
    def __init__(self, server_id: int):
        """
        Initializes the Server class.

        Args:
            server_id: Server's ID on the db

        Creates:
            A thread that creates the server socket. (create_socket()) 
            ip: Server's ip
            port: Server's port
            fetching_connection: Conntection to the db used to fetch data
            storing_connection: Conntection to the db used to store data
            client_trained_counter: Keep track of the number of clients trained during one global round
            client_ids: List containing the ids of the currently connected clients
            server_id: Server's is in the database
        """
        #self.ip = 'localhost'
        self.ip = '192.168.1.8'
        self.port = 9998
        self.server_id = server_id
        self.server_db = f'server_data/server_{self.server_id}_database.db'
        self.device = self._get_default_device()
        print('Device', self.device)
        training_event.set()
        self.client_trained_counter = 0
        self.client_ids = []
        torch.manual_seed(32)
        self.client_model = ClientModel()
        torch.manual_seed(32)
        self.client_classifier = ClientClassifier()
        torch.manual_seed(32)
        self.server_model = ServerModel()
        self.server_model.load_state_dict(torch.load('New_Base3/Sixth10/server_weights.pt'))
        torch.manual_seed(32)
        self.server_model.to(device=self.device)
        threading.Thread(target=self._create_database_schema, args=()).start()
        threading.Thread(target=self.create_socket, args=(())).start()
        

    def _create_database_schema(self):
        query_clients = """
        CREATE TABLE clients(
            id INT PRIMARY KEY,
            ip VARCHAR(50),
            port INT
        )
        """
        query_training = """
            CREATE TABLE training(
                client_id INT PRIMARY KEY,
                training_batches INT,
                training_outputs BLOB,
                training_labels BLOB,
                training_loss BLOB,
                FOREIGN KEY (client_id) REFERENCES clients (id)
            )
        """
        query_weights = """
            CREATE TABLE weights(
                client_id INT PRIMARY KEY,
                model_aggregated_weights BLOB,
                model_updated_weights BLOB,
                classifier_updated_weights BLOB,
                classifier_aggregated_weights BLOB,
                server_updated_weights BLOB,
                FOREIGN KEY (client_id) REFERENCES clients (id)
            )
        """
        query_validation = """
            CREATE TABLE validation(
                client_id INT PRIMARY KEY,
                validation_batches INT,
                validation_outputs BLOB,
                validation_labels BLOB,
                FOREIGN KEY (client_id) REFERENCES clients (id)
            )
        """
        query_testing = """
            CREATE TABLE testing(
            client_id INT PRIMARY KEY,
            testing_batches INT,
            testing_outputs BLOB,
            testing_labels BLOB,
            FOREIGN KEY (client_id) REFERENCES clients (id)
            )
        """

        query_servers = """
            CREATE TABLE servers(
                id INT AUTO_INCREMENT PRIMARY KEY,
                ip VARCHAR(50),
                port INT,
                aggregated_weights BLOB
            )
        """
        query_statistics = """
            CREATE TABLE statistics(
                client_id INT,
                epoch INT,
                server_training_loss REAL,
                global_loss REAL,
                server_validation_loss REAL,
                server_validation_acc REAL,
                server_test_acc REAL,
                PRIMARY KEY (client_id, epoch),
                FOREIGN KEY (client_id) REFERENCES clients(id)
            )
        """
        # If any tables don't exist create them
        self._execute_query(query_clients, None) if not self._check_table_existence('clients') else None
        self._execute_query(query_training, None) if not self._check_table_existence('training',) else None
        self._execute_query(query_weights, None) if not self._check_table_existence('weights',) else None
        self._execute_query(query_validation, None) if not self._check_table_existence('validation',) else None
        self._execute_query(query_testing, None) if not self._check_table_existence('testing',) else None
        self._execute_query(query_servers, None) if not self._check_table_existence('servers',) else None
        self._execute_query(query_statistics, None) if not self._check_table_existence('statistics',) else None
        query = f"""
            INSERT INTO servers (id, ip, port, aggregated_weights) VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET ip = ?, port = ?, aggregated_weights = ?
        """
        self._execute_query(query=query, values=(self.server_id, self.ip, self.port, pickle.dumps(self.server_model.state_dict()), self.ip, self.port, pickle.dumps(self.server_model.state_dict())))
        return

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
            connection = sqlite3.Connection(self.server_db)
            cursor = connection.cursor()
            cursor.execute(query, values) if values is not None else cursor.execute(query)
            fetched_data = (cursor.fetchall() if fetch_all_flag else cursor.fetchone()[0]) if fetch_data_flag else None
            connection.commit()
            connection.close()        
            return fetched_data

    def create_socket(self):
        """
        Creates a socket object and binds it to the specified address

        Creates:
            A thread responsible for accepting incoming connections from clients
        """
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(f'\n[+] Server: {self.server_id} initialized successfully at {self.ip, self.port}')
            server.bind((self.ip, self.port))
            server.listen()
            listen_thread = threading.Thread(target=self.listen_for_connections, args=(server,))
            listen_thread.start()
        except socket.error as error:
            print(f'Socket creation failed with error: {error}')
            server.close()


    def send_data(self, data, socket: socket):
        """
            Pickles and sends data to the specified client
            Args:
                message: Keyword
                socket: Client's socket
                data: Data to be sent
        """
        with send_data_lock:
            #message += '<SEPERATOR>'
            serialized_data = pickle.dumps(data) if data.__class__ is not bytes else data
            #packet = bytes(message, 'utf-8') + serialized_data + b'<DONE>'
            socket.sendall(b'<START>' + serialized_data + b'<END>')

    def _get_default_device(self):
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    
    def listen_for_connections(self, server: socket.socket):
        """
        Thread responsible for accepting incoming connections from clients.
        Checks if a client with this address already exists in the db, if not creates a new record for the client 
        Creates:
            A thread responsible for client communication (listen_for_data())
            A thread for client handling (client_handler())
        Args:
            server: Server's socket
        """
        while True:
            client_socket, client_address = server.accept()
            threading.Thread(target=self._store_client, args=(client_address, client_socket)).start()

    def _store_client(self, client_address, client_socket):
        query = 'SELECT id FROM clients ORDER BY id DESC LIMIT 1'
        last_id = self._execute_query(query, fetch_data_flag=True, fetch_all_flag=True)
        # Check if clients table is empty to either assign the first id or increment by 1
        client_id = 1 if len(last_id) == 0 else last_id[0][0] + 1
        query = f"""
                INSERT INTO clients (id, ip, port) VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET ip = ?, port = ?
            """
        self._execute_query(query=query, values=(client_id, client_address[0], client_address[1], client_address[0], client_address[1]))
        self.client_ids.append(client_id)
        print(f'\n[+] Connection with {client_id, client_address} established.\n[+] Connected Clients: {len(self.client_ids)}')
        communication_thread = threading.Thread(target=self._listen_for_data, args=(client_socket, client_id))
        communication_thread.start()
        client_thread = threading.Thread(target=self.client_handler, args=(client_socket, client_address, client_id))
        client_thread.start()


    def _listen_for_data(self, client_socket:socket.socket, client_id: int):
        """
        Responsible for client communication.

        Communication Format:
        <MESSAGE><SEPERATOR><DATA><DONE>
            Message: Indicative of the context of the data
            <SEPERATOR>: Constant to help us split the message from the data
            Data: Data sent from client
            <DONE>: Constant indicating end of trasmission
    
        Creates:
        A thread responsible for storing the data received from clients to the database

        Args:
        client_socket: Client's socket
        client_address: Client's address
        client_id: Client's ID from database
        """
        try:
            data = b''
            while True:
                data_chunk = client_socket.recv(4096)
                data += data_chunk
                if b'<END>' in data:
                    threading.Thread(target=self._get_payload, args=(data, client_socket, client_id)).start()
                    data = b''
        except ConnectionResetError:
                client_socket.close()

        
    def _get_payload(self, data, client_socket, client_id):
        if b'<START>' in data:
            data = data.split(b'<START>')[1]
        if b'<END>' in data:
            data = data.split(b'<END>')[0]
            threading.Thread(target=self.store_data, args=(pickle.loads(data), client_id)).start()
            client_socket.send(b'<RECVD>')
            return
    
    def store_data(self, data: dict, client_id: int):
        events = {'training': training_packet_event, 'validation': validation_packet_event, 'weights': updated_weights_event, 'testing': testing_packet_event}
        tables = ['training', 'validation', 'testing', 'weights']
        table = next((t for t in tables if any(t in key for key in data.keys())), None)
        for key, value in data.items():
            query = f"""
                INSERT INTO {table} (client_id, {key}) VALUES (?, ?)
                ON CONFLICT(client_id) DO
                UPDATE SET {key} = ?"""
            self._execute_query(query, (client_id, pickle.dumps(value), pickle.dumps(value))) if value.__class__ is not int else self._execute_query(query, (client_id, value, value)) 
        events[table].set() if table in events.keys() else None
        storing_event.set()
        return
    

    def client_handler(self, client_socket: socket.socket, client_address: tuple[str, int], client_id: int):
        """
        Send default weights and start training procedure.
        """
        #data_packet = {'model_weights': self.client_model.state_dict(), 'classifier_weights': self.client_classifier.state_dict()}
        if client_id == 1:
            data_packet = {'model_weights': torch.load('New_Base3/Sixth10/client1_model_weights.pt'), 'classifier_weights': torch.load('New_Base3/Sixth10/client1_classifier_weights.pt')}
        elif client_id == 2:
            data_packet = {'model_weights': torch.load('New_Base3/Sixth10/client2_model_weights.pt'), 'classifier_weights': torch.load('New_Base3/Sixth10/client2_classifier_weights.pt')}
        self.send_data(data=data_packet, socket=client_socket)
        self.store_data(data={'model_aggregated_weights': (pickle.dumps(self.client_model.state_dict()))}, client_id=client_id)
        self.store_data(data={'classifier_aggregated_weights': (pickle.dumps(self.client_classifier.state_dict()))}, client_id=client_id)
        # Define loss function and optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.server_model.parameters(), lr=0.001, momentum=0.9)
        # Start Training
        train_losses, val_losses, val_accs, global_losses = self.train(EPOCHS, self.server_model, optimizer, loss_fn, client_socket, client_address, client_id)
        test_acc = self.test(self.server_model, client_socket, client_id, client_address)
        self._store_statistics(client_id, global_losses, train_losses, val_losses, val_accs, 0)
        #client_socket.close()
    
    
    def train(self, epochs: int, server_model: nn.Module, optimizer: torch.optim, criterion: torch.nn, clientsocket: socket.socket, client_address: tuple[str, int], client_id: int):
        """
        Training procedure.

        Args:
            epochs: Global Rounds
            server_model: Instance of server's model
            optimizer: Optimizer
            loss_fn: Loss Function
            client_socket: Client's socket
            client_address: Client's address
            client_id: Client's id

        """
        best_vloss = 1_000_000.
        global_losses = []
        train_losses = []
        val_losses = []
        val_accs = []
        for epoch in range(0, epochs):
            aggregation_event.clear()
            global IS_WEIGHTED_AVERAGE_COMPUTED
            global IS_SERVER_AGGR_SAVED
            IS_WEIGHTED_AVERAGE_COMPUTED = False
            IS_SERVER_AGGR_SAVED = False
            # Acquire number of client's training batches
            # Start training with a client at a time
            with training_lock:
                server_model.train()
                clientsocket.send(b'<TRAIN>')
                print(f'\n[+] Training with {client_id, client_address} for global round {epoch + 1} intiated.')
                training_packet_event.wait()
                training_packet_event.clear()
                query = 'SELECT training_batches FROM training WHERE client_id = ?'
                client_batches = self._execute_query(query=query, values=(client_id,), fetch_data_flag=True)
                avg_global_loss, avg_server_loss = self.train_one_epoch(optimizer, criterion, server_model, clientsocket, client_batches, client_id)
                # Receive client's updated weights
                updated_weights_event.wait()
                updated_weights_event.clear()
                # Validate
                server_model.eval()
                avg_vloss, accuracy = self._validate(client_id, clientsocket, server_model, criterion)
                global_losses.append(avg_global_loss)
                train_losses.append(avg_server_loss)
                val_losses.append(avg_vloss)
                val_accs.append(accuracy)
            # Track number of trained clients   
            self.client_trained_counter += 1
            print(f'\n[+] Global round {epoch + 1} Client: {client_id, client_address}\n   Average Global Training Loss: {avg_global_loss: .3f}\n   Average Server Training Loss: {avg_server_loss: .3f}\n   Average Server Validation Loss: {avg_vloss: .3f}\n   Validation Accuracy: {accuracy: .2f} %\n   Remaining Clients: {len(self.client_ids) - self.client_trained_counter}')
            # Update record of server's training weights with specific client
            query = 'UPDATE weights SET server_updated_weights = ? WHERE client_id = ?'
            self._execute_query(query, (pickle.dumps(server_model.state_dict()), client_id))
            # Aggregate models when you have trained with all client for one global round
            aggregation_event.set() if self.client_trained_counter == len(self.client_ids) else None
            aggregation_event.wait()
            self._aggregate_models(client_id, clientsocket)
            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
        return train_losses, val_losses, val_accs, global_losses 
        

    def train_one_epoch(self, optimizer: torch.optim, criterion: torch.nn, servermodel: nn.Module, clientsocket: socket.socket, client_batches: int, client_id: int):
        running_server_loss = 0.
        running_global_loss = 0.
        for i in range(client_batches):
            global GAMMA
            #print('Batch', i)
            optimizer.zero_grad()
            clientsocket.send(bytes('<SEND>', 'utf-8'))
            #print('Waiting on training pack')
            training_packet_event.wait()
            training_packet_event.clear()
            storing_event.wait()

            query = 'SELECT training_outputs FROM training WHERE client_id = ?'
            client_outputs = pickle.loads(self._execute_query(query=query, values=(client_id,), fetch_data_flag=True))
            client_outputs = client_outputs.to(device=self.device)
            outputs2 = servermodel(client_outputs)

            query = 'SELECT training_labels FROM training WHERE client_id = ?'
            client_labels = pickle.loads(self._execute_query(query=query, values=(client_id,), fetch_data_flag=True))
            client_labels = client_labels.to(device=self.device)
            loss = criterion(outputs2, client_labels)
            query = 'SELECT training_loss FROM training WHERE client_id = ?'
            client_loss = pickle.loads(self._execute_query(query=query, values=(client_id,), fetch_data_flag=True))
            storing_event.clear()
            global_loss = (GAMMA * client_loss) + (GAMMA * loss)
            #global_loss.backward()
            #data_packet = {'client_grads': client_outputs.grad.clone().detach(), 'global_loss': global_loss}
            data_packet = {'client_grads': '', 'global_loss': loss.to(torch.device('cpu'))}
            
            global_loss.backward()
            #data_packet = {'client_grads': client_outputs.grad.clone(), 'global_loss': loss}
            self.send_data(data_packet, clientsocket)
            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_server_loss += loss.item()
            running_global_loss += global_loss.item()
            del client_outputs, client_labels
            torch.cuda.empty_cache()

        return (running_global_loss / client_batches), (running_server_loss / client_batches) 

    def _validate(self, client_id: int, clientsocket:socket.socket, server_model: nn.Module, criterion: torch.nn):
        """
        Validation procedure.
        """
        # Get number of validation batches from client
        storing_event.wait()
        clientsocket.send(b'<SEND>')
        storing_event.clear()
        validation_packet_event.wait()
        validation_packet_event.clear()
        query = 'SELECT validation_batches FROM validation WHERE client_id = ?'
        client_batches = self._execute_query(query=query, values=(client_id,), fetch_data_flag=True)
        running_vloss = 0.0
        total = 0
        correct = 0
        print(f'\n[+] Validating with client {client_id}')
        # Iterate the batches
        for batch in range(client_batches):
            clientsocket.send(b'<SEND>')
            # Acquire client's outputs
            validation_packet_event.wait()
            query = 'SELECT validation_outputs FROM validation WHERE client_id = ?'
            client_outputs = pickle.loads(self._execute_query(query=query, values=(client_id,), fetch_data_flag=True))
            # Acquire client's labels
            query = 'SELECT validation_labels FROM validation WHERE client_id = ?'
            validation_labels = pickle.loads(self._execute_query(query=query, values=(client_id,), fetch_data_flag=True))
            validation_packet_event.clear()
            client_outputs, validation_labels = client_outputs.to(device=self.device), validation_labels.to(device=self.device)
            # Foward pass + Calculate Loss
            voutputs = server_model(client_outputs)
            vloss = criterion(voutputs, validation_labels)
            running_vloss += vloss.item()
            _, predicted = torch.max(voutputs.data, 1)
            total += validation_labels.size(0)
            correct+= (predicted == validation_labels).sum().item()
            del client_outputs, validation_labels
            torch.cuda.empty_cache()
        avg_vloss = running_vloss / (batch + 1)
        accuracy = 100 * (correct / total)
        return avg_vloss, accuracy
    
    def _aggregate_models(self, client_id: int, clientsocket: socket.socket):
        """
        Takes care of aggregating the models. Calculates the weighted averages once using a flag to save on time and resourses.
        """
        global IS_WEIGHTED_AVERAGE_COMPUTED
        global IS_SERVER_AGGR_SAVED
        self.client_trained_counter = 0
        # Calculate weighted averages only once per global round
        all_weighted_averages, IS_WEIGHTED_AVERAGE_COMPUTED= (list(self._calculate_weighted_averages()), True) if not IS_WEIGHTED_AVERAGE_COMPUTED else (all_weighted_averages, False)
        # Aggregate for specific client and send weights
        aggregated_weights_model, aggregated_weights_classifier = self.aggregate_client_models(client_id, all_weighted_averages[0], all_weighted_averages[1])
        #self.send_data('<AGGR_MODEL>', clientsocket, aggregated_weights_model)
        data_packet = {'model_weights': aggregated_weights_model, 'classifier_weights': aggregated_weights_classifier}
        self.send_data(data_packet, clientsocket)
        # Store server aggregated weights on db once per global round
        if not IS_SERVER_AGGR_SAVED:
            IS_SERVER_AGGR_SAVED = True
            # Store server weights
            query = f"""
            INSERT INTO servers (id, aggregated_weights) VALUES (?, ?)
            ON CONFLICT(id) DO
            UPDATE SET aggregated_weights = ?
        """
            self._execute_query(query, (self.server_id, pickle.dumps(all_weighted_averages[-1]), pickle.dumps(all_weighted_averages[-1])))
            self.server_model.load_state_dict(state_dict=all_weighted_averages[-1])
            print('[+] Aggregated Server Weights Loaded and Stored')
        return
            
    def _calculate_weighted_averages(self):
        # Retrieve all client-server updated weights for current epoch
        query = "SELECT model_updated_weights FROM weights WHERE client_id IN (" + ", ".join(str(id) for id in self.client_ids) + ")"
        all_client_model_weights = self._execute_query(query=query, values=None, fetch_data_flag=True, fetch_all_flag=True)

        query = "SELECT classifier_updated_weights FROM weights WHERE client_id IN (" + ", ".join(str(id) for id in self.client_ids) + ")"
        all_client_classifier_weights = self._execute_query(query=query, values=None, fetch_data_flag=True, fetch_all_flag=True)

        
        query = "SELECT server_updated_weights FROM weights WHERE client_id IN (" + ", ".join(str(id) for id in self.client_ids) + ")"
        all_server_model_weights = self._execute_query(query=query, values=None, fetch_data_flag=True, fetch_all_flag=True)

        # Get the each client's datasize and convert to integer dtype to calculate total data size                    
        query = "SELECT training_batches FROM training WHERE client_id IN (" + ", ".join(str(id) for id in self.client_ids) + ")"
        datasizes = self._execute_query(query, values=None, fetch_data_flag=True, fetch_all_flag=True)
        datasizes = [int(row[0]) for row in datasizes]
        total_data = sum(datasize for datasize in datasizes)

        # We have a dictionary where for each client we have a dictionary of his weights for each layer of the model. (models)
        # We will create a dictionary where each key is a layer and each value is the weighted average of all client's weight for every layer. (weighted_average)
        weighted_average_model = self._weighted_average_(all_client_model_weights, datasizes, total_data)
        weighted_average_classifier = self._weighted_average_(all_client_classifier_weights, datasizes, total_data)
        weighted_average_server = self._weighted_average_(all_server_model_weights, datasizes, total_data)

        return weighted_average_model, weighted_average_classifier, weighted_average_server
    
    def _weighted_average_(self, all_weights_fetched: list, data_per_client: list, total_data: int):
        """
        We calculate the weighted average of each client's weights layer-wise.

        Args:
            dict_of_weight_dicts = Dictionary that contains a dictionary with each client's weights per model layer for all clients
            
            data_per_client = Each client's datasize for all clients
            
            total_data = total amount of training batches of 32 images

        Returns:
        A dictionary where the key is the layer name and the value is its weighted average across all clients
        """
        weights = {}
        for i in range(len(all_weights_fetched)):
            if all_weights_fetched[i][0]is not None:
                # Load the data and rebuild the OrderedDict
                weight_dict = pickle.loads(all_weights_fetched[i][0])
                # For each layer compute the weighted average
                for key in weight_dict.keys():
                    # Check if we insert for the first time or not
                    if key not in weights:
                        weights[key] = weight_dict[key] * (data_per_client[i] / total_data)
                    else:
                        weights[key] += weight_dict[key] *  (data_per_client[i] / total_data)
        return weights
    
    def aggregate_client_models(self, client_id: int, model_weighted_average: dict, classifier_weighted_average: dict):
        """
        Perform client side model aggregation.

            aggregated_model = GEN_PER_THR * updated_weights + (1 - GEN_PER_THR) * Σ updated_weights * (client_datasize / total_data)

        """
        # Retrieve specific client's weights for both model and classifier
        query = 'SELECT model_updated_weights FROM weights WHERE client_id = ?'
        curr_client_model_weights = pickle.loads(self._execute_query(query=query, values=(client_id,), fetch_data_flag=True))

        query = 'SELECT classifier_updated_weights FROM weights WHERE client_id = ?'
        curr_client_classifier_weights = pickle.loads(self._execute_query(query=query, values=(client_id,), fetch_data_flag=True))
        global LAMBDA
        # Calculate weighted sum of the weighted average and the specific clients weights
        curr_client_model_weights = {key: value * LAMBDA for key,value in curr_client_model_weights.items()}
        curr_client_classifier_weights = {key: value * LAMBDA for key,value in curr_client_classifier_weights.items()}

        weighted_average_model = {key: value * (1 - LAMBDA) for key,value in model_weighted_average.items()}
        weighted_average_classifier = {key: value * (1 - LAMBDA) for key,value in classifier_weighted_average.items()}
        # Perform the aggregation
        aggregated_weights_model = {key: curr_client_model_weights[key] + weighted_average_model[key] for key in curr_client_model_weights}
        aggregated_weights_classifier = {key: curr_client_classifier_weights[key] + weighted_average_classifier[key] for key in curr_client_classifier_weights}
        #Update db with aggregated weights for said client
        query = 'UPDATE weights SET model_aggregated_weights = ? WHERE client_id = ?'
        self._execute_query(query, values=(pickle.dumps(aggregated_weights_model), client_id))
        query = 'UPDATE weights SET classifier_aggregated_weights = ? WHERE client_id = ?'
        self._execute_query(query=query, values=(pickle.dumps(aggregated_weights_classifier), client_id))
        
        print(f'[+] Aggregated model for client_id: {client_id} stored.')
        return aggregated_weights_model, aggregated_weights_classifier
    
    def test(self, server_model: nn.Module, clientsocket: socket.socket, client_id, client_address):
        with testing_lock:
            clientsocket.send(b'<TEST>')
            server_model.eval()
            print(f'\n[+] Testing with {client_id, client_address}.')
            testing_packet_event.wait()
            testing_packet_event.clear()
            storing_event.wait()
            query = 'SELECT testing_batches FROM testing WHERE client_id = ?'
            client_batches = self._execute_query(query=query, values=(client_id,), fetch_data_flag=True)
            storing_event.clear()
            total = 0
            correct = 0
            for i in range(client_batches):
                with torch.inference_mode():
                    clientsocket.send(b'<SEND>')
                    testing_packet_event.wait()
                    testing_packet_event.clear()
                    query = 'SELECT testing_outputs FROM testing WHERE client_id = ?'
                    client_outputs = pickle.loads(self._execute_query(query=query, values=(client_id, ), fetch_data_flag=True))
                    query = 'SELECT testing_labels FROM testing WHERE client_id = ?'
                    client_labels = pickle.loads(self._execute_query(query=query, values=(client_id, ), fetch_data_flag=True))
                    testing_packet_event.clear()
                    client_outputs, client_labels = client_outputs.to(self.device), client_labels.to(self.device)
                    server_outputs = server_model(client_outputs)
                    _, predicted = torch.max(server_outputs, 1)
                    total += client_labels.size(0)
                    correct += (predicted == client_labels).sum().item()
                    del client_outputs, client_labels
                    torch.cuda.empty_cache()
            print(f'[+] Testing Accuracy with {client_id, client_address}: {correct / total: .2%}')
            return 100 * (correct / total)
    
    def _store_statistics(self, client_id, global_losses, train_losses, val_losses, val_accs, test_acc):
        #self.produce_graph(global_losses, 'Epochs', 'Global Loss', 'Global Learning Curve', f'server_data/server_{self.server_id}/figures/global_loss')
        #self.produce_graph(train_losses, 'Epochs', 'Training Loss', 'Server Learning Curve', f'server_data/server_{self.server_id}/figures/train_loss')
        #self.produce_graph(val_losses, 'Epochs', 'Validation Loss', 'Server Evaluation Curve', f'server_data/server_{self.server_id}/figures/val_loss')
        #self.produce_graph(val_accs, 'Epochs', 'Accuracy Percentage', 'Server Validation Accuracy', f'server_data/server_{self.server_id}/figures/val_acc')
        query = """
            INSERT INTO statistics(client_id, epoch, global_loss, server_training_loss, server_validation_loss, server_validation_acc) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (client_id, epoch) DO UPDATE SET global_loss = ?, server_training_loss = ?, server_validation_loss = ?, server_validation_acc = ?
            """
        for idx in range(len(global_losses)):
            self._execute_query(query, (client_id, idx, global_losses[idx], train_losses[idx], val_losses[idx], val_accs[idx], global_losses[idx], train_losses[idx], val_losses[idx], val_accs[idx]))
        
        query = 'UPDATE statistics SET server_test_acc = ? WHERE client_id = ? AND epoch = ?'
        self._execute_query(query=query, values=(test_acc, client_id, EPOCHS - 1))

        query = f"SELECT * FROM statistics WHERE client_id = {client_id}"
        conn = sqlite3.Connection(self.server_db)
        pd.read_sql_query(sql=query, con=conn).to_csv(f'server_data/stats_{client_id}.csv')
        conn.close()
        print('[+] Successfully stored data in database')

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

class ServerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3)
        self.fc1 = nn.Linear(2*2*512, 256)
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
    # In order to pass the serverid from the command line
    arg = 1 if len(sys.argv) < 2 else int(sys.argv[1])
    Server(arg)
    