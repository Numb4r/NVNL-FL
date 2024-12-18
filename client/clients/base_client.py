import flwr as fl
from models import create_cnn, create_dnn,create_lenet5
from dataset_utils import ManageDatasets, load_data_flowerdataset
from client.common.utils import flat_parameters

class BaseNumpyClient(fl.client.NumPyClient):
    def __init__(self,cid,niid,dataset,num_clients,dirichlet_alpha,dataset_magager):
        self.cid = int(cid)
        self.niid = niid
        self.dataset = dataset
        self.num_clients = num_clients
        self.dirichlet_alpha = dirichlet_alpha
        self.dataset_magager = dataset_magager
        if dataset == 'MNIST' or dataset == 'CIFAR10':
            self.x_train, self.y_train, self.x_test, self.y_test = load_data_flowerdataset(self)
        else:
            self.x_train, self.y_train, self.x_test, self.y_test = self.load_har(dataset) #self.load_data()
            
        if dataset == 'CIFAR10':
            self.model  = create_lenet5(self.x_train.shape, 10)

        else:
            self.model  = create_dnn(self.x_train.shape, 10)
        
        self.len_shared_data  =  len(flat_parameters(self.model.get_weights()))     
    def get_parameters(self, config):
        parameters = self.model.get_wieghts()
        return parameters
    def load_har(self, dataset):
        if dataset == 'UCIHAR':
            return self.dataset_manager.load_UCIHAR()
        if dataset == 'ExtraSensory':
            return self.dataset_manager.load_ExtraSensory()
        if dataset == 'MotionSense':
            return self.dataset_manager.load_MotionSense()
    def fit(self, parameters, config):
        return [], 0, {}
    def evaluate(self, parameters, config):
        return 0,0,{}
