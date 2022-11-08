import torch
import numpy as np
import time

from fastprogress.fastprogress import format_time, master_bar, progress_bar
from model import SpanEmo
from data_loader import DataClass
from torch.utils.data import DataLoader


class EvaluateOnTest(object):
    """
    Class to encapsulate evaluation on the test set. Based off the "Tonks Library"
    :param model: PyTorch model to use with the Learner
    :param test_data_loader: dataloader for all of the validation data
    :param model_path: path of the trained model
    """

    def __init__(self, model, test_data_loader, model_path):
        self.model = model
        self.test_data_loader = test_data_loader
        self.model_path = model_path

    def predict(self, device='cuda:0', pbar=None):
        """
        Evaluate the model on a validation set
        :param device: str (defaults to 'cuda:0')
        :param pbar: fast_progress progress bar (defaults to None)
        :returns: None
        """
        self.model.to(device).load_state_dict(torch.load(self.model_path))
        self.model.eval()
        current_size = len(self.test_data_loader.dataset)
        preds_dict = {
            'y_true': np.zeros([current_size, 11]),
            'y_pred': np.zeros([current_size, 11])
        }
        start_time = time.time()
        with torch.no_grad():
            index_dict = 0
            for step, batch in enumerate(progress_bar(self.test_data_loader, parent=pbar, leave=(pbar is not None))):
                _, num_rows, y_pred, targets = self.model(batch, device)
                current_index = index_dict
                preds_dict['y_true'][current_index: current_index +
                                     num_rows, :] = targets
                preds_dict['y_pred'][current_index: current_index +
                                     num_rows, :] = y_pred
                index_dict += num_rows

        y_true, y_pred = preds_dict['y_true'], preds_dict['y_pred']
        return y_pred


def evaluate_input(filename='test.txt'):
    device = init_device()
    args = {
        '--max-length': 128,
        '--lang': 'English',
    }
    test_dataset = DataClass(args, filename)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1,
                                  shuffle=False)
    model = SpanEmo(lang='English')
    learn = EvaluateOnTest(model, test_data_loader, model_path='checkpoint.pt')
    return learn.predict(device=device)


def init_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if str(device) == 'cuda:0':
        print("Currently using GPU: {}".format(device))
        np.random.seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Currently using CPU")
    return device


def emotion_indexes_to_lables(index_arr):
    labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
              'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
    label_array = [labels[i] for i in range(11) if index_arr[i] != 0]
    return label_array if len(label_array) > 0 else ['neutral']


def save_input_to_file(input_text):
    with open('test.txt', 'w') as f:
        header = '''ID\tTweet\tanger\tanticipation\tdisgust\tfear\tjoy\tlove\toptimism\tpessimism\tsadness\tsurprise\ttrust\n'''
        body = '1\t' + input_text + '\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0'
        f.write(header + body)
