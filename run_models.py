import torch
import torch.nn as nn
import numpy as np
import os

current_file_dir = os.path.dirname(os.path.abspath(__file__))


class YourModel(nn.Module):
    def __init__(self, input_size):
        super(YourModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)


model_cache = {}


def load_model(model_path, input_size):
    if model_path not in model_cache:
        model = YourModel(input_size)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)  
        model.eval()
        model_cache[model_path] = model
    return model_cache[model_path]


def process_input(input_x):
    if isinstance(input_x, torch.Tensor):
        input_x = input_x.detach().numpy()  
    input_flat = np.reshape(input_x, (1, -1))
    return torch.from_numpy(input_flat).float()



def predict(model_type, input_x, file_name):
    input_tensor = process_input(input_x)
    model_path = os.path.join(current_file_dir, f'resources/models/{model_type}/{file_name}')
    model = load_model(model_path, input_tensor.shape[1])
    with torch.no_grad():
        predictions = model(input_tensor)
        predicted_indices = torch.argmax(predictions, dim=1)

        index_to_label = {0: -1, 1: 0, 2: 1}
        predicted_labels = index_to_label[predicted_indices.item()]

    return predicted_labels


def get_direction_for_softmax(input_x, model_type):
    return predict(model_type, input_x, 'Softmax_model_v100.pth')


def get_direction_for_log(input_x, model_type):
    return predict(model_type, input_x, 'log_dataset_model_v002.pth')


def get_direction_for_sigmoid(input_x, model_type):
    return predict(model_type, input_x, 'sigmoid_dataset_model_v001.pth')


def get_direction_for_exp(input_x, model_type):
    # return predict(model_type, input_x, 'exp_dataset_model_v100.pth')

    return predict(model_type, input_x, 'exp_dataset_model_v101.pth')

def get_direction_for_tanh(input_x, model_type):
    return predict(model_type, input_x, 'tanh_dataset_model_v001.pth')


def get_direction_for_sqrt(input_x, model_type):
    return predict(model_type, input_x, 'sqrt_dataset_model_v001.pth')


def get_direction_for_rsqrt(input_x, model_type):
    return predict(model_type, input_x, 'rsqrt_dataset_model_v001.pth')


def get_direction_for_relu(input_x, model_type):
    return predict(model_type, input_x, 'relu_model_v100.pth')


def get_direction_for_conv2d(input_x, model_type):
    return predict(model_type, input_x, 'conv2d_dataset_model_v001.pth')


def predict_for_binary_parameter(input_x, input_y, model_type, file_name):
    # Flatten and concatenate input_x and input_y
    if isinstance(input_x, torch.Tensor):
        input_x = input_x.detach().numpy()
    if isinstance(input_y, torch.Tensor):
        input_y = input_y.detach().numpy()

    input_x_flat = np.reshape(input_x, (1, -1))
    input_y_flat = np.reshape(input_y, (1, -1))

    combined_input = np.concatenate((input_x_flat, input_y_flat), axis=1)

    # Convert to PyTorch tensor
    input_tensor = torch.from_numpy(combined_input).float()

    # Load the model
    # model_path = os.path.join(current_file_dir, f'resources/models/{model_type}/matmul_dataset_model_v001.pth')
    model_path = os.path.join(current_file_dir, f'resources/models/{model_type}/{file_name}')
    model = load_model(model_path, input_tensor.shape[1])

    # Make prediction
    with torch.no_grad():
        predictions = model(input_tensor)
        predicted_labels = torch.argmax(predictions, dim=1)

    # Return the prediction adjusted by -1, as in your previous functions
    return predicted_labels.item() - 1


def get_direction_for_matmul(input_x, input_y, model_type):
    return predict_for_binary_parameter(input_x, input_y, model_type, 'matmul_dataset_model_v001.pth')


def get_direction_for_div(input_x, input_y, model_type):
    return predict_for_binary_parameter(input_x, input_y, model_type, 'matmul_div_model_v001.pth')


def get_direction_for_cos_sim(input_x, input_y, model_type):
    return predict_for_binary_parameter(input_x, input_y, model_type, 'cosine_similarity_model_v002.pth')
