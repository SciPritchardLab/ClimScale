import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

input_dim = 175
output_dim = 55
ylims = [.1, 1]
ignore_threshold = 200
hp_floats = ['dropout', 'lr', 'hidden_units', 'num_layers']

def get_file_list(folder_path):
    file_list = os.listdir(folder_path)
    return [f for f in file_list if re.match(r'keras-tuner', f)]

def get_num_parameters(input_dim, output_dim, hidden_units, num_layers):
    num_parameters = input_dim * hidden_units + hidden_units
    for layer in range(int(num_layers) - 1):
        num_parameters += hidden_units * hidden_units + hidden_units
    num_parameters += hidden_units * output_dim + output_dim
    return num_parameters

def get_trial_dict(file_path):
    trial_dict = {}
    hyperparameter_dict = {}
    with open(file_path, 'r') as file:
        content = file.read()
        numlines = content.count('\n')
    trial_nums = re.findall(r'Search: Running Trial #(\d+)', content)
    trial_heads = ['Search: Running Trial #' + x for x in trial_nums]
    trial_labels = ['Trial #' + x for x in trial_nums]
    trial_head_lines = []
    for trial_head in trial_heads:
        match = re.search(trial_head, content)
        if match:
            trial_head_lines.append(content.count('\n', 0, match.start()) + 1)
        else:
            trial_head_lines.append(None)
        assert trial_head_lines[-1] is not None
        
    trial_splits = []
    for i in range(len(trial_head_lines)):
        lines = content.split('\n')
        if i == len(trial_head_lines) - 1:
            extracted_content = '\n'.join(lines[trial_head_lines[i]-1:])
        else:
            extracted_content = '\n'.join(lines[trial_head_lines[i]-1:trial_head_lines[i+1]])
        trial_splits.append(extracted_content)

    for i in range(len(trial_splits)):
        trial_split = trial_splits[i]
        for line in trial_split.split('\n'):
            match = re.search(r'([\d\.e+-]+)\s+\|\?\s+\|(\w+)', line)
            if match:
                value, name = match.groups()
                hyperparameter_dict[name] = value
        epoch_matches = re.finditer(r'Epoch \d+/\d+', trial_split)
        line_numbers = [trial_split.count('\n', 0, match.start()) + 2 for match in epoch_matches]
        trial_split_list = trial_split.split('\n')
        training_lines = [trial_split_list[line_number-1] for line_number in line_numbers]
        training_lines = [x for x in training_lines if 'nan' not in x]
        loss_values = np.array([float(re.search(r'loss: (\d+\.\d+)', line).group(1)) for line in training_lines])
        mse_values = np.array([float(re.search(r'mse: (\d+\.\d+)', line).group(1)) for line in training_lines])
        val_loss_values = np.array([float(re.search(r'val_loss: (\d+\.\d+)', line).group(1)) for line in training_lines])
        val_mse_values = np.array([float(re.search(r'val_mse: (\d+\.\d+)', line).group(1)) for line in training_lines])
        trial_dict[trial_labels[i]] = {}
        trial_dict[trial_labels[i]]['loss'] = loss_values
        trial_dict[trial_labels[i]]['mse'] = mse_values
        trial_dict[trial_labels[i]]['val_loss'] = val_loss_values
        trial_dict[trial_labels[i]]['val_mse'] = val_mse_values
        trial_dict[trial_labels[i]]['hyperparameters'] = hyperparameter_dict
    return trial_dict

def get_combined_dict(folder_path):
    file_list = get_file_list(folder_path)
    combined_dict = {}
    for file in file_list:
        trial_dict = get_trial_dict(os.path.join(folder_path, file))
        combined_dict.update(trial_dict)
    return combined_dict

def plot_loss_curves(combined_dict, save_fig = True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    for trial in combined_dict.keys():
        if sum(combined_dict[trial]['mse']) < ignore_threshold:
            ax1.plot(combined_dict[trial]['mse'], label=trial)
            ax1.set_yscale('log')
            ax1.set_title('Training mse')
            ax1.set_ylim(ylims)  # Set the y-axis limits for ax1
            ax1.grid(True, which='major', linestyle='-', linewidth=0.5)
            ax1.grid(True, which='minor', linestyle=':', linewidth=0.25)

    for trial in combined_dict.keys():
        if sum(combined_dict[trial]['val_mse']) < ignore_threshold:
            ax2.plot(combined_dict[trial]['val_mse'], label=trial)
            ax2.set_yscale('log')
            ax2.set_title('Validation mse')
            ax2.set_ylim(ylims)  # Set the y-axis limits for ax2
            ax2.grid(True, which='major', linestyle='-', linewidth=0.5)
            ax2.grid(True, which='minor', linestyle=':', linewidth=0.25)

    if save_fig:
        # Save the figure
        fig.savefig('loss_curves.png')

def main():
    combined_dict = get_combined_dict("logs")
    plot_loss_curves(combined_dict)

if __name__ == "__main__":
    main()
