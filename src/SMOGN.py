import os
import numpy as np
import pandas as pd
import smogn


def load_data(feature_file, label_file):
    """
    Load feature and label data from text files.
    """
    features = np.loadtxt(feature_file)
    labels = np.loadtxt(label_file)

    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    return features, labels


def apply_smogn(df_shifted, target_label_col, offset_value):
    """
    Apply SMOGN to shifted data and unshift the result.
    Returns None on failure.
    """
    # Apply SMOGN to shifted data
    df_smogn = smogn.smoter(
        data=df_shifted,
        y=target_label_col,
        k=7,
        samp_method='balance'
    )

    # Unshift data
    df_unmasked = df_smogn - offset_value
    return df_unmasked

def process_data(folder_name, target_dim):
    """
    Process data and generate training set for a specified label dimension.

    Parameters:
    - folder_name (str): Folder containing data
    - target_dim (int): Index of the label dimension to process

    Returns:
    - dict: Shape information of processed data {'features': shape, 'labels': shape}
    """
    features_path = os.path.join('data/raw', folder_name, 'train_features.txt')
    labels_path = os.path.join('data/raw', folder_name, 'train_labels.txt')

    features, labels = load_data(features_path, labels_path)

    # Validate target dimension
    assert target_dim < labels.shape[1], "Target dimension out of range"

    # Extract label data for the specified dimension
    feature_cols = [f'feature_{i}' for i in range(features.shape[1])]
    label_col = f'label_{target_dim}'

    df_features = pd.DataFrame(features, columns=feature_cols)
    df_labels = pd.DataFrame(labels[:, target_dim].reshape(-1, 1), columns=[label_col])

    # Combine features and current label
    df_combined = pd.concat([df_features, df_labels], axis=1)

    # Offset value for SMOGN
    offset_value = 1000
    df_shifted = df_combined + offset_value

    # Apply SMOGN
    df_smogn_unmasked = apply_smogn(df_shifted, label_col, offset_value)

    # Split back into features and label
    smogn_features = df_smogn_unmasked[feature_cols]
    smogn_label = df_smogn_unmasked[[label_col]]

    # Save to files
    output_folder = os.path.join('data/processed', folder_name)
    os.makedirs(output_folder, exist_ok=True)

    features_file = os.path.join(output_folder, f'smogn_features_dim{target_dim}.txt')
    labels_file = os.path.join(output_folder, f'smogn_labels_dim{target_dim}.txt')

    np.savetxt(features_file, smogn_features.values, fmt='%.6f')
    np.savetxt(labels_file, smogn_label.values, fmt='%.6f')

    # Return shape information
    return {
        'features': smogn_features.shape,
        'labels': smogn_label.shape
    }


if __name__ == "__main__":
    # Dataset 1 - Dimension 0
    shape_info = process_data(folder_name='1', target_dim=0)
    print(f"[Dataset: dataset/1, Dim: 0] Features shape: {shape_info['features']}, Labels shape: {shape_info['labels']}")

    # Dataset 2 - Dimension 0
    shape_info = process_data(folder_name='2', target_dim=0)
    print(f"[Dataset: dataset/2, Dim: 0] Features shape: {shape_info['features']}, Labels shape: {shape_info['labels']}")

    # Dataset 2 - Dimension 1
    shape_info = process_data(folder_name='2', target_dim=1)
    print(f"[Dataset: dataset/2, Dim: 1] Features shape: {shape_info['features']}, Labels shape: {shape_info['labels']}")

    # Dataset 2 - Dimension 2
    shape_info = process_data(folder_name='2', target_dim=2)
    print(f"[Dataset: dataset/2, Dim: 2] Features shape: {shape_info['features']}, Labels shape: {shape_info['labels']}")

    # Dataset 3 - Dimension 0
    shape_info = process_data(folder_name='3', target_dim=0)
    print(f"[Dataset: dataset/3, Dim: 0] Features shape: {shape_info['features']}, Labels shape: {shape_info['labels']}")

    # Dataset 3 - Dimension 1
    shape_info = process_data(folder_name='3', target_dim=1)
    print(f"[Dataset: dataset/3, Dim: 1] Features shape: {shape_info['features']}, Labels shape: {shape_info['labels']}")

    # Dataset 4 - Dimension 0
    shape_info = process_data(folder_name='4', target_dim=0)
    print(f"[Dataset: dataset/4, Dim: 0] Features shape: {shape_info['features']}, Labels shape: {shape_info['labels']}")