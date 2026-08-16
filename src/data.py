import json
import math
import os
import pandas as pd


PROCFOLDER = 'processedMovies'
AVGGFILE = 'Avg.tif'
MASKFILE = 'Masks.tif'
ANNOTATIONFILE = 'Annotations.tif'


def load_all_excels(folder='./experiment_lists', 
                    exclusions = ['ChAT', 'iAChSnFR', 'iGluSnFR', 'IHC recordings annotated', 'NeuroD', 'Pax2', 'SNAP25', 'Snap25','Cav','TMEM','Otof','FVB','Atoh1selected'],
                    inclusions = None,
                    drive = None
                    ):
    '''
    Load all excel files in the specified folder, excluding certain files based on forbidden keywords, and concatenate them into a single DataFrame. The function also processes the 'Folder' column to ensure consistent formatting and removes any rows marked for discard.
    '''
    dfs = []
    if (exclusions is not None ) and  (inclusions is not None):
        raise ValueError("Cannot specify both exclusions and inclusions. Please choose one.")
    
    for file in os.listdir(folder):
        if file.endswith('.xlsx'):
            skip = False
            if exclusions is not None:
                for forbidden in exclusions:
                    if forbidden in file:
                        skip = True
                        break
            elif inclusions is not None:
                skip = True
                for allowed in inclusions:
                    if allowed in file:
                        skip = False
                        break
            if not skip:
                df = pd.read_excel(os.path.join(folder, file))
                df['Source_file'] = file.strip('.xlsx')
                dfs.append(df)

    alldata = pd.concat(dfs, ignore_index=True)
    alldata['Folder'] = alldata['Folder'].str.replace('\\', '/')
    #discard discarded
    alldata = alldata[alldata['discard'] != 1]
    #MACOS
    if drive is not None:
        alldata['Folder'] = alldata['Folder'].str.replace('Z:', drive)
    alldata.dropna(subset=['Folder'], inplace=True)

    #Create unique id for each Independent experiment
    alldata['UniqueID'] = alldata['Source_file'] + alldata['Independent recordings number'].astype(str)
    return alldata

import subprocess
import tqdm
from shutil import copy2
def generate_dataset(alldata,copy_type='local',overwrite=False,local_data_folder='data',require_annotations=True,maskfilename = MASKFILE):
    # Crawl the folder and check that avg and masks exists, move them to the 'data' folder
    '''
    Generate a dataset by verifying the existence of average, mask, and annotation files for each entry in the provided DataFrame. The function can optionally copy these files to a local data folder, and it returns a new DataFrame containing the relevant file paths and metadata.
    Parameters:
    - alldata: DataFrame containing the original data with folder paths and metadata.
    - copy_type: String indicating the type of copy to perform ('local': copy files locally, 'local_no_copy': only check for local file existence without copying, 'remote': use remote file paths).
    - overwrite: Boolean indicating whether to overwrite existing files in the local data folder.
    - local_data_folder: Path to the local data folder where files will be copied if copy_type is 'local'.
    Returns:
    - dfdata: DataFrame containing the new file paths and relevant metadata for each entry.
    '''
    dfdata = []
    for index, row in tqdm.tqdm(alldata.iterrows(), total=alldata.shape[0]):
        folder = row['Folder']
        sanitised_folder = folder.split('/')[-2:]
        sanitised_folder = '_'.join(sanitised_folder)
        avgfile = os.path.join(folder, PROCFOLDER, AVGGFILE)
        maskfile = os.path.join(folder, PROCFOLDER, maskfilename)
        annotationfile = os.path.join(folder, PROCFOLDER, ANNOTATIONFILE)

        # Copy the files to the data folder
        if (copy_type == 'local') or (copy_type == 'local_no_copy'):
            if copy_type == 'local':
                if not os.path.exists(avgfile):
                    print(f"Avg file does not exist: {avgfile}")
                    continue
                if not os.path.exists(maskfile):
                    print(f"Masks file does not exist: {maskfile}")
                    continue
                if require_annotations and not os.path.exists(annotationfile):
                    print(f"Annotation file does not exist: {annotationfile}")
                    continue

            new_avgfile = os.path.join(local_data_folder, f"{sanitised_folder}_Avg.tif")
            new_maskfile = os.path.join(local_data_folder, f"{sanitised_folder}_Masks.tif")
            if require_annotations:
                new_annotationfile = os.path.join(local_data_folder, f"{sanitised_folder}_Annotations.tif")

        elif copy_type == 'remote':
            new_avgfile = avgfile
            new_maskfile = maskfile
            new_annotationfile = annotationfile
        else:
            raise ValueError(f"Invalid copy_type: {copy_type}. Must be 'local', 'local_no_copy', or 'remote'.")

        dfdata.append({
            'UniqueID': row['UniqueID'],
            'Source_file': row['Source_file'],
            'Independent recordings number': row['Independent recordings number'],
            'Folder': folder,
            'Avgfile': new_avgfile,
            'Maskfile': new_maskfile,
            'Annotationfile': new_annotationfile if require_annotations else None
        })
        #copy the files
        if copy_type == 'local':
            if (not os.path.exists(new_avgfile)) or overwrite:
                copy2(avgfile, new_avgfile)
            if (not os.path.exists(new_maskfile)) or overwrite:
                copy2(maskfile, new_maskfile)
            if require_annotations:
                if (not os.path.exists(new_annotationfile)) or overwrite:
                    copy2(annotationfile, new_annotationfile)
        elif copy_type == 'local_no_copy':
            if not os.path.exists(new_avgfile):
                print(f"Avg file does not exist locally: {new_avgfile}")
                continue
            if not os.path.exists(new_maskfile):
                print(f"Masks file does not exist locally: {new_maskfile}")
                continue
            if require_annotations:
                if not os.path.exists(new_annotationfile):
                    print(f"Annotation file does not exist locally: {new_annotationfile}")
                    continue

    dfdata = pd.DataFrame(dfdata)
    dfdata.to_csv(os.path.join(local_data_folder, 'dfdata.csv'), index=False)
    #dfdata.set_index('Avgfile', inplace=True)

    return dfdata

import tifffile


def _load_split_json(split_json_path):
    if split_json_path is None or not os.path.exists(split_json_path):
        return set(), set()

    with open(split_json_path, 'r') as f:
        split_data = json.load(f)

    if not isinstance(split_data, dict):
        raise ValueError("Split JSON must contain a dictionary.")

    if 'splits' in split_data:
        split_data = split_data['splits']
    elif 'experiments' in split_data:
        split_data = split_data['experiments']

    split_keys = {'train', 'training', 'train_ids', 'validation', 'val', 'validation_ids', 'val_ids'}
    if split_keys.intersection(split_data):
        train_ids = split_data.get(
            'train',
            split_data.get('training', split_data.get('train_ids', []))
        )
        val_ids = split_data.get(
            'validation',
            split_data.get('val', split_data.get('validation_ids', split_data.get('val_ids', [])))
        )
        return set(train_ids), set(val_ids)

    train_ids = []
    val_ids = []
    for experiment_id, split in split_data.items():
        if isinstance(split, dict):
            split = split.get('split')
        if split in ('train', 'training'):
            train_ids.append(experiment_id)
        elif split in ('val', 'validation'):
            val_ids.append(experiment_id)

    return set(train_ids), set(val_ids)


def _json_safe_id(experiment_id):
    if hasattr(experiment_id, 'item'):
        return experiment_id.item()
    return experiment_id


def _desired_validation_count(test_size, n_experiments):
    if n_experiments == 0:
        return 0
    if isinstance(test_size, int):
        if test_size < 0 or test_size > n_experiments:
            raise ValueError("Integer test_size must be between 0 and the number of experiments.")
        return test_size
    if not 0 < test_size < 1:
        raise ValueError("Float test_size must be between 0 and 1.")
    return math.ceil(n_experiments * test_size)


def _save_split_json(split_json_path, train_ids, val_ids, column, test_size, random_state):
    split_folder = os.path.dirname(split_json_path)
    if split_folder:
        os.makedirs(split_folder, exist_ok=True)

    split_data = {
        'column': column,
        'test_size': test_size,
        'random_state': random_state,
        'train': sorted([_json_safe_id(experiment_id) for experiment_id in train_ids], key=str),
        'validation': sorted([_json_safe_id(experiment_id) for experiment_id in val_ids], key=str),
    }

    with open(split_json_path, 'w') as f:
        json.dump(split_data, f, indent=2)


def _get_experiment_id(dfdata, avgfile, column):
    df_key = avgfile[2:] if avgfile.startswith('./') else avgfile

    if df_key in dfdata.index:
        return dfdata.loc[df_key, column]

    if 'Avgfile' in dfdata.columns:
        matches = dfdata[dfdata['Avgfile'].isin([df_key, avgfile])]
        if not matches.empty:
            return matches.iloc[0][column]

    raise KeyError(f"Could not find {avgfile} in dfdata.")


def train_test_split(dfdata, column='UniqueID', test_size=0.2, random_state=42,
                     split_json_path=None, save_split_json=False,
                     data_dir='data', require_annotations=True):
    '''
    Split the dataset into training and validation sets based on unique identifiers in the specified column. The function checks for the existence of average, mask, and annotation files, ensuring they have matching shapes before including them in the final dataset. It returns two lists of dictionaries representing the training and validation datasets.
    Parameters:
    - dfdata: DataFrame containing the dataset with file paths and metadata.
    - column: String specifying the column name to use for splitting the dataset (default is 'UniqueID').
    - test_size: Float representing the proportion of the dataset to include in the validation set (default is 0.2).
    - random_state: Integer seed for random number generator to ensure reproducibility of the split (default is 42).
    - split_json_path: Optional JSON file containing existing train/validation experiment assignments.
    - save_split_json: Boolean indicating whether to save the updated split to split_json_path.
    - data_dir: Directory containing the local Avg, Masks, and Annotations tif files.
    Returns:
    - train_data: List of dictionaries representing the training dataset.
    - val_data: List of dictionaries representing the validation dataset.
    '''
    data = []
    for file in os.listdir(data_dir):
        if file.endswith('_Avg.tif'):
            avgfile = os.path.join(data_dir, file)
            maskfile = os.path.join(data_dir, file.replace('_Avg.tif', '_Masks.tif'))
            if require_annotations:
                annotationfile = os.path.join(data_dir, file.replace('_Avg.tif', '_Annotations.tif'))
            else:
                annotationfile = None
            if os.path.exists(maskfile):
                #check that the avg and mask have the same shape
                avg = tifffile.imread(avgfile)
                mask = tifffile.imread(maskfile)
                if require_annotations:
                    annotation = tifffile.imread(annotationfile)
                
                if avg.shape != mask.shape:
                    print(f"Shape mismatch: {avgfile} and {maskfile}")
                    continue
                if require_annotations and (avg.shape != annotation.shape):
                    print(f"Shape mismatch: {avgfile} and {annotationfile}")
                    continue

                experiment_id = _get_experiment_id(dfdata, avgfile, column)

                sample = {
                    'image': avgfile,
                    'label': maskfile,
                    "image_path": avgfile,
                    "label_path": maskfile,
                    'shape': avg.shape,
                    column: experiment_id
                }
                if require_annotations:
                    sample['annotation'] = annotationfile
                    sample["annotation_path"] = annotationfile

                data.append(sample)

    data = pd.DataFrame(data)
    if data.empty:
        if save_split_json:
            if split_json_path is None:
                raise ValueError("split_json_path is required when save_split_json=True.")
            saved_train_ids, saved_val_ids = _load_split_json(split_json_path)
            _save_split_json(
                split_json_path,
                saved_train_ids,
                saved_val_ids,
                column,
                test_size,
                random_state
            )
        return [], []

    uniqueIDs = data[column].unique()
    previous_train_ids, previous_val_ids = _load_split_json(split_json_path)
    duplicate_ids = previous_train_ids.intersection(previous_val_ids)
    if duplicate_ids:
        raise ValueError(f"Experiments assigned to both train and validation in JSON: {sorted(duplicate_ids, key=str)}")

    current_ids = set(uniqueIDs)
    previous_train_ids = previous_train_ids.intersection(current_ids)
    previous_val_ids = previous_val_ids.intersection(current_ids)
    previous_ids = previous_train_ids.union(previous_val_ids)
    new_ids = [experiment_id for experiment_id in uniqueIDs if experiment_id not in previous_ids]

    target_val_count = _desired_validation_count(test_size, len(uniqueIDs))
    new_val_count = target_val_count - len(previous_val_ids)
    new_val_count = max(0, min(len(new_ids), new_val_count))

    if len(new_ids) == 0:
        new_train_ids = []
        new_val_ids = []
    elif new_val_count == 0:
        new_train_ids = new_ids
        new_val_ids = []
    elif new_val_count == len(new_ids):
        new_train_ids = []
        new_val_ids = new_ids
    else:
        from sklearn.model_selection import train_test_split
        new_train_ids, new_val_ids = train_test_split(
            new_ids,
            test_size=new_val_count,
            random_state=random_state
        )

    train_idx = previous_train_ids.union(new_train_ids)
    val_idx = previous_val_ids.union(new_val_ids)

    train_data = data[data[column].isin(train_idx)]
    val_data = data[data[column].isin(val_idx)]

    if save_split_json:
        if split_json_path is None:
            raise ValueError("split_json_path is required when save_split_json=True.")
        saved_train_ids, saved_val_ids = _load_split_json(split_json_path)
        _save_split_json(
            split_json_path,
            saved_train_ids.union(train_idx),
            saved_val_ids.union(val_idx),
            column,
            test_size,
            random_state
        )

    #Dataframe to list of dictionaries
    train_data = train_data.to_dict('records')
    val_data = val_data.to_dict('records')
    return train_data, val_data


from monai.data import Dataset, DataLoader
def create_datasets_and_loaders(train_data, val_data, train_transforms, val_transforms,
                                batch_size=2, num_workers=0, val_batch_size=1,
                                shuffle_train=True, shuffle_val=False):
    train_ds = Dataset(data=train_data, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)

    val_ds = Dataset(data=val_data,transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=shuffle_val, num_workers=num_workers)

    return train_ds, train_loader, val_ds, val_loader
