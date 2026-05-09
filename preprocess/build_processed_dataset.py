"""
    4.                      train    test    valid
"""
import os
import shutil
import random
from tqdm import tqdm  # Internal processing step.
import json

def split_dataset(dataset_root, output_root, split_ratio):
    """
                    train      valid      test  
    - img: XX.nrrd
    - label: XX.json
    -              
    """
    # Internal processing step.
    img_path = os.path.join(dataset_root, 'img')
    seg_path = os.path.join(dataset_root, 'seg')
    scapula_path = os.path.join(dataset_root, 'scapula')
    humerus_path = os.path.join(dataset_root, 'humerus')
    type__path = os.path.join(dataset_root, 'type')

    if not os.path.exists(img_path) or not os.path.exists(seg_path) or not os.path.exists(scapula_path) \
            or not os.path.exists(humerus_path) or not os.path.exists(type__path):
        raise FileNotFoundError("   'img'   'scapula'              ")

    # Internal processing step.
    train_dataset = os.path.join(output_root, 'train')
    test_dataset = os.path.join(output_root, 'test')
    valid_dataset = os.path.join(output_root, 'valid')

    for folder in [train_dataset, test_dataset, valid_dataset]:
        os.makedirs(os.path.join(folder, "img"), exist_ok=True)
        os.makedirs(os.path.join(folder, "seg"), exist_ok=True)
        os.makedirs(os.path.join(folder, "humerus"), exist_ok=True)
        os.makedirs(os.path.join(folder, "scapula"), exist_ok=True)
        os.makedirs(os.path.join(folder, "type"), exist_ok=True)

    # Internal processing step.
    img_files = sorted([f for f in os.listdir(img_path) if f.endswith('.nrrd')])
    seg_files = sorted([f for f in os.listdir(seg_path) if f.endswith('.nrrd')])
    scapula_file = sorted([f for f in os.listdir(scapula_path) if f.endswith('.json')])
    humerus_file = sorted([f for f in os.listdir(humerus_path) if f.endswith('.json')])
    type_file = sorted([f for f in os.listdir(type__path) if f.endswith('.json')])

    img_basenames = {os.path.splitext(f)[0] for f in img_files}
    seg_basenames = {os.path.splitext(f)[0] for f in seg_files}
    scapula_basenames = {os.path.splitext(f)[0] for f in scapula_file}
    humerus_basenames = {os.path.splitext(f)[0] for f in humerus_file}
    type_basenames = {os.path.splitext(f)[0] for f in type_file}
    common_basenames = sorted(list(img_basenames & seg_basenames & scapula_basenames & humerus_basenames & type_basenames))  # Internal processing step.

    # Internal processing step.
    type_0_list = []  # Internal processing step.
    type_1_list = []  # Internal processing step.
    for basename in common_basenames:
        with open(os.path.join(type__path, basename + ".json"), 'r') as f:
            info = json.load(f)
            if info.get("type") == 0:
                type_0_list.append(basename)
            elif info.get("type") == 1:
                type_1_list.append(basename)

    def stratified_split(subset, ratio):
        random.shuffle(subset)
        n_total = len(subset)
        n_train = int(n_total * ratio[0])
        n_valid = int(n_total * ratio[1])
        n_test = n_total - n_train - n_valid
        return subset[:n_train], subset[n_train:n_train + n_valid], subset[n_train + n_valid:]

    # Internal processing step.
    train_0, valid_0, test_0 = stratified_split(type_0_list, split_ratio)
    train_1, valid_1, test_1 = stratified_split(type_1_list, split_ratio)

    # Internal processing step.
    train_set = train_0 + train_1
    valid_set = valid_0 + valid_1
    test_set = test_0 + test_1

    print(f"\n        :")
    print(f"  Train:    {len(train_1)},    {len(train_0)},   {len(train_set)}")
    print(f"  Valid:    {len(valid_1)},    {len(valid_0)},   {len(valid_set)}")
    print(f"  Test :    {len(test_1)},    {len(test_0)},   {len(test_set)}")

    # Internal processing step.
    def move_files(file_list, target_folder):
        for basename in tqdm(file_list, desc=f"    {target_folder}"):
            try:
                shutil.copy(os.path.join(img_path, basename + ".nrrd"), os.path.join(target_folder, "img", basename + ".nrrd"))
                shutil.copy(os.path.join(seg_path, basename + ".nrrd"), os.path.join(target_folder, "seg", basename + ".nrrd"))
                shutil.copy(os.path.join(scapula_path, basename + ".json"), os.path.join(target_folder, "scapula", basename + ".json"))
                shutil.copy(os.path.join(humerus_path, basename + ".json"), os.path.join(target_folder, "humerus", basename + ".json"))
                shutil.copy(os.path.join(type__path, basename + ".json"), os.path.join(target_folder, "type", basename + ".json"))
            except Exception as e:
                print(f"      {basename}       {e}")

    # Internal processing step.
    move_files(train_set, train_dataset)
    move_files(valid_set, valid_dataset)
    move_files(test_set, test_dataset)

def split_dataset_plus(dataset_root, new_type):
    refer_train_root = os.path.join(dataset_root, 'train', 'img')
    refer_valid_root = os.path.join(dataset_root, 'valid', 'img')
    refer_test_root = os.path.join(dataset_root, 'test', 'img')

    # Internal processing step.
    target_root = os.path.join(dataset_root, 'original', new_type)

    # Internal processing step.
    new_train_root = os.path.join(dataset_root, 'train', new_type)
    new_valid_root = os.path.join(dataset_root, 'valid', new_type)
    new_test_root = os.path.join(dataset_root, 'test', new_type)

    os.makedirs(new_train_root, exist_ok=True)
    os.makedirs(new_valid_root, exist_ok=True)
    os.makedirs(new_test_root, exist_ok=True)

    def copy_files(ref_dir, out_dir):
        for fname in os.listdir(ref_dir):
            if not fname.endswith(".nrrd"):
                continue
            file_id = os.path.splitext(fname)[0]  # Internal processing step.
            src_file = os.path.join(target_root, f"{file_id}.nrrd")
            dst_file = os.path.join(out_dir, f"{file_id}.nrrd")
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)
            else:
                print(f"[  ]      : {src_file}")

    copy_files(refer_train_root, new_train_root)
    copy_files(refer_valid_root, new_valid_root)
    copy_files(refer_test_root, new_test_root)
    
    
    
    

if __name__ == '__main__':
    dataset_root = r'<HOME>/Joint_seg_landmark/dataset/251128/orig'
    output_root = r'<HOME>/Joint_seg_landmark/dataset/251128'
    split_ratio = [0.6, 0.2, 0.2]

    split_dataset(dataset_root, output_root, split_ratio)
    # split_dataset_plus(output_root, new_type='diagnosis_zoom')


    
