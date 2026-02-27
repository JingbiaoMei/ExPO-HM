from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from tqdm import tqdm
import numpy as np
import json
#from data_loader.feature_loader import get_attrobj_from_ids
from sklearn.preprocessing import MultiLabelBinarizer
import re
"""
For linear probe/fine tune/zero-shot
"""


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (str)):
        return data
    if isinstance(data, list) and all(isinstance(elem, list) for elem in data):
        return data
    if isinstance(data, list) and all(isinstance(elem, str) for elem in data):
        return data
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data.to(device)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


## Feature loader for loading pre-extracted CLIP features
# Used for fine_tune/zero_shot/linear_classifier
class feature_dataset(Dataset):
    def __init__(self, features, labels, device):
        self.features = features.clone().detach().type(torch.float32)
        self.labels = labels.clone().detach().type(torch.float32).to(device)
        self.length = self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return self.length


# Video and Text Dataset
# Used for HateMM video-based inference
class video_text_dataset(Dataset):
    def __init__(
        self,
        video_data,
        preprocess=None,
        device="cuda",
        label_str=False,
    ):
        # video_data is a list of [list_video_path,list_text,list_label,list_ids]
        list_video_path, list_text, list_label, list_ids = video_data
        self.video_path = list_video_path
        self.text = list_text
        self.label = list_label
        self.list_ids = list_ids
        self.preprocess = preprocess
        self.device = device
        self.label_str = label_str

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        # For video data, we return the path directly since video processing
        # will be handled by the inference engine
        video_path = self.video_path[idx]
        text = self.text[idx]
        label = self.label[idx]
        
        if not self.label_str:
            label = torch.tensor(label)
            
        return video_path, text, label, self.list_ids[idx]


# Image and Text Dataset
# Used for fine_tune/zero_shot/linear_classifier/generate clip embeddings
class image_text_dataset(Dataset):
    def __init__(
        self,
        img_data,
        preprocess=None,
        device="cuda",
        image_size=224,
        label_str=False,
    ):
        # img_data is a list of [list_image_path,list_text,list_label,list_ids]
        list_image_path, list_text, list_label, list_ids = img_data
        self.image_path = list_image_path
        self.text = list_text  # you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.label = list_label
        self.list_ids = list_ids
        self.preprocess = preprocess
        self.device = device
        # HuggingFace Tokenizer has different definition vs OPENAI CLIP
        self.image_size = image_size
        self.label_str = label_str

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if self.preprocess is not None:
            image = self.preprocess(
                images=Image.open(self.image_path[idx]).convert('RGB').resize((self.image_size, self.image_size)), return_tensors="pt"
            )
            image["pixel_values"] = image["pixel_values"].squeeze()
        else:
            image = self.image_path[idx]

        text = self.text[idx]
  
        label = self.label[idx]
        if not self.label_str:
            label = torch.tensor(label)
        return image, text, label, self.list_ids[idx] 

def get_values_from_gt(dataset, split):
    """
    Extract image/video path, text, gt_label and image/video ids from gt files
    input: dataset name, split name
    output: list of ... for input to image_text_dataset or video_text_dataset
    """

    if dataset.lower() == "hatemm":
        # Handle HateMM video dataset - use unfiltered CSV files
        if split == "train":
            gt_file = "./data/gt/HateMM/train.csv"
        elif split == "val" or split == "dev_seen":
            gt_file = "./data/gt/HateMM/valid.csv"
        elif split == "test" or split == "test_seen":
            gt_file = "./data/gt/HateMM/test.csv"
        else:
            raise ValueError(f"Unknown split '{split}' for HateMM dataset")
        general_annotation = "./data/gt/HateMM/HateMM_annotation.csv"
        # Load HateMM dataset from CSV
        gt_df = pd.read_csv(gt_file, header=None, names=['video_id'])
        
        list_video_path = []
        list_text = []
        list_label = []
        list_ids = []
        
        for _, row in gt_df.iterrows():
            video_id = row['video_id'].strip()
            
            # Extract label from video ID prefix
            if video_id.startswith("hate_video_"):
                label = 1
            elif video_id.startswith("non_hate_video_"):
                label = 0
            else:
                print(f"Warning: Unknown video ID format: {video_id}")
                label = 0
            
            # Build video path
            video_path = f"./data/video/HateMM/{video_id}.mp4"
            
            # Placeholder for text (no text data in standard HateMM, could be extended with ASR captions)
            text = ""
            
            list_video_path.append(video_path)
            list_text.append(text)
            list_label.append(label)
            list_ids.append(video_id)
        
        return list_video_path, list_text, list_label, list_ids

    elif dataset.lower() == "fb-fine-grained-pc" or dataset.lower() == "fb-fine-grained-attack":
        gt_file = "./data/gt/fine_grained_hateful_memes/" + split + ".json"
        gt_df = pd.read_json(gt_file, lines=True, dtype=False)
        # Get the ordered list of image ids
        list_imgs = gt_df["img"].values
        list_ids = [imgs.replace("img/", "").replace(".png", "") for imgs in list_imgs]
        # Get the ordered list of text and labels
        list_text = gt_df["text"].to_list()
    elif dataset != "MultiOFF" and "Memotion" not in dataset and "fine-grain" not in dataset.lower():
        gt_file = "./data/gt/" + dataset + "/" + split + ".jsonl"
        gt_df = pd.read_json(gt_file, lines=True, dtype=False)

        # Get the ordered list of image ids
        list_ids = gt_df["id"].values
        # Get the ordered list of text and labels
        list_text = gt_df["text"].to_list()
        
    elif "mami-fine-grain" in dataset.lower():
        gt_file = "./data/gt/MAMI/" + split + "_attack.json"
        gt_df = pd.read_json(gt_file, lines=True, dtype=False)
                # Get the ordered list of image ids
        list_ids = gt_df["id"].values
        # Get the ordered list of text and labels
        list_text = gt_df["text"].to_list()
        
    elif "pridemm-fine-grain" in dataset.lower():
        if "stance" in dataset.lower():
            gt_file = "./data/gt/PrideMM/" + split + "_fb_style_stance.json"
        elif "target" in dataset.lower():
            gt_file = "./data/gt/PrideMM/" + split + "_fb_style_target.json"
        else:
            print("Error, do not know the fine-grained task within PrideMM")
            return None
        gt_df = pd.read_json(gt_file, lines=True, dtype=False)
                # Get the ordered list of image ids
        list_ids = gt_df["id"].values
        # Get the ordered list of text and labels
        list_text = gt_df["text"].to_list()
    else:
        raise ValueError("{} Dataset not supported".format(dataset))
        gt_df = None
    # Get the ordered list of image paths
    list_image_path = []

    
    
    if dataset == "FB" or dataset.lower() == "fb":
        list_label = gt_df["label"].to_list()
        for img_id in list_ids:
            list_image_path.append("./data/image/" + dataset + "/All/" + img_id + ".png")
    elif dataset.lower() == "fb-fine-grained-pc":
        list_label = gt_df["gold_pc"].to_list()
        for img_id in list_ids:
            list_image_path.append("./data/image/" + "FB" + "/All/" + img_id + ".png")
    elif dataset.lower() == "fb-fine-grained-attack":
        list_label = gt_df["gold_attack"].to_list()
        for img_id in list_ids:
            list_image_path.append("./data/image/" + "FB" + "/All/" + img_id + ".png")
    elif dataset == "MAMI" or "mami" in dataset.lower():
        
        for img_id in list_ids:
            list_image_path.append("./data/image/" + "MAMI" + "/All/" + img_id + ".jpg")
        if not "fine-grain" in dataset.lower():
            list_label = gt_df["label"].to_list()
        else:
            print("Fine-grained MAMI")
            list_label = gt_df["gold_attack"].to_list()
            
    elif dataset == "PrideMM" or "pridemm" in dataset.lower():
        
        for img_id in list_ids:
            list_image_path.append("./data/image/" + "PrideMM" + "/All/" + img_id + ".png")
        if not "fine-grain" in dataset.lower():
            list_label = gt_df["label"].to_list()
        elif "stance" in dataset.lower():
            list_label = gt_df["gold_stance"].to_list()
        elif "target" in dataset.lower():
            list_label = gt_df["gold_target"].to_list()
        else:
            print("Error, do not know the fine-grained task within PrideMM")
    elif dataset.lower() == "harmp" or dataset.lower() == "harmeme":

        list_label = gt_df["labels"]
        list_label_converted = []
        for item in list_label:
            if 'not harmful' in item:
                list_label_converted.append(0)
            else:
                list_label_converted.append(1)  # harmful
        assert len(list_label_converted) == len(list_label)
        list_label = list_label_converted
        for img_id in list_ids:
            list_image_path.append("./data/image/" + dataset + "/All/" + img_id + ".png")
        # Remove the last "\n"  in the text
        #for index, text in enumerate(list_text):
        #    list_text[index] = text[:-1]
            
    elif dataset.lower() == "harmc":
        # The HarmC is the same as HarMeme, but here we refer to 3 classes
        # 0: not harmful, 1: 
        list_label = gt_df["labels"]
        list_label_converted = []
        for item in list_label:
            if 'not harmful' in item:
                list_label_converted.append(0)
            elif "somewhat harmful" in item:
                list_label_converted.append(1)
            elif "very harmful" in item:
                list_label_converted.append(2)
        assert len(list_label_converted) == len(list_label)
        list_label = list_label_converted
        for img_id in list_ids:
            list_image_path.append("./data/image/" + dataset + "/All/" + img_id + ".png")
    elif dataset == "Propaganda":
        #list_label = gt_df["label"].to_list()
        list_image = gt_df["image"].to_list()
        for image_id in list_image:
            list_image_path.append("./data/image/" + dataset + "/All/" + image_id)
        fine_grained_labels = ['Black-and-white Fallacy/Dictatorship', 'Name calling/Labeling', 'Smears', 'Reductio ad hitlerum', 'Transfer', 'Appeal to fear/prejudice', \
            'Loaded Language', 'Slogans', 'Causal Oversimplification', 'Glittering generalities (Virtue)', 'Flag-waving', "Misrepresentation of Someone's Position (Straw Man)", \
            'Exaggeration/Minimisation', 'Repetition', 'Appeal to (Strong) Emotions', 'Doubt', 'Obfuscation, Intentional vagueness, Confusion', 'Whataboutism', 'Thought-terminating cliché', \
            'Presenting Irrelevant Data (Red Herring)', 'Appeal to authority', 'Bandwagon']
        mlb = MultiLabelBinarizer().fit([fine_grained_labels])
        gt_df = gt_df.join(pd.DataFrame(mlb.transform(gt_df['labels']), 
                                        columns=mlb.classes_, 
                                        index=gt_df.index))
        list_label = []
        for i in range(len(list_ids)):
            list_label.append(gt_df.iloc[i][fine_grained_labels].values.tolist())
            # A list of list 
            # Each sublist is something like [1,1,1,0,...,0] with 22 elements
            
    elif dataset == "Tamil" or  dataset.lower() == "tamil":
        list_label = gt_df["label"].to_list()
        list_image = gt_df["image_id"].to_list()
        for img_id in list_image:
            list_image_path.append("./data/image/" + dataset + "/All/" + img_id)
    elif dataset == "MMHS" or dataset.lower() == "mmhs":
        for index, text in enumerate(list_text):
            # Remove the url and @user
            
            text = re.sub(r' https\S+', "", text)
            text = re.sub(r'@\S+ ', "", text)
            list_text[index] = text
        
        list_label = gt_df["label"].to_list()
        for img_id in list_ids:
            list_image_path.append("./data/image/" + dataset + "/All/" + str(img_id) + ".jpg")
    elif dataset == "MultiOFF" or dataset.lower() == "multioff":
        if split == "train":
            gt_file = "./data/gt/" + dataset + "/" + "Training_meme_dataset.csv"
        elif split == "val":
            gt_file = "./data/gt/" + dataset + "/" + "Validation_meme_dataset.csv"
        elif split == "test":
            gt_file = "./data/gt/" + dataset + "/" + "Testing_meme_dataset.csv"
        gt_df = pd.read_csv(gt_file)
        list_image = gt_df["image_name"].to_list()
        list_ids = list_image
        list_text = gt_df["sentence"].to_list()
        list_label_text = gt_df["label"].to_list()
        for img_id in list_image:
            list_image_path.append("./data/image/" + dataset + "/All/" + img_id)
        list_label = []
        for label in list_label_text:
            if label == "Non-offensiv":
                list_label.append(0)
            elif label == "offensive":
                list_label.append(1)
            else:
                print("MultiOFF: Error, do not know the label")        
    elif "Memotion" in dataset or "memotion" in dataset.lower():
        # Humour, Sarcasm, Offense, Motivation
        if split == "train":
            gt_file = "./data/gt/" + "Memotion" + "/" + "labels.csv"
            gt_df = pd.read_csv(gt_file)
            list_image = gt_df["image_name"].to_list()
            for img_id in list_image:
                list_image_path.append("./data/image/" + "Memotion" + "/All/" + img_id)
            #print("start to test images")
            #for i, image_pth in tqdm(enumerate(list_image_path)):
            #    try:
            #        Image.open(image_pth).convert('RGB')
            #    except:
            #        print("Error found in image {}".format(i))
            list_ids = list_image
            list_text = gt_df["text_corrected"].to_list()
            list_text_supplement = gt_df["text_ocr"].to_list()
            #print(list_text[119])
            #print(list_text[119] == "nan")
            #print(list_text[119] == list_text[119])
            for i, (text, text_sup) in enumerate(zip(list_text, list_text_supplement)):
                # Address nan in input text
                if text != text:
                    print("{} Text corrected is empty, replace with OCR".format(i))
                    if text_sup == text_sup:
                        list_text[i] = text_sup
                    else:
                        # sine text sup is also nan, using empty string 
                        list_text[i] = " "
            
            list_label = []
            if dataset == "Memotion_H":
                humour = gt_df["humour"].to_list()
                for item in humour:
                    if item == "not_funny":
                        list_label.append(0)
                    else:
                        list_label.append(1)
   
            elif dataset == "Memotion_S":
                sarcasm = gt_df["sarcasm"].to_list()
                for item in sarcasm:
                    if item == "not_sarcastic":
                        list_label.append(0)
                    else:
                        list_label.append(1)
            
            elif dataset == "Memotion_O":
                offensive = gt_df["offensive"].to_list()
                for item in offensive:
                    if item == "not_offensive":
                        list_label.append(0)
                    else:
                        list_label.append(1)
            elif dataset == "Memotion_M":
                motivation = gt_df["motivational"].to_list()
                for item in motivation:
                    if item == "not_motivational":
                        list_label.append(0)
                    else:
                        list_label.append(1)
            else:
                print("Memotion: Error, do not know the task within this dataset")
        else:
            gt_file_1 = "./data/gt/" + "Memotion" + "/" + "2000_testdata.csv"
            gt_file_2 = "./data/gt/" + "Memotion" + "/" + "Meme_groundTruth.csv"
            gt_df = pd.read_csv(gt_file_1)
            gt_df_labels = pd.read_csv(gt_file_2)
            list_image = gt_df["Image_name"].to_list()
            for img_id in list_image:
                list_image_path.append("./data/image/" + "Memotion" + "/All/" + img_id)
            #print("start to test images")
            #for i, image_pth in tqdm(enumerate(list_image_path)):
            #    try:
            #        Image.open(image_pth).convert('RGB')
            #    except:
            #        print("Error found in image {}".format(i))
            
            list_ids = list_image
            list_text = gt_df["corrected_text"].to_list()
            list_text_supplement = gt_df["OCR_extracted_text"].to_list()
            for i, (text, text_sup) in enumerate(zip(list_text, list_text_supplement)):
                if text != text:
                    print("{} Text corrected is empty, replace with OCR".format(i))
                    if text_sup == text_sup:
                        list_text[i] = text_sup
                    else:
                        # sine text sup is also nan, using empty string 
                        list_text[i] = " "
            labels_pool = gt_df_labels["Labels"].to_list()
            labels_pool = [ label.split("_")[1] for label in labels_pool]
            list_label = []
            if dataset == "Memotion_H":
                for label in labels_pool:
                    #print(label)
                    list_label.append(int(label[0]))
            elif dataset == "Memotion_S":
                for label in labels_pool:
                    list_label.append(int(label[1]))
            elif dataset == "Memotion_O":
                for label in labels_pool:
                    list_label.append(int(label[2]))
            elif dataset == "Memotion_M":
                for label in labels_pool:
                    list_label.append(int(label[3]))
            else:
                print("Memotion: Error, do not know the task within this dataset")
                    
    else:
        raise ValueError("{} Dataset not supported".format(dataset))
    return list_image_path, list_text, list_label, list_ids


def get_img_ids(dataset, split):
    """
    get ordered image ids from gt files
    """
    gt_file = "./data/gt/" + dataset + "/" + split + ".jsonl"
    gt_df = pd.read_json(gt_file, lines=True, dtype=False)
    list_ids = gt_df["id"].values
    return list_ids


# Load values into DL
def get_Dataloader(
    preprocess=None,
    batch_size=128,
    num_workers=4,
    train_batch_size=32,
    device="cuda",
    image_size=224,
    dataset="FB",
):
    if dataset.lower() == "hatemm":
        # Use video dataset for HateMM
        video_dataset = video_text_dataset(
            get_values_from_gt(dataset, "train"),
            preprocess,
            label_str=True if "fine-grained" in dataset.lower() else False,
        )
        train = DataLoader(
            video_dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True
        )
        train = DeviceDataLoader(train, device)
        
        # Validation set
        video_dataset = video_text_dataset(
            get_values_from_gt(dataset, "val"),
            preprocess,
            label_str=True if "fine-grained" in dataset.lower() else False,
        )
        dev_seen = DataLoader(
            video_dataset, batch_size=batch_size, num_workers=num_workers
        )
        dev_seen = DeviceDataLoader(dev_seen, device)
        
        # Test set
        video_dataset = video_text_dataset(
            get_values_from_gt(dataset, "test"),
            preprocess,
            label_str=True if "fine-grained" in dataset.lower() else False,
        )
        test_seen = DataLoader(
            video_dataset, batch_size=batch_size, num_workers=num_workers
        )
        test_seen = DeviceDataLoader(test_seen, device)
        
        return train, dev_seen, test_seen
        
    else:
        # Original image-based datasets
        imgtxt_dataset = image_text_dataset(
            get_values_from_gt(dataset, "train"),
            preprocess,
            image_size=image_size,
            label_str=True if "fine-grained" in dataset.lower() else False,
        )
        train = DataLoader(
            imgtxt_dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True
        )  # Define your own dataloader
        train = DeviceDataLoader(train, device)
        if dataset == "FB":
            dev_seen_name = "dev_seen"
        elif "fb-fine-grained" in dataset.lower():
            dev_seen_name = "dev_seen"
        else:
            dev_seen_name = "val"
        imgtxt_dataset = image_text_dataset(
            get_values_from_gt(dataset, dev_seen_name),
            preprocess,
            image_size=image_size,
            label_str=True if "fine-grained" in dataset.lower() else False,
        )
        dev_seen = DataLoader(
            imgtxt_dataset, batch_size=batch_size, num_workers=num_workers
        )  # Define your own dataloader
        dev_seen = DeviceDataLoader(dev_seen, device)
        
        if dataset == "FB":
            test_seen_name = "test_seen"
        elif "fb-fine-grained" in dataset.lower():
            test_seen_name = "dev_unseen"
        else:
            test_seen_name = "test"
        
        imgtxt_dataset = image_text_dataset(
            get_values_from_gt(dataset, test_seen_name),
            preprocess,
            image_size=image_size,
            label_str=True if "fine-grained" in dataset.lower() else False,
        )
        test_seen = DataLoader(
            imgtxt_dataset, batch_size=batch_size, num_workers=num_workers
        )  # Define your own dataloader
        test_seen = DeviceDataLoader(test_seen, device)
        
        if dataset == "FB":
            imgtxt_dataset = image_text_dataset(
                get_values_from_gt(dataset, "test_unseen"),
                preprocess,
                image_size=image_size,
            )
            
            test_unseen = DataLoader(
                imgtxt_dataset, batch_size=batch_size, num_workers=num_workers
            )  # Define your own dataloader
            test_unseen = DeviceDataLoader(test_unseen, device)
            return train, dev_seen, test_seen, test_unseen
        
        else:
            return train, dev_seen, test_seen
