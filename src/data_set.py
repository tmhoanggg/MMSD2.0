from torch.utils.data import Dataset
import logging
import os
from PIL import Image
import json

logger = logging.getLogger(__name__)
WORKING_PATH="/kaggle/input/full-uit-multimodal-sarcasm-dataset"

class MyDataset(Dataset):
    def __init__(self, mode, text_name, limit=None):
        self.text_name = text_name
        self.data = self.load_data(mode, limit)
        self.image_ids=list(self.data.keys())
        for id in self.data.keys():
            #self.data[id]["image_path"] = os.path.join(WORKING_PATH, "dataset_image/dataset_image", str(id))
            self.data[id]["image_path"] = os.path.join(WORKING_PATH, "dataset_image/dataset_image", self.data[id]["image"])
    
    def load_data(self, mode, limit):
        cnt = 0
        data_set=dict()
        label_mapping = {
            "not-sarcasm": 0,
            "multi-sarcasm": 1,
            "text-sarcasm": 2,
            "image-sarcasm": 3
        }
        if mode in ["train"]:
            f1= open(os.path.join(WORKING_PATH, self.text_name, mode+".json"),'r',encoding='utf-8')
            datas = json.load(f1)
            #for data in datas:
            for key, data in datas.items():
                if limit != None and cnt >= limit:
                    break

                # image = data['image']
                # sentence = data['caption']
                # label = data['label']
                file_name = data['image']
                sentence = data['caption']
                label = label_mapping[data['label']]
 
                # if os.path.isfile(os.path.join(WORKING_PATH, "dataset_image/dataset_image", str(image)+".jpg")):
                #     data_set[int(image)]={"caption":sentence, 'label': label}
                #     cnt += 1
                cur_img_path = os.path.join(WORKING_PATH, "dataset_image/dataset_image", str(file_name))
                if not os.path.exists(cur_img_path):
                    print(f"{cur_img_path} not found!")
                    continue
                
                data_set[key] = {
                    "image": file_name,
                    "caption": sentence,
                    "label": label
                }
                cnt += 1
                    
        
        if mode in ["test", "val"]:
            f1= open(os.path.join(WORKING_PATH, self.text_name ,mode+".json"), 'r',encoding='utf-8')
            datas = json.load(f1)
            #for data in datas:
            for key, data in datas.items():
                # image = data['image']
                # sentence = data['caption']
                # label = data['label']
                file_name = data['image']
                sentence = data['caption']
                label = label_mapping[data['label']]

                # if os.path.isfile(os.path.join(WORKING_PATH, "dataset_image/dataset_image", str(image)+".jpg")):
                #     data_set[int(image)]={"caption":sentence, 'label': label}
                #     cnt += 1
                cur_img_path = os.path.join(WORKING_PATH, "dataset_image/dataset_image", str(file_name))
                if not os.path.exists(cur_img_path):
                    print(f"Image {file_name} not found.")
                    continue
                
                data_set[key] = {
                    "image": file_name,
                    "caption": sentence,
                    "label": label
                }
                cnt += 1
        return data_set


    def image_loader(self,id):
        return Image.open(self.data[id]["image_path"])
    def text_loader(self,id):
        return self.data[id]["caption"]


    def __getitem__(self, index):
        id=self.image_ids[index]
        text = self.text_loader(id)
        image_feature = self.image_loader(id)
        label = self.data[id]["label"]
        return text, image_feature, label, id

    def __len__(self):
        return len(self.image_ids)
    @staticmethod
    def collate_func(batch_data):
        batch_size = len(batch_data)
 
        if batch_size == 0:
            return {}

        text_list = []
        image_list = []
        label_list = []
        id_list = []
        for instance in batch_data:
            text_list.append(instance[0])
            image_list.append(instance[1])
            label_list.append(instance[2])
            id_list.append(instance[3])
        return text_list, image_list, label_list, id_list