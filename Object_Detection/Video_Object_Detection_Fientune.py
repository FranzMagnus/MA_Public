import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.optim.adam
import torch.utils.data as data

from pycocotools.coco import COCO

import torchvision
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from helpers.engine import train_one_epoch, evaluate, eval_forward

import tqdm

from pickle import load, dump

from bidict import bidict

#Merging Duplicate Keys in a Dictionary
def merge_keys(ds):
    keys = set.union(*map(set, ds))
    return keys

path_base = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data', 'data_sorted')

coco_annotations_file_root = os.path.join(path_base,'labels', 'COCO')
coco_images_dir_root = os.path.join(path_base, 'frames')

#Creating a Mapping from Annotated Classes to Labels. Removes empty Classes
dict_list = []

for j in os.listdir(coco_images_dir_root):
    coco_annotations_file = os.path.join(coco_annotations_file_root, j+'.json')
    coco = COCO(coco_annotations_file)
    count_dict = {}
    classes = []
    for x in coco.cats:
        classes.append(coco.cats[x]['name'])
    
    for x in classes:
        catIds = coco.getCatIds(catNms=x) 
        imgIds = coco.getImgIds(catIds=catIds)
        for y in catIds:
            imgIds = coco.getImgIds(catIds=y)
            if len(imgIds) > 0:
                count_dict[x] =  len(imgIds)

    dict_list.append(count_dict)

keys = merge_keys(dict_list)
name_dict = dict.fromkeys(keys, 0)
name_dict.update((k, i+1) for i, k in enumerate(name_dict))
name_dict['No Annotation'] = len(name_dict)+1
names_to_annotation = bidict(name_dict)

with open(os.path.dirname(os.getcwd()) + "\\Object_Detection\\dataloaders\\names_to_annotation.pkl", 'wb') as file:
    dump(names_to_annotation, file)
    
#Dataset for Annotations, used for filtering unannotated Images
class AnnotationDataset(torch.utils.data.Dataset):
    from Video_Object_Detection_Fientune import names_to_annotation

    def __init__(self, root, annotation):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        
        #Add emptiness to not annotated frames
        if not boxes:
            boxes.append([0,0,1280,720])


        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        labels = []
        for i in range(num_objs):
            labels.append(coco_annotation[i]['category_id'])
        
        if not labels:
            labels.append(110)

        new_labels = []
        for x in labels:
            cat = coco.loadCats(x)
            name = cat[0]['name']
            new_labels.append(names_to_annotation.get(name))
        
        
        labels = torch.as_tensor(new_labels, dtype=torch.int64)

        # img_id
        img_id = int(img_id)

        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
            
        
        if not areas:
            areas.append((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]))
        
        #areas = torch.as_tensor(areas, dtype=torch.float32)
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        if iscrowd.numel() == 0:
            iscrowd = torch.zeros((1,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["coco"] = self.coco
        my_annotation["root"] = self.root
        my_annotation["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=[720, 1280])
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        return my_annotation

    def __len__(self):
        return len(self.ids)

#Dataset for Training the Model. Returns (input, target)/(img, annotations)
class TrainingsDataset(torch.utils.data.Dataset):
    def __init__(self, labels, transforms=None):
        self.labels = labels
        self.transforms = transforms
    

    def __getitem__(self, index):
        coco = self.labels[index]["coco"]
        root = self.labels[index]["root"]
        img_id = self.labels[index]["image_id"]
        path = coco.loadImgs(img_id)[0]['file_name']
        
        img = read_image(os.path.join(root, path))
        img = tv_tensors.Image(img)
        
        my_annotation = {}
        my_annotation["boxes"] = self.labels[index]["boxes"]
        my_annotation["labels"] = self.labels[index]["labels"]
        my_annotation["image_id"] = img_id
        my_annotation["area"] = self.labels[index]["area"]
        my_annotation["iscrowd"] = self.labels[index]["iscrowd"]
       
        if self.transforms is not None:
            img, my_annotation = self.transforms(img, my_annotation)

        return img, my_annotation

    def __len__(self):
        return len(self.labels)

#Stops Training early if no significant reduction of loss occurred
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def collate_fn(batch):
    return tuple(zip(*batch))

#Flips the Image and Annotations horizontal/vertical/both
def get_transform(train, h_flip, v_flip):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(h_flip))
        transforms.append(T.RandomVerticalFlip(v_flip))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

#Instantiate the pretrained model
def get_model_instance_segmentation(num_classes):
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

#Compute Evaluation loss
def evaluate_loss(model, data_loader, device):
    val_loss = 0
    with torch.no_grad():
      for images, targets in data_loader:
          images = list(image.to(device) for image in images)
          targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
          loss_dict, detections = eval_forward(model, images, targets)
         
          losses = sum(loss for loss in loss_dict.values())

          val_loss += losses
          
    validation_loss = val_loss/ len(data_loader)    
    return validation_loss

#Returns TRUE for manually selected images with no information. Returns FALSE for all other Images
def is_empty(dict):
    root = dict['root']
    id = dict['image_id']
    if 'p1_v1' in root:
        if id >= 4960 and id <= 5026:
            return True
        elif id >= 5433 and id <= 5530:
            return True
        elif id >= 5699 and id <= 5745:
            return True
        elif id >= 6360 and id <= 6494:
            return True
        elif id >= 7359 and id <= 7617:
            return True
        else:
            return False
    elif 'p1_v2' in root:
        if id >= 961 and id <= 964:
            return True
        elif id >= 974 and id <= 987:
            return True
        elif id >= 3494 and id <= 3578:
            return True
        else:
            return False
    elif 'p1_v3' in root:
        if id >= 1177 and id <= 1189:
            return True
        elif id >= 2630 and id <= 2658:
            return True
        else:
            return False
    elif 'p14_v1' in root:
        if id >= 1 and id <= 98:
            return True
        elif id >= 726 and id <= 749:
            return True
        elif id >= 1097 and id <= 1205:
            return True
        else:
            return False
    if 'p14_v2' in root:
        if id >= 477 and id <= 614:
            return True
        elif id >= 1404 and id <= 1449:
            return True
        elif id >= 2076 and id <= 2139:
            return True
        elif id >= 2211 and id <= 2262:
            return True
        else:
            return False
    if 'p14_v3' in root:
        if id >= 1 and id <= 21:
            return True
        elif id >= 2079 and id <= 2097:
            return True
        elif id >= 3542 and id <= 3713:
            return True
        elif id >= 5816 and id <= 6214:
            return True
        elif id >= 7709 and id <= 7938:
            return True
        else:
            return False
    if 'p14_v4' in root:
        if id >= 2205 and id <= 2331:
            return True
        else:
            return False
    elif 'p14_v5' in root:
        if id >= 5718 and id <= 5950:
            return True
        elif id >= 6215 and id <= 6308:
            return True
        elif id >= 6987 and id <= 7070:
            return True
        else:
            return False
    else:
        return False

# Compute if Changes to the dataset

ds_list = []
for j in os.listdir(coco_images_dir_root):
    if 'p1_v4' in j:
        print('skipped')
    else:
        coco_annotations_file = os.path.join(coco_annotations_file_root, j+'.json')
        coco_images_dir = os.path.join(coco_images_dir_root, j)
        ds_list.append(AnnotationDataset(root=coco_images_dir, annotation=coco_annotations_file))

concat_ds_list = data.ConcatDataset(ds_list)     

annotated_labels = []

for x in tqdm.tqdm(concat_ds_list):
    dict = x
    if len(dict['labels']) > 1:
            annotated_labels.append(dict)
    elif len(dict['labels']) == 1:
        y = dict['labels'].item()
        if  names_to_annotation.inv.get(y) != "No Annotation":
            annotated_labels.append(dict)
        else:
            if is_empty(dict):
                annotated_labels.append(dict)
        
with open(os.path.dirname(os.getcwd()) + "\\Object_Detection\\dataloaders\\labels.pkl", 'wb') as file:
        dump(annotated_labels, file)


# Load last computeted Dataset
'''
with open(os.path.dirname(os.getcwd()) + "\\Object_Detection\\dataloaders\\labels.pkl", "rb") as input_file:
   annotated_labels = load(input_file)
'''

# Create TrainingsDataset with four Augmentations: Normal, horizontal flip, vertical flip, horizontal & vertical flip 
annotated_ds = []
annotated_ds.append(TrainingsDataset(annotated_labels, transforms=get_transform(train=True, h_flip=0, v_flip=0)))
annotated_ds.append(TrainingsDataset(annotated_labels, transforms=get_transform(train=True, h_flip=1, v_flip=0)))
annotated_ds.append(TrainingsDataset(annotated_labels, transforms=get_transform(train=True, h_flip=0, v_flip=1)))
annotated_ds.append(TrainingsDataset(annotated_labels, transforms=get_transform(train=True, h_flip=1, v_flip=1)))

concat_ds = data.ConcatDataset(annotated_ds)

#Setup: Train-Validation-Split & Dataloaders
generator1 = torch.Generator().manual_seed(99)
annotated_ds_train, annotated_ds_test = data.random_split(concat_ds, [0.8, 0.2], generator=generator1)

train_batch_size = 8

annotated_data_loader = data.DataLoader(annotated_ds_train,
                            batch_size=train_batch_size,
                            shuffle=True,
                            num_workers=0,
                            collate_fn= collate_fn)

annotated_data_loader_test = data.DataLoader(annotated_ds_test,
                                batch_size=train_batch_size,
                                shuffle=True,
                                num_workers=0,
                                collate_fn= collate_fn)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

#Setup: Training parameters

#annotated classes + Background
num_classes = len(name_dict)+1

model = get_model_instance_segmentation(num_classes)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(
    params,
    lr=0.0001,
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)

num_epochs = 50
early_stopper = EarlyStopper(patience=3, min_delta=0.001)

#For saving Training results
logger_dict = {}
evaluator_dict = {}
val_loss_dict = {}

#Training
for epoch in range(num_epochs):
    logger_dict[epoch] = train_one_epoch(model, optimizer, annotated_data_loader, device, epoch, print_freq=100)
    lr_scheduler.step()
    validation_loss  = evaluate_loss(model, annotated_data_loader_test, device=device)
    val_loss_dict[epoch] = validation_loss
    print("Validation Loss:", validation_loss)
    evaluator_dict[epoch] = evaluate(model, annotated_data_loader_test, device=device)
    if early_stopper.early_stop(validation_loss.item()):             
        break

#Save Training Outputs by Epoch
torch.save(model, os.path.dirname(os.getcwd()) + "\\Object_Detection\\Models\\model_whole.pt")
torch.save(model.state_dict(), os.path.dirname(os.getcwd()) + "\\Object_Detection\\Models\\model_states.pt")

with open(os.path.dirname(os.getcwd()) + "\\Object_Detection\\training_output\\logger.pkl", 'wb') as file:
    dump(logger_dict, file)

with open(os.path.dirname(os.getcwd()) + "\\Object_Detection\\training_output\\lval_loss.pkl", 'wb') as file:
    dump(val_loss_dict, file)
 
with open(os.path.dirname(os.getcwd()) + "\\Object_Detection\\training_output\\evluator.pkl", 'wb') as file:
    dump(evaluator_dict, file)

print("That's it!")