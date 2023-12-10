from PIL import Image
from ultralytics import YOLO

import shutil
import os 
import yaml

from tqdm   import tqdm
from utils  import *


def organize():
    
    tsize   = 0.8 
    
    win     = 'I:/research/head/data/mix'
    linux   = '/home/rensso/heads/hollywoodheads-640x384'
    root    = win if u_whichOS()=='win' else linux

    win     = 'I:/research/head/data2'
    linux   = '/home/rensso/heads/data'
    data    = win if u_whichOS()=='win' else linux
    
    files   = u_listFileAll(root, 'jpeg')
    nfiles  = len(files)

    bound   = int(nfiles * tsize)

    print('image number ', nfiles)

    train   = files[:bound]
    val     = files[bound:]

    print('train number ', len(train))
    print('val number ', len(val))

    ## building directories....................................................
    
    images  = data + '/' + 'images'
    labels  = data + '/' + 'labels'
        
    i_train = images + '/' + 'train'
    i_val   = images + '/' + 'val'

    l_train = labels + '/' + 'train'
    l_val   = labels + '/' + 'val'

    u_mkdir(data)
    u_mkdir(images)
    u_mkdir(labels)
    u_mkdir(i_train)
    u_mkdir(i_val)
    u_mkdir(l_train)
    u_mkdir(l_val)

    ## copying files...........................................................

    for im in tqdm(train, 'train'):
        name = os.path.basename(im[:-5])
        lb   = im[:-4] + 'txt' 

        if os.path.isfile(lb):
            im_ = i_train + '/' + name + '.jpeg' 
            lb_ = l_train + '/' + name + '.txt'  

            shutil.copyfile(im, im_)
            shutil.copyfile(lb, lb_)


    for im in tqdm(val, 'val'):
        name = os.path.basename(im[:-5])
        lb   = im[:-4] + 'txt' 

        if os.path.isfile(lb):
            im_ = i_val + '/' + name + '.jpeg' 
            lb_ = l_val + '/' + name + '.txt'  

            shutil.copyfile(im, im_)
            shutil.copyfile(lb, lb_)

    ######## yaml file dump ...................................................

    info = {
        'path'  : data,
        'train' : 'images/train',
        'val'   : 'images/val',

        'names': {
            0: 'head'
            }
        }
        
    file=open("data.yaml","w")
    yaml.dump(info, file)
    file.close()


def yolotrain():
    model   = YOLO('yolov8n.pt')
    results = model.train(data='data.yaml', epochs=15, batch=8)


##############################################################
if __name__ == '__main__':
    #organize()
    yolotrain()

# Load a pretrained YOLOv8n model
#model = YOLO('yolov8n.pt')


#results = model.train(data='conf.yaml', epochs=5)



# Run inference on 'bus.jpg'
#results = model("I:/research/head/data/imgs/mov_001_007588.jpeg")  # results list

# Show the results
#for r in results:
#    im_array = r.plot()  # plot a BGR numpy array of predictions
#    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#    im.show()  # show image
#    im.save('results.jpg')  # save image

    ...


