import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_detections(detections_dict):
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.clf()
    plt.figure(figsize=(3 * len(detections_dict), 3))
    for pid, title in enumerate(detections_dict.keys()):
        input_img, detections, model_img_size, classes, eps = detections_dict[title]
        if len(input_img.shape) == 4:
            input_img = input_img[0]
        plt.subplot(1, len(detections_dict), pid + 1)
        plt.title(title)
        plt.imshow(input_img)
        current_axis = plt.gca()
        for box in detections:
            xmin = max(int(box[-4] * input_img.shape[1] / model_img_size[1]), 0)
            ymin = max(int(box[-3] * input_img.shape[0] / model_img_size[0]), 0)
            xmax = min(int(box[-2] * input_img.shape[1] / model_img_size[1]), input_img.shape[1])
            ymax = min(int(box[-1] * input_img.shape[0] / model_img_size[0]), input_img.shape[0])
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='small', color='black', bbox={'facecolor': color, 'alpha': 1.0})
        plt.axis('off')
    plt.show()

def save_detections(detections_dict):

    for pid, title in enumerate(detections_dict.keys()):
        input_img, model_img_size, classes, file, eps, type, img_type = detections_dict[title]

    save_path = "/home/canadyre/data/VOCdevkit/VOC_"+str(img_type)+"_"+str(type)+'_'+title+str(eps)+'/'
    txt_path = "/home/canadyre/data/VOC_"+str(img_type)+"_"+str(type)+'_'+title+str(eps)+".txt"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    ## Save image locations in a text file for YOLO training
    with open(txt_path, 'a') as file1:
        file1.write(save_path+file+'\n')
        file1.close()

    input_img.save(save_path+file)
