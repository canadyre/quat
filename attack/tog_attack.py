from attack_utils.preprocessing import letterbox_image_padded, remove_letterbox_image_padded
from attack_utils.save_image import visualize_detections, save_detections
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import load_model
from attack_utils.yolov3 import YOLOv3_Darknet53
from attack_utils.frcnn import FRCNN
from attack_utils.ssd import SSD300
from attack_utils.ssd import SSD512
from PIL import Image
from attack_utils.attacks import *
import os
import csv
import argparse
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = argparse.ArgumentParser(description='YOLO v3 TOG Attack')
    parser.add_argument("--images", dest = 'images', help =
                        "Image / Directory containing images to perform detection upon",
                        default = './voc/2007_test.txt', type = str)
    parser.add_argument("--image_type", dest = 'image_type',
                        help = "train or test data",
                        default = 'train', type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--weights", dest = 'weightsfile', help =
                        "weightsfile",
                        default = "weights/frcnn.pth", type = str)
    parser.add_argument("--type", dest = 'type', help =
                        "untargeted, vanishing, fabrication",
                        default = "untargeted", type = str)
    parser.add_argument("--model_type", dest = 'model_type', help =
                        "yolo, ssd300, ssd512, frcnn",
                        default = "frcnn", type = str)
    parser.add_argument("--eps", dest = "eps", help = "Epsilon", default = 16)
    parser.add_argument("--eps_iter", dest = "eps_iter", help = "Attack Learning Rate", default = 2)
    parser.add_argument("--n_iter", dest = "n_iter", help = "Batch size", default = 5)
    return parser.parse_args()

args = arg_parse()


print("# GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

weights = args.weightsfile  # TODO: Change this path to the victim model's weights
images = args.images
conf_threshold = args.confidence
type = args.type
img_type = args.image_type

eps = int(args.eps) / 255.       # Hyperparameter: epsilon in L-inf norm
eps_iter = int(args.eps_iter) / 255.  # Hyperparameter: attack learning rate
n_iter = args.n_iter          # Hyperparameter: number of attack iterations

if args.model_type == 'yolo':
    detector = YOLOv3_Darknet53(weights=weights)
elif args.model_type == 'ssd300':
    detector = SSD300(weights=weights)
elif args.model_type == 'ssd512':
    detector = SSD512(weights=weights)
elif args.model_type == 'frcnn':
    detector = FRCNN().cuda(device=0).load(weights)

f = open(images, "r")
img_list = []
for line in f:
    stripped_line = line.strip()
    line_list = stripped_line.split()
    img_list.append(line_list)
f.close()

for x in img_list:

    fpath = x[0]
    fpath = fpath.replace('Dropbox', 'main_Dropbox')
    temp = fpath.split('/')
    filename = temp[-1]
    print(filename)
    input_img = Image.open(fpath)

    x_query, x_meta, w, h = letterbox_image_padded(input_img, size=detector.model_img_size)

    # Generation of the adversarial example
    if (type == 'untargeted'):

        x_adv_untargeted = tog_untargeted(victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter)
        x_adv_untargeted = remove_letterbox_image_padded(x_adv_untargeted, x_meta, (w,h), in_size=detector.model_img_size)
        save_detections({'TOG-untargeted': (x_adv_untargeted, detector.model_img_size, detector.classes, filename, int(args.eps), str(args.model_type), str(args.image_type))})

    elif (type == 'vanishing'):

        x_adv_untargeted = tog_vanishing(victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter)
        x_adv_untargeted = remove_letterbox_image_padded(x_adv_untargeted, x_meta, (w,h), in_size=detector.model_img_size)
        save_detections({'TOG-vanishing': (x_adv_untargeted, detector.model_img_size, detector.classes, filename, int(args.eps), str(args.model_type), str(args.image_type))})

    elif (type == 'fabrication'):

        x_adv_untargeted = tog_fabrication(victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter)
        x_adv_untargeted = remove_letterbox_image_padded(x_adv_untargeted, x_meta, (w,h), in_size=detector.model_img_size)
        save_detections({'TOG-fabrication': (x_adv_untargeted, detector.model_img_size, detector.classes, filename, int(args.eps), str(args.model_type), str(args.image_type))})

    elif (type == 'mislabel'):

        x_adv_untargeted = tog_fabrication(victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter)
        x_adv_untargeted = remove_letterbox_image_padded(x_adv_untargeted, x_meta, (w,h), in_size=detector.model_img_size)
        save_detections({'TOG-mislabeling': (x_adv_untargeted, detector.model_img_size, detector.classes, filename, int(args.eps), str(args.model_type), str(args.image_type))})
