from PIL import Image
import numpy as np


def letterbox_image_padded(image, size=(416, 416)):
    """ Resize image with unchanged aspect ratio using padding """
    image_copy = image.copy()
    iw, ih = image_copy.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image_copy = image_copy.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0, 0, 0))
    new_image.paste(image_copy, ((w - nw) // 2, (h - nh) // 2))
    new_image = np.asarray(new_image)[np.newaxis, :, :, :] / 255.
    meta = ((w - nw) // 2, (h - nh) // 2, nw + (w - nw) // 2, nh + (h - nh) // 2, scale)

    return new_image, meta, iw, ih

def remove_letterbox_image_padded(image, meta, size, in_size=(416, 416)):
    """ Revert image back to normal size """
    l, b, r, t, scale = meta[0], meta[1], meta[2], meta[3], meta[4]
    nw = size[0]
    nh = size[1]

    image_copy = image.copy()
    image_copy = Image.fromarray(np.uint8(image_copy[0]*255), 'RGB')
    new_image = image_copy.crop((l, in_size[1] - t, r, in_size[1]-b))
    new_image = new_image.resize((nw,nh), Image.BICUBIC)

    return new_image
