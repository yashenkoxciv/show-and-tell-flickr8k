from flickr8k import parse_or_load

NPZ_FILE = 'flickr8k.npz'
IMG_ZIP = '/home/ahab/dataset/flickr8k/Flickr8k_Dataset.zip'
CAP_ZIP = '/home/ahab/dataset/flickr8k/Flickr8k_text.zip'

RESIZE = (128, 128)

VOCSIZE = 6996
START = '<beg>'
END = '<end>'
UNKNOWN = '<unk>'
ABSENT = '<abs>'

BATCH_SIZE = 10
EPOCHS = 10
BATCHES = int((8000*5) / BATCH_SIZE)

MAXLEN = 20
MINLEN = 1
PAD_EMPTY = False # pad y with full of zeros vectors
SHIFT = 1
PERMUTE = False

E_DIM = 100 # dimensions of word embeddings
Z_DIM = 100 # dimensions of image embeddings

def get_flickr8k():
    return parse_or_load(IMG_ZIP, CAP_ZIP, RESIZE, VOCSIZE,
                         START, END, UNKNOWN, ABSENT, NPZ_FILE)

