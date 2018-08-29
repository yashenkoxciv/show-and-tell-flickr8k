import os.path
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from zipfile import ZipFile
from nltk import word_tokenize


class Flickr8k:
    @staticmethod
    def _load_file_names(txt):
        return set(str(txt, encoding='ascii').split('\n')[:-1])
    
    @staticmethod
    def _load_captions_map(txt, lower):
        file_token = str(txt, encoding='ascii').split('\n')[:-1]
        cm = {}
        for t in file_token:
            parts = t.split('\t')
            caption = parts[1]
            image_file = parts[0].split('#')[0]
            if image_file not in cm:
                cm[image_file] = [caption.lower() if lower else caption]
            else:
                cm[image_file].append(caption.lower() if lower else caption)
        return cm
    
    def _make_vocab(self, vsize):
        text = ' '.join([' '.join(c5) for c5 in self.captions])
        tokens = word_tokenize(text)
        tkf = {}
        for token in tokens:
            if token not in tkf:
                tkf[token] = 1
            else:
                tkf[token] += 1
        vocab = sorted(tkf.keys(), key=lambda t: tkf[t])
        if vsize:
            vocab = vocab[-vsize:]
        vocab.insert(0, self.start)
        vocab.insert(0, self.end)
        vocab.insert(0, self.unknown)
        vocab.insert(0, self.absent)
        return vocab
    
    def _caption_to_sequence(self, c):
        unk_idx = self.vocab.index(self.unknown)
        end_idx = self.vocab.index(self.end)
        #abs_idx = self.vocab.index(self.absent)
        s = [self.vocab.index(self.start)]
        for t in word_tokenize(c):
            if t in self.vocab:
                s.append(self.vocab.index(t))
            else:
                s.append(unk_idx)
        s.append(end_idx)
        return s
    
    def _captions_to_sequences(self):
        seq = []
        for c5 in self.captions:
            seq.append([])
            for c in c5:
                seq[-1].append(self._caption_to_sequence(c))
        return np.array(seq)
    
    def sequence_to_onehot(self, s):
        #cs = s[1:] # + [self.vocab.index(self.absent)]
        #aw_abs = np.argwhere(cs == self.vocab.index(self.absent))
        #if aw_abs.size != 0:
        #    awm = np.min(aw_abs)
        #    cs = cs[:awm]
        s1h = np.zeros([len(s), len(self.vocab)])
        s1h[np.arange(len(s)), s] = 1
        #s1h[-1, self.vocab.index(self.absent)] = 1
        return s1h
    
    def onehot_sequence_to_string(self, s1h):
        s = []
        for t in s1h:
            s.append(self.vocab[np.argmax(t)])
        return ' '.join(s)
    
    def __init__(self,
                 images_zip=None, captions_zip=None, resize=None,
                 lower=True, vsize=None, start=None,
                 end=None, unknown=None, absent=None,
                 smaxlen=None, npz_file=None
                 ):
        if npz_file:
            self._load(npz_file)
        else:
            self.captions = []
            self.images = []
            self.train_idx, self.test_idx, self.valid_idx = [], [], []
            self.resize = resize
            self.start = start
            self.end = end
            self.unknown = unknown
            self.absent = absent
            self.smaxlen = smaxlen
            with ZipFile(captions_zip) as cz:
                train_files = Flickr8k._load_file_names(cz.read('Flickr_8k.trainImages.txt'))
                test_files = Flickr8k._load_file_names(cz.read('Flickr_8k.testImages.txt'))
                valid_files = Flickr8k._load_file_names(cz.read('Flickr_8k.devImages.txt'))
                captions_map = Flickr8k._load_captions_map(cz.read('Flickr8k.token.txt'), lower)
            with ZipFile(images_zip) as imgsz:
                fn = 0
                for n in imgsz.namelist():
                    if n.startswith('Flicker8k_Dataset/') and n.endswith('.jpg'):
                        img = Image.open(BytesIO(imgsz.read(n)))
                        if resize:
                            cimg = img.resize(resize)
                        else:
                            cimg = img
                        imga = np.array(cimg).flatten()
                        imgfile = n.split('/')[1]
                        if imgfile in train_files:
                            self.train_idx.append(fn)
                        elif imgfile in test_files:
                            self.test_idx.append(fn)
                        elif imgfile in valid_files:
                            self.valid_idx.append(fn)
                        else:
                            logging.warning(imgfile + ' skipped')
                            continue
                        self.images.append(imga)
                        self.captions.append(captions_map[imgfile])
                        fn += 1
            self.images = np.array(self.images)
            self.captions = np.array(self.captions)
            self.vocab = self._make_vocab(vsize)
            self.sequences = self._captions_to_sequences()
    
    def save(self, npz_file):
        np.savez(npz_file,
                 images=self.images,
                 captions=self.captions,
                 vocab=self.vocab,
                 sequences=self.sequences,
                 train_idx=self.train_idx,
                 test_idx=self.test_idx,
                 valid_idx=self.valid_idx,
                 resize=self.resize,
                 start=self.start,
                 end=self.end,
                 unknown=self.unknown,
                 absent=self.absent
        )
    
    def _load(self, npz_file):
        f = np.load(npz_file)
        self.images = f['images']
        self.captions = f['captions']
        self.vocab = list(f['vocab'])
        self.sequences = f['sequences']
        self.train_idx = f['train_idx']
        self.test_idx = f['test_idx']
        self.valid_idx = f['valid_idx']
        self.resize = f['resize']
        self.start = str(f['start'])
        self.end = str(f['end'])
        self.unknown = str(f['unknown'])
        self.absent = str(f['absent'])
    
    @property
    def voclen(self):
        return len(self.vocab)
    
    def next_batch(self, batch_size, idxs, permute):
        idxs = np.random.choice(idxs, batch_size)
        imgs_batch = self.images[idxs].reshape([batch_size] + list(self.resize) + [3]) / 256
        sqns_batch = self.sequences[idxs, np.random.randint(0, 5, batch_size)]
        s1h_batch = np.array([self.sequence_to_onehot(s) for s in sqns_batch])
        if permute:
            sqns_batch = self.sequences[idxs, np.random.randint(0, 5, batch_size)]
        return [imgs_batch, sqns_batch], s1h_batch


def plain_batch(f8k, batch_size, idxs, permute):
    x, y = f8k.next_batch(batch_size, idxs, permute)
    sqns = np.array([np.array(s) for s in x[1]])
    return [x[0], sqns], y

def crop_sequence(s, maxlen, absent_idx):
    if len(s) > maxlen:
        s = s[:maxlen]
    else:
        s.extend([absent_idx]*(maxlen - len(s)))
    return s

def crop_one_hot(onehot, maxlen, absent_idx, voclen):
    if onehot.shape[0] > maxlen:
        onehot = onehot[:maxlen]
    else:
        crop = np.zeros([maxlen - onehot.shape[0], voclen])
        if absent_idx is not None:
            crop[:, absent_idx] = 1
        onehot = np.concatenate([onehot, crop], axis=0)
    return onehot

def cropped_batch(f8k, batch_size, maxlen, pad_empty, idxs, permute):
    # pad_empty If True, then padding y with <abs>, otherwise pad it with zeros
    x, y = f8k.next_batch(batch_size, idxs, permute)
    raw_sqns = x[1]
    absent_idx = f8k.vocab.index(f8k.absent)
    cropped_sqns = [crop_sequence(s, maxlen, absent_idx) for s in raw_sqns]
    if pad_empty:
        cropped_onehot = [crop_one_hot(onehot, maxlen, absent_idx, f8k.voclen) for onehot in y]
    else:
        cropped_onehot = [crop_one_hot(onehot, maxlen, None, f8k.voclen) for onehot in y]
    return [x[0], np.array(cropped_sqns)], np.array(cropped_onehot)

def shifted_batch(f8k, batch_size, shift, idxs, permute):
    x, y = f8k.next_batch(batch_size, idxs, permute)
    sqns = [s[:-shift] for s in x[1]]
    onehot = [ih[shift:] for ih in y]
    return [x[0], np.array(sqns)], np.array(onehot)

def shifted_cropped_batch(f8k, batch_size, maxlen, pad_empty, shift, idxs, permute):
    x, y = cropped_batch(f8k, batch_size, maxlen+1, pad_empty, idxs, permute)
    sqns = [s[:-shift] for s in x[1]]
    onehot = [ih[shift:] for ih in y]
    return [x[0], np.array(sqns)], np.array(onehot)

def batch_generator(batch_emitter, **kwargs):
    def generator():
        while True:
            yield batch_emitter(**kwargs)
    return generator

# min caption length: 4, max: 40, mean: 14
def augmented_batch(f8k, batch_size, minlen, maxlen, idxs):
    wlen = np.random.randint(minlen, maxlen)
    x, y = f8k.next_batch(batch_size, idxs, False)
    absent_idx = f8k.vocab.index(f8k.absent)
    sqns = []
    onehot = []
    for i in range(x[1].shape[0]):
        s = x[1][i]
        h = y[i]
        if len(s) <= (wlen + 1):
            sqns.append(crop_sequence(s, wlen+1, absent_idx)[:-1])
            onehot.append(crop_one_hot(h, wlen+1, None, f8k.voclen)[1:])
        else:
            ri = np.random.randint(0, len(s)-wlen-1)
            sqns.append(s[ri:ri+wlen+1][:-1])
            onehot.append(h[ri:ri+wlen+1][1:])
        rvar = np.random.uniform()
        if rvar < 0.3:
            x[0][i] = np.rot90(x[0][i], np.random.randint(1, 3), axes=(0, 1))
        if rvar < 0.5:
            x[0][i] += np.clip(np.random.randn(*f8k.resize, 3), -0.1, 0.1)
    return [x[0], np.array(sqns)], np.array(onehot)

def parse_or_load(
        img_zip, cap_zip, resize, vocsize,
        start, end, unknown, absent, npz_file):
    if os.path.exists(npz_file):
        logging.info('Flickr8k loading')
        return Flickr8k(npz_file=npz_file)
    else:
        logging.info('Flickr8k should be parsed')
        f8k = Flickr8k(
                img_zip, cap_zip, resize, vsize=vocsize, start=start,
                end=end, unknown=unknown, absent=absent)
        f8k.save(npz_file)
        return f8k
        

def show_examples(f, idxs):
    for i in idxs:
        print('\n'.join(f.captions[i]))
        plt.imshow(f.images[i].reshape(list(f.resize) + [3]))
        plt.show()

def show_batch_example(f, n):
    x, y = f.next_batch(n, f.train_idx)
    imgs = x[0]
    for i in range(n):
        print(f.onehot_sequence_to_string(y[i]))
        plt.imshow(imgs[i])
        plt.show()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flickr8k parser')
    parser.add_argument('--imgzip', type=str, required=True, help='path to Flickr8k_Dataset.zip')
    parser.add_argument('--capzip', type=str, required=True, help='path to Flickr8k_text.zip')
    parser.add_argument('--npz', type=str, required=True, help='path to parsed .npz file')
    parser.add_argument('--vocsize', type=int, required=True, help='vocabulary size')
    parser.add_argument('--resize', type=int, required=True, help='resize')
    args = parser.parse_args()
    
    f8k = Flickr8k(
            args.imgzip, args.capzip, (args.resize, args.resize),
            True, args.vocsize, '<beg>',
            '<end>', '<unk>', '<abs>')
    f8k.save(args.npz)
    