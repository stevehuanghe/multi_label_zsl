"""
Adapted from PyTorch's text library.
https://github.com/rowanz/neural-motifs/blob/master/lib/word_vectors.py
modified by He Huang
"""

import array
import os
import zipfile

import six
import torch
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm

import sys

def get_word_vectors(names, wv_type='glove.840B', wv_dir='./data/glove', wv_dim=300, random=True, multi_avg=True, sep=" "):
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

    if random:
        vectors = torch.Tensor(len(names), wv_dim)
        vectors.normal_(0,1)
    else:
        vectors = torch.zeros([len(names), wv_dim])
        
    print("Loading word embeddings...")
    for i, token in enumerate(names):
        wv_index = wv_dict.get(token, None)
        joined_token = ''.join(token.split(" "))
        joined_token2 = ''.join(token.split("_"))
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        elif wv_dict.get(joined_token, None) is not None:
            wv_index = wv_dict.get(joined_token, None)
            vectors[i] = wv_arr[wv_index]
        elif wv_dict.get(joined_token2, None) is not None:
            wv_index = wv_dict.get(joined_token2, None)
            vectors[i] = wv_arr[wv_index]
        elif multi_avg:
            words = token.split(" ")
            if len(words) < 2:
                words = token.split("_")
            embed = torch.zeros(wv_dim)
            cnt = 0
            for word in words:
                wv_index = wv_dict.get(word, None)
                if wv_index is not None:
                    embed += wv_arr[wv_index]
                    cnt += 1
            if cnt > 0:
                embed /= cnt
                vectors[i] = embed
            else:
                print("Fail on {}".format(token))
        else:
            # Try the longest word (hopefully won't be a preposition
            lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
            print("{} -> {} ".format(token, lw_token))
            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                print("Fail on {}".format(token))
    print("Finish loading word embeddings")
    return vectors

URL = {
        'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
        }


def load_word_vectors(root, wv_type, dim):
    """Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    fname = os.path.join(root, wv_type + '.' + dim)
    if os.path.isfile(fname + '.pt'):
        fname_pt = fname + '.pt'
        print('Loading word vectors from', fname_pt)
        try:
            return torch.load(fname_pt)
        except Exception as e:
            print("""
                Error loading the model from {}

                This could be because this code was previously run with one
                PyTorch version to generate cached data and is now being
                run with another version.
                You can try to delete the cached files on disk (this file
                  and others) and re-running the code

                Error message:
                ---------
                {}
                """.format(fname_pt, str(e)))
            sys.exit(-1)
    if os.path.isfile(fname + '.txt'):
        fname_txt = fname + '.txt'
        cm = open(fname_txt, 'rb')
        cm = [line for line in cm]
    elif os.path.basename(wv_type) in URL:
        url = URL[wv_type]
        print('Downloading word vectors from {}'.format(url))
        filename = os.path.basename(fname)
        if not os.path.exists(root):
            os.makedirs(root)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fname, _ = urlretrieve(url, fname, reporthook=reporthook(t))
            with zipfile.ZipFile(fname, "r") as zf:
                print('Extracting word vectors into {}'.format(root))
                zf.extractall(root)
        if not os.path.isfile(fname + '.txt'):
            raise RuntimeError('No word vectors of requested dimension found')
        return load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError('Unable to load word vectors')

    wv_tokens, wv_arr, wv_size = [], array.array('d'), None
    if cm is not None:
        for line in tqdm(range(len(cm)), desc="Loading word vectors from {}".format(fname_txt)):
            entries = cm[line].strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('Non-UTF8 token', repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
    ret = (wv_dict, wv_arr, wv_size)
    torch.save(ret, fname + '.pt')
    return ret

def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optionala
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner
