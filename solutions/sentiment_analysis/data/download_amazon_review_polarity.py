import numpy as np
import pandas as pd
import os.path
import wget

AZ_ACC = "amazonsentimenik"
AZ_CONTAINER = "textclassificationdatasets"
ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
print("Alphabet %d characters: " % len(ALPHABET), ALPHABET)
FEATURE_LEN = 1014


def download_file(url):
    local_filename = url.split('/')[-1]
    if os.path.isfile(local_filename):
        print("The file %s already exist in the current directory" % local_filename)
    else:
        print('downloading data: %s' % url)
        response = wget.download(url)
        print('saved data')


def create_features(infile, outprefix):
    """
    If we have 100 reviews to create features from, this will output a y-vector
    that is 100 x 1 (containing 1 or 0 for positive and negative review), and
    a 100 x 1014 (character/padding) x 69 (dimensions) x-tensor.

    The x-tensor will be saved with a character (vector) per line, so it will
    need to be parsed every 1014 characters (equals one review).

    For example the first 4 lines = first 4 characters:
    000000000000000000000000000000000000000000001000000000000000000000000
    000000000000000000000000000000000000000001000000000000000000000000000
    000000000000000000000000000000000000000000000000000000000001000000000
    000000100000000000000000000000000000000000000000000000000000000000000"""

    # Get data from windows blob
    download_file('https://%s.blob.core.windows.net/%s/%s' % (AZ_ACC, AZ_CONTAINER, infile))
    # load data into dataframe
    df = pd.read_csv(infile,
                     header=None,
                     names=['sentiment', 'summary', 'text'])

    # concat summary, review; trim to 1014 char; reverse; lower
    df['rev'] = (df.summary + " " + df.text).str[:FEATURE_LEN].str[::-1].str.lower()
    df.drop(['text', 'summary'], axis=1, inplace=True)
    df.sentiment -= 1
    Y_split = np.asarray(df.sentiment, dtype='int')

    # character-encoding
    voc_hash = np.identity(len(ALPHABET))
    df_hash = pd.DataFrame(voc_hash, columns=ALPHABET)
    X_split = np.zeros([df.shape[0], FEATURE_LEN, len(ALPHABET)], dtype='int')

    print('creating character encoding')
    # create character tensor
    for ti, tx in enumerate(df.rev):
        if (ti + 1) % 100000 == 0:
            print("%d rows processed" % ti)
        chars = list(tx)
        for ci, ch in enumerate(chars):
            if ch in ALPHABET:
                X_split[ti][ci] = np.array(df_hash[ch])

    # Save to disk in universal format
    # Assume we have 5 reviews
    # Y -> 5 x 1
    print("Saving y list ...")
    np.savetxt(outprefix + 'y.csv', Y_split, fmt='%i')
    print(Y_split[:2])

    # 3D tensor i.e. 5 by 1014 = 5070
    print("Saving x tensor ...")
    with open(outprefix + 'x.csv', 'wb') as outfile:
        for data_slice in X_split:
            # print(data_slice.shape) # (1014, 69)
            np.savetxt(outfile, data_slice, fmt='%i'*len(ALPHABET))
    print(X_split[:2])

if __name__ == '__main__':
    create_features(infile='amazon_review_polarity_test.csv', outprefix='amazon_test_')
    print("saved test set")
    create_features(infile='amazon_review_polarity_train.csv', outprefix='amazon_train_')
    print("saved train set")
