import chardet
import argparse
import os

# TODO add parser for command-line options


def transformer(filepath, to_encode='utf-8', from_encode=None, backup_path=None):
    with open(filepath, 'rb+') as fin:
        data = fin.read()
        if from_encode is None:
            from_encode = chardet.detect(data)['encoding']
            print("[INFO] Detected file {}'s encoding is {}".format(filepath, from_encode))
        fin.seek(0)
        fin.truncate()
        fin.write(data.decode(from_encode).encode(to_encode))
        if backup_path is None:
            backup_path = filepath + '.backup'
        with open(backup_path, 'wb') as fout:
            fout.write(data)
    print("[INFO] Transformed file {}'s encoding from {} to {}, backed up as {}.".format(filepath, from_encode, to_encode, backup_path))


if __name__ == "__main__":
    for file in os.listdir('G:/study/毕业论文/论文/chapters'):
        if file.split('.')[-1] == 'tex':
            transformer('G:/study/毕业论文/论文/chapters/' + file, 'utf-8', 'GB2312')
