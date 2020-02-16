import chardet

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
    s = '这是测试字符串'
    s_gbk = s.encode('GBK')
    s_utf8 = s.encode('utf-8')
    print(s, s_gbk, s_utf8)
    test_path = 'test.txt'
    with open(test_path, 'wb') as f:
        f.write(s_gbk)
    transformer(test_path, 'utf-8', 'gbk')
    with open(test_path + '.backup', 'r', encoding='gbk') as f:
        print(f.read())
    with open(test_path, 'r', encoding='utf-8') as f:
        print(f.read())