from pathlib import Path

data_path = r'E:\Python\TrAdaboost\canruntr\MyTrAdaBoost\labelData\AllDATA\TransferData'
data_path = Path(data_path)
for phase_name in ['training', 'validation', 'test']:
    print(data_path / f"{phase_name}.txt")

if __name__ == '__main__':
    path = r'F:\test\test.txt'
    with open(path,'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split()  # 去掉列表中每一个元素的换行符
            # print(line[0])
            # print(line[1])
            # print(line[2])
            # print(line[3])
            print(line[4])
