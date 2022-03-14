import pandas as pd
def read_log(file_path):
    data = []
    with open(file_path) as f:
        while True:
            lines = f.readline()
            if not lines:
                break
            line = lines.split('|')
            data.append([
                line[0].split('Time:')[1].strip(),
                float(line[1].split(':')[1].strip().replace('%','')),
                int(line[2].split(':')[1].strip()),
                int(line[3].split(':')[1].strip()),
                float(line[4].split(':')[1].strip()),
                float(line[5].split(':')[1].strip())
            ])
    data = pd.DataFrame(data,columns=['time','finish','qty','n','rlack_list_i','score'])
    return data

if __name__ == 'main':
    print(read_log(file_path='logging.log'))
