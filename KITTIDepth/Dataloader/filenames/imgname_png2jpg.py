import os
from tqdm import tqdm


def renamefile(filename):
    """
    One can use this file to convert filename path
    """
    new_data_list = []
    with open(filename, 'r') as f:
        data_list = f.read().split('\n')

        print('Generating new data list..')
        for data in tqdm(data_list):
            if len(data) == 0:
                continue
            data_info = data.split(' ')

            #data_info[0] = data_info[0].replace('jpg', 'png')
            #data_info[1] = data_info[1].replace('jpg', 'png')
            for it, name in enumerate(data_info):
                data_info[it] = '/'.join(name.split('/')[1:])
            if data_info[2].find('extras') == -1:
                new_data_list.append(' '.join(data_info))

    with open(filename, 'w') as f:
        print('writing new data names..')

        for it, data in tqdm(enumerate(new_data_list)):
            if len(data) == 0:
                continue

            if it == len(new_data_list)-1:
                f.write(data)
            else:
                f.write(data+'\n')

        print('Done.')
