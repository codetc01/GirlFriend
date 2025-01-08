import json
import os

if __name__ == '__main__':
    train_json_path = 'D:/st-gcn-master/mydata/kinetics-skeleton/kinetics_train'
    val_json_path = 'D:/st-gcn-master/mydata/kinetics-skeleton/kinetics_val'
    test_json_path = 'D:/st-gcn-master/mydata/kinetics-skeleton/kinetics_test'

    output_train_json_path = 'D:/st-gcn-master/mydata/kinetics-skeleton/kinetics_train_label.json'
    output_val_json_path = 'D:/st-gcn-master/mydata/kinetics-skeleton/kinetics_val_label.json'
    output_test_json_path = 'D:/st-gcn-master/mydata/kinetics-skeleton/kinetics_test_label.json'
    #
    train_json_names = os.listdir(train_json_path)
    val_json_names = os.listdir(val_json_path)
    test_json_names = os.listdir(test_json_path)

    train_label_json = dict()
    val_label_json = dict()
    test_label_json = dict()


    for file_name in train_json_names:
        name = file_name.split('.')[0]
        json_file_path = '{}/{}'.format(train_json_path, file_name)
        json_file = json.load(open(json_file_path))

        file_label = dict()
        if len(json_file['data']) == 0:
            file_label['has_skeleton'] = False
        else:
            file_label['has_skeleton'] = True
        file_label['label'] = json_file['label']
        file_label['label_index'] = json_file['label_index']

        train_label_json['{}'.format(name)] = file_label

        print('{} success'.format(file_name))

    with open(output_train_json_path, 'w') as outfile:
        json.dump(train_label_json, outfile)

    for file_name in val_json_names:
        name = file_name.split('.')[0]
        json_file_path = '{}/{}'.format(val_json_path, file_name)
        json_file = json.load(open(json_file_path))

        file_label = dict()
        if len(json_file['data']) == 0:
            file_label['has_skeleton'] = False
        else:
            file_label['has_skeleton'] = True
        file_label['label'] = json_file['label']
        file_label['label_index'] = json_file['label_index']

        val_label_json['{}'.format(name)] = file_label

        print('{} success'.format(file_name))

    with open(output_val_json_path, 'w') as outfile:
        json.dump(val_label_json, outfile)

    for file_name in test_json_names:
        name = file_name.split('.')[0]
        json_file_path = '{}/{}'.format(test_json_path, file_name)
        json_file = json.load(open(json_file_path))

        file_label = dict()
        if len(json_file['data']) == 0:
            file_label['has_skeleton'] = False
        else:
            file_label['has_skeleton'] = True
        file_label['label'] = json_file['label']
        file_label['label_index'] = json_file['label_index']

        test_label_json['{}'.format(name)] = file_label

        print('{} success'.format(file_name))

        with open(output_test_json_path, 'w') as outfile:
            json.dump(test_label_json, outfile)
