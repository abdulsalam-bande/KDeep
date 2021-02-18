from keras.models import Model, load_model
import os
import json
import numpy as np
import csv

if __name__ == '__main__':
    data_dir = "../dataset"
    test_data_dir = os.path.join(data_dir, 'test_data')
    pdbmap = json.load(open(os.path.join(test_data_dir, 'pdbmap.json'), 'r'))
    model = load_model('experiments/vggnet4/model.h5')  # weights.10-2.15.ckpt')
    results = []
    for k, v in pdbmap.items():
        voxels = np.array(
            [np.load(os.path.join(test_data_dir, 'id_' + str(v - i) + '.npy')) for i in range(24 * 7)])
        pred = model.predict(voxels).reshape(-1)
        mean_pred = np.mean(pred)
        mean_pred = round(mean_pred, 3)
        results.append([k, mean_pred])
        print("{}: {}".format(k, mean_pred))

    with open('predictions.txt', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)
