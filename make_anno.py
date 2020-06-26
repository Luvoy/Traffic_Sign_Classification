import pandas as pd
import os
import cv2

from hyper_parameters import ROOT
from hyper_parameters import PHASE
from hyper_parameters import NUM_CLASSES
from hyper_parameters import DATA_INFO
from hyper_parameters import ANNOTATION_FILE_NAME

# f = open('img_size_info.txt', "w")

for p in PHASE:
    file_count = 0
    for c in range(NUM_CLASSES):
        zero_prefix = "0000" if c < 10 else "000"

        data_dir = ROOT + os.sep + p + os.sep + zero_prefix + str(c)

        file_names_in_dir = os.listdir(data_dir)
        file_count += len(file_names_in_dir)
        for file_name in file_names_in_dir:
            file_path = os.path.join(data_dir, file_name)
            try:
                img = cv2.imread(file_path)
                # temp_shape = img.shape
                # ratio = max(temp_shape[0], temp_shape[1]) / min(
                #     temp_shape[0], temp_shape[1])
                # f.write(str(temp_shape) + "---" + str(ratio) + "\n")
            except OSError:
                print(OSError)
            else:
                DATA_INFO[p]['image_path'].append(file_path)
                DATA_INFO[p]['class_id'].append(c)

    annotation = pd.DataFrame(DATA_INFO[p])
    annotation.to_csv(ANNOTATION_FILE_NAME % p)
    print((ANNOTATION_FILE_NAME + ' file is saved.') % p)
    print('%s file count: %d' % (p, file_count))

# f.close()
