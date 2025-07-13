
from ray_tracing_synthesis import SoSToTimeShiftTransformer, image_to_sos_map
from utils import remove_blind_region_psf0, remove_blind_region_psf7p5, remove_blind_region_minuspsf7p5
from PIL import Image
import numpy as np
import random
import time

def preparing_training_data(shared_lock_1, shared_lock_2, img_idx_tobe_loaded, shared_queue_time_lag, total_train_imgs, buff_len, test_data_parent_dir):
    transformer = SoSToTimeShiftTransformer()
    BATCH_GENERATING = 30

    while True:
        if len(img_idx_tobe_loaded) == 0:
            idx_list = list(range(total_train_imgs))
            random.shuffle(idx_list)
            with shared_lock_1:
                img_idx_tobe_loaded[:] = idx_list

        tmp_batch = []
        for _ in range(BATCH_GENERATING):
            if len(img_idx_tobe_loaded) == 0:
                break
            idx_curr = img_idx_tobe_loaded.pop()
            image = Image.open(f"{test_data_parent_dir}grayscale_cropped_{idx_curr}.JPEG")
            sos_map = image_to_sos_map(image)
            tmp_batch.append(
                (
                    np.array([
                        remove_blind_region_psf7p5(transformer.transform(sos_map, "7p5psf")),
                        remove_blind_region_psf0(transformer.transform(sos_map, "0psf")),
                        remove_blind_region_minuspsf7p5(transformer.transform(sos_map, "minus7p5psf")),
                    ]),
                    sos_map,
                )
            )
        if len(shared_queue_time_lag) <= buff_len:
            with shared_lock_2:
                shared_queue_time_lag.extend(tmp_batch)
        time.sleep(0.1)