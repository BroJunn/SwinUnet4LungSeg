import os, requests, json
from tqdm import tqdm
from typing import Union, List, Tuple
import cv2
from sklearn.model_selection import train_test_split

class PreprocessDataset:
    def __init__(self, all_img_json_dir_path, sem_seg_mask_dir_path, save_dir_path, train_test_split: List=[0.8, 0.1, 0.1], random_state=2024):
        self.all_img_json_dir_path = all_img_json_dir_path
        self.sem_seg_mask_dir_path = sem_seg_mask_dir_path
        self.train_test_split = train_test_split
        self.save_dir_path = save_dir_path
        self.random_state = random_state
        assert sum(train_test_split) == 1.0, "The ratio for splitting dataset must sum up to 1!"

        self.raw_img_save_dir_path = os.path.join(save_dir_path, 'raw_images')
        self.resized_img_save_dir_path = os.path.join(save_dir_path, 'inputs')
        self.resized_mask_save_dir_path = os.path.join(save_dir_path, 'GTs')

        self._check_id_correspondence()

        self.failure_download_ids = []

    def _check_id_correspondence(self):
        self.list_id_filename_all_images = os.listdir(self.all_img_json_dir_path)
        self.list_id_filename_mask = os.listdir(self.sem_seg_mask_dir_path)
        self.ids_all_images = [n.rstrip('.json') for n in self.list_id_filename_all_images]
        self.ids_mask = []
        for n in self.list_id_filename_mask:
            if n.endswith('.png'):
                self.ids_mask.append(n.rstrip('.png'))
        self.filtered_ids_all_images = []
        for id in self.ids_all_images:
            if id in self.ids_mask:
                self.filtered_ids_all_images.append(id)
        
        self.train_ids, val_test_ids = train_test_split(self.filtered_ids_all_images,
                                                        test_size=self.train_test_split[1]+ self.train_test_split[2],
                                                        random_state=self.random_state)
        self.val_ids, self.test_ids = train_test_split(val_test_ids,
                                                       test_size=self.train_test_split[2]/(self.train_test_split[1]+ self.train_test_split[2]),
                                                       random_state=self.random_state)

    def _read_json(self, id):
        with open(os.path.join(self.all_img_json_dir_path, id+'.json')) as f:
            data = json.load(f)
        return {
            'url': data['image']['url']
        }

    def _download_img_from_json(self, id, flag: str):
        url = self._read_json(id)['url']
        success_download = self._download_img(url, os.path.join(self.raw_img_save_dir_path, id + '.png'))

        if not success_download:
            self.failure_download_ids.append(id)

    def _download_img(self, url, save_file):
        res = requests.get(url)
        if res.status_code == 200:
            with open(save_file, 'wb') as s:
                s.write(res.content)
            return True
        else:
            print(f"CANNOT download image from url {url}")
            return False

    def download_all_iamges(self):
        for id in tqdm(self.filtered_ids_all_images):
            self._download_img_from_json(id)
        print(f"Download of raw images finishes: {len(self.filtered_ids_all_images)} images in total; {len(self.failure_download_ids)} requests failed.")

    def resize_all_img_n_mask(self, size: Union[Tuple, List]):

        for ids_list, flag in zip([self.train_ids, self.val_ids, self.test_ids], ['train', 'val', 'test']):
            os.makedirs(os.path.join(os.path.join(self.save_dir_path, flag), 'inputs'), exist_ok=True)
            os.makedirs(os.path.join(os.path.join(self.save_dir_path, flag), 'GTs'), exist_ok=True)
            for id in tqdm(ids_list):
                raw = cv2.imread(os.path.join(self.raw_img_save_dir_path, id+'.png'), cv2.IMREAD_GRAYSCALE)
                resized = cv2.resize(raw, size, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(os.path.join(os.path.join(self.save_dir_path, flag), 'inputs'), id+'.png'), resized)

                mask = cv2.imread(os.path.join(self.sem_seg_mask_dir_path, id+'.png'), cv2.IMREAD_GRAYSCALE)
                resized_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(os.path.join(os.path.join(self.save_dir_path, flag), 'GTs'), id+'.png'), resized_mask)

if __name__ == '__main__':
    A = PreprocessDataset(
        'covid-19-xray-dataset/annotations/all-images',
        'covid-19-xray-dataset/annotations/all-images-semantic-masks',
        'covid-19-xray-dataset/data'
    )
    A.download_all_iamges()
    A.resize_all_img_n_mask((768, 512))