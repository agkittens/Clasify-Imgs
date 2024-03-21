import os
import constants
import cv2
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess
import matplotlib.pyplot as plt


class Clasifier:

    def __init__(self):
        self.resnet50 = ResNet50(weights='imagenet', pooling='avg', include_top=False)

        self.status = False

        self.path = 'Final_images_dataset'
        self.img_paths = self.make_paths()
        self.img_feat = self.load_features()
        self.clusters = 0
        self.nums = None
        self.kmeans = None


    def make_paths(self):
        img_paths = []
        folder = os.listdir(self.path)

        for file in folder:
            merged = os.path.join(self.path, file)
            img_paths.append(merged)

        return img_paths


    def load_features(self):
        img_feat  = []
        features_file = "features.npy"

        if not constants.EXTRACT_FEAT:
            img_feat = np.load(features_file)

        else:
            for path in self.img_paths:

                img = self.process_img(path)
                feat_R = self.resnet(img)
                # feat_I = self.inception(img)
                # feat_V = self.vgg(img)


                # all = np.concatenate([feat_R, feat_I, feat_V])
                # img_feat.append(all)
                img_feat.append(feat_R)

            np.save(features_file, img_feat)

        return np.array(img_feat)




    def make_categories(self):
        self.kmeans = KMeans(n_clusters=self.clusters, n_init=100, random_state=constants.RANDOM1, algorithm="elkan")
        self.nums = self.kmeans.fit_predict(self.img_feat)

        for i in range(self.clusters):
            temp = self.nums.tolist()
            img_num = temp.count(i)

            folder = i

            if constants.SAVE_IMGS:
                for j, image_path in enumerate(self.img_paths):
                    if self.nums[j] == i:
                        img = cv2.imread(image_path)
                        self.save(folder, img, j)


            if constants.SHOW_IMGS:
                fig = plt.figure(figsize=(7, 7))

                for j, image_path in enumerate(self.img_paths):
                    if self.nums[j] == i:
                        img = cv2.imread(image_path)
                        self.show_clusters(img, img_num, fig)
                        img_num -= 1

        plt.show()



    def resnet(self,img):
        model = self.resnet50
        arr_expanded = np.expand_dims(img, axis=0)
        arr_preprocess = preprocess(arr_expanded)
        features = model.predict(arr_preprocess)

        return features.flatten()


    def save(self, folder_num, image, name):
        path = str(os.path.join(f'{folder_num}', f'wno{name}.jpg'))

        if os.path.isfile(path):
            os.remove(path)

        cv2.imwrite(path, image)


    def show_clusters(self, img, img_num, fig):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig.add_subplot(5,5, img_num)
        plt.imshow(img)
        plt.axis('off')


    def process_img(self,path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (constants.SIZE, constants.SIZE))
        img = np.array(img)

        return img


    def find_different_imgs(self):
        self.kmeans = KMeans(n_clusters=self.clusters, n_init=100, random_state=constants.RANDOM2, algorithm="elkan")
        self.nums = self.kmeans.fit_predict(self.img_feat)

        centroids = self.kmeans.cluster_centers_

        dist_values = self.find_indices(centroids)
        indices = np.argsort(dist_values)[-8:]

        folder = 'outliners'

        num = len(indices)

        fig = plt.figure(figsize=(5, 5))

        for idx in indices:
            img = cv2.imread(self.img_paths[idx])

            if constants.SAVE_IMGS:
                self.save(folder, img, num)

            self.show_clusters(img, num, fig)
            num = num - 1
        plt.show()


    def find_indices(self, centroids):
        shape = self.img_feat.shape
        inverted_arr = np.reshape(self.img_feat, [shape[0],1,shape[1]])

        sum = np.sum((inverted_arr - centroids) ** 2, axis=2)
        distances = np.sqrt(sum)

        min_values = np.min(distances, axis=1)


        return min_values


