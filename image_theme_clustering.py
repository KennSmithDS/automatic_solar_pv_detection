import pandas as pd
import numpy as np
import traceback, time
from tqdm import tqdm
import os, math, argparse, sys
import rasterio as rio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import multiprocessing as mp
from multiprocessing import Pool
import cv2, shutil

class TifImageClusterer:
    def __init__(self):
        self.folder_path = ''
        self.n_clusters = 2
        self.tif_list = []
        self.flat_contours = []
        self.zero_matrix = []
        self.X = None
        self.Y = None
        self.tif_df = pd.DataFrame()
        # self.pbar = 0

    def get_timer(self):
        print(self.timer, "\n")
        self.timer += 1

    def set_folder_path(self, folder_path):
        self.folder_path = folder_path

    def set_kmeans(self, n_clusters):
        self.n_clusters = n_clusters

    def set_tif_list(self):
        self.tif_list = [os.path.join(self.folder_path, tif) for tif in os.listdir(self.folder_path) if tif.__contains__('.tif')]

    def scale_min_max_contours(self, m, r_min, r_max, t_min, t_max):
        try:

            x_scale = (m[0] - t_min)/(t_max - t_min) * (r_max - r_min) + r_min
            y_scale = (m[1] - t_min)/(t_max - t_min) * (r_max - r_min) + r_min
            return (int(x_scale), int(y_scale))

        except Exception as e:
            print(f"Exception in code block to scale contours:")
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)

    def build_empty_matrix(self, width, height):
        self.zero_matrix = np.zeros((width, height), int)
        x, y = np.linspace(0, width, width), np.linspace(0, height, height)
        self.X, self.Y = np.meshgrid(x, y)

    def calculate_color_band_stats(self):
        pass

    def get_contour_scale(self, dim, points):
        try:

            t_min = round(np.array(points).min(),0)
            t_max = round(np.array(points).max(),0)
            return 0, dim-1, int(t_min), int(t_max)

        except Exception as e:
            print(f"Exception in code block to get contour min/max:")
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)

    def build_flat_contour_array(self, img):
        try:

            width, length = img.shape[0], img.shape[1]
            self.build_empty_matrix(width, length)
            # print(self.zero_matrix.shape)
            cs = plt.contour(self.X, self.Y, img)
            contour_points = []
            for i, collection in enumerate(cs.collections[1:]):
                for path in collection.get_paths():
                    if path.to_polygons():
                        for npoly, polypoints in enumerate(path.to_polygons()):
                            for polypoint in polypoints:
                                # contour_points.append([polypoint[0], polypoint[1]])
                                contour_points.append((polypoint[0], polypoint[1]))
            r_min, r_max, t_min, t_max = self.get_contour_scale(width, contour_points)
            # print(r_min, r_max, t_min, t_max)
            contour_points = np.apply_along_axis(self.scale_min_max_contours, 1, contour_points, r_min, r_max, t_min, t_max)
            for point in contour_points:
                self.zero_matrix[point[0]][point[1]] = 1
            # print(np.sum(self.zero_matrix))
            return self.zero_matrix.T.flatten()

        except Exception as e:
            print(f"Exception in code block to build flat contour arrays:")
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)
            sys.exit(0)

    # def set_contour_element(self, contour, file):
    #     self.flat_contours[file] = contour

    # method to handle multiprocessing pool
    def multithread_contour_process(self, task_pool):
        num_processors = 8 #int(mp.cpu_count() / 2)
        # self.pbar = tqdm(total=len(task_pool))
        p = Pool(processes=num_processors)
        results = p.map(self.set_contour_list, task_pool)
        p.close()
        p.join()
        # self.pbar.close()
        self.flat_contours = results

    def set_contour_list(self, curr_file):
        # self.get_timer()
        # self.pbar.update(1)
        # print(f'Processing contours for {curr_file}')
        curr_cv_img = cv2.imread(curr_file, 2)
        contour = self.build_flat_contour_array(curr_cv_img)
        int_contour = [int(c) for c in contour]
        
        curr_rio_img = rio.open(curr_file)
        
        # contour_df = pd.DataFrame(contour, columns=['flat_contours'])
        # contour_df = pd.concat([contour_df[col].apply(pd.Series) for col in contour_df.columns])
        # print(contour_df.info())
        return 

    def configure_sparse_df(self):
        try:
            # print(len(self.flat_contours))
            # print(len(self.flat_contours[0]))
            sparse_df = pd.DataFrame({'contours': self.flat_contours})
            sparse_df = pd.concat([sparse_df[col].apply(pd.Series) for col in sparse_df.columns], axis=1, ignore_index=True)
            sparse_df['file'] = self.tif_list
            # print(sparse_df.head())
            self.tif_df = sparse_df

        except Exception as e:
            print(f'Exception in code block to build sparse dataframe:')
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)
            sys.exit(0)

    def cluster_tif_files(self):
        try:
            # print(self.tif_df.shape)
            # print(self.tif_df.head())

            df_cols = self.tif_df.columns.tolist()
            df_cols.remove('file')

            self.tif_df.fillna(0, inplace=True)

            contour_cols = np.linspace(0, len(df_cols)-1, len(df_cols))
            X = self.tif_df[contour_cols].values

            km = KMeans(n_clusters=self.n_clusters).fit(X)
            self.tif_df['cluster'] = km.labels_
            # self.tif_df.to_csv('tif_clusters.csv', index=False)
            
            # print(self.tif_df.head())

        except Exception as e:
            print(f'Exception in code block to cluster using Kmeans:')
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)
            sys.exit(0)

    def multithread_split_process(self, task_pool):
        num_processors = 8
        p = Pool(processes=num_processors)
        p.map(self.split_tif_files, task_pool)

    def split_tif_files(self, curr_file): # multiprocess this method
        try:
            file_name = os.path.split(curr_file)[1]
            # print(file_name)
            new_folder = 'cluster_{}'.format(str(self.tif_df.loc[self.tif_df['file']==curr_file]['cluster'].iloc[0]))
            # print(new_folder)

            if not os.path.exists(os.path.join(self.folder_path, new_folder)):
                os.mkdir(os.path.join(self.folder_path, new_folder))

            output_path = os.path.join(self.folder_path, new_folder, file_name)
            # print(output_path)
            shutil.move(curr_file, output_path)

        except Exception as e:
            print(f'Exception in code block to split tif files into cluster folders:')
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)
            sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--inpath', help='full directory path to location of tif files')
    parser.add_argument('-k', '--n_clusters', help='number of clusters for kmeans algorithm (creates and separates images into k folders)')
    args = parser.parse_args()

    start_time = time.time()

    tic = TifImageClusterer()
    tic.set_folder_path(args.inpath)
    tic.set_tif_list()

    print(f'Generating flattened contour information for {len(tic.tif_list)} images')
    tic.multithread_contour_process(tic.tif_list)

    print('Configuring sparse matrix dataframe')
    tic.configure_sparse_df()

    print('Performing KMeans clustering')
    tic.set_kmeans(int(args.n_clusters))
    tic.cluster_tif_files()

    print('Moving tif files to new folder structure based on clusters')
    # tic.multithread_split_process(tic.tif_list)
    for tif in tic.tif_list:
        tic.split_tif_files(tif)

    finish_time = time.time()
    print(f'Entire contour KMeans clustering and file splitting process took {finish_time - start_time} seconds.')