import os, queue
from zipfile import ZipFile
import multiprocessing as mp
from multiprocessing import Process, Pool, Queue, cpu_count
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='full directory path to location of zip files')
args = parser.parse_args()
path = args.path

def unzipFolder(zip_folder):
    try:
        # zip_folder = task_queue.get()
        # rename in and out folders
        in_folder = zip_folder
        print(f'\nFolder to unzip: {in_folder}')
        out_folder = zip_folder.replace('.zip','')
        print(f'\nOutput folder name: {out_folder}')
        
        # check that folder doesn't already exist
        if out_folder not in os.listdir(path):

            zf = ZipFile(in_folder, 'r')
            zf.extractall(out_folder)
            zf.close()

    except queue.Empty:
        print('Queue empty!')

def pool_handler(pool_me):
    num_processors = int(mp.cpu_count() / 2)
    p = Pool(processes=num_processors)
    p.map(unzipFolder, pool_me) 

if __name__ == "__main__":

    zip_folders = os.listdir(path)
    zip_folders = [os.path.join(path, file) for file in zip_folders if file.__contains__('.zip')]
    pool_handler(zip_folders)