import os, argparse, sys
from multiprocessing import Process, Pool, Queue, cpu_count
from itertools import product
import multiprocessing as mp
import rasterio as rio
from rasterio import windows
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

# declare global variables
dst_crs = 'EPSG:4326'
# out_path = r'D:/Raster/HRO/parsed_ortho_files/'
list_of_tifs = []

# fetch argparse params
parser = argparse.ArgumentParser()
parser.add_argument('--inpath', help='full directory path to location of tif files')
parser.add_argument('--outpath', help='full directory path to location to store clipped tiles')
parser.add_argument('--width', help='specify the width for spliced image')
parser.add_argument('--height', help='specify the height for spliced image')
args = parser.parse_args()

# argparse variables
arg_inpath = args.inpath
arg_outpath = args.outpath
arg_width = args.width
arg_height = args.height

# check if input folder exists
if not os.path.exists(arg_inpath):
    print('Source folder does not exist, exiting program...')
    sys.exit(0)

# check if output folder exists
if not os.path.exists(arg_outpath):
    print(f'Destination folder does not exist, creating it...')
    os.mkdir(arg_outpath)

# function to fetch the path of every .TIF file within subfolder of inpath
def get_tif_path(start):
    curr_dir = os.listdir(start)
    for root, folders, files in os.walk(start):
        for file in files:
            if file.__contains__('.tif'):
                list_of_tifs.append(os.path.join(root, file))

# method to create and return a dimensional window based on argparse width and height
def generate_tiles(input_tif, width=500, height=500):
    n_cols, n_rows = input_tif.meta['width'], input_tif.meta['height']
    offsets = product(range(0, n_cols, width), range(0, n_rows, height))
    full_window = windows.Window(col_off=0, row_off=0, width=n_cols, height=n_rows)
    for col_off, row_off in offsets:
        window=windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(full_window)
        transform = windows.transform(window, input_tif.transform)
        yield window, transform

# method to clip a single 5000x5000 tif file into NxM tiles based on argparse width and height
def clip_full_raster(in_file, in_path, out_file, out_path, dst_crs):
    with rio.open(in_path, num_threads=8) as in_tiff: #, num_threads=8
        tile_width, tile_height = int(arg_width), int(arg_height)
        meta = in_tiff.meta.copy()
        number_of_bands = meta['count']

        # iterate over each window returned to read only active window dimensions
        for window, transform in generate_tiles(in_tiff, tile_width, tile_height):
            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
            meta['photometric'] = 'RGB'
            print(meta)
            full_dest_path = os.path.join(out_path, out_file.format(int(window.col_off), int(window.row_off)))
            raster_window = in_tiff.read(window=window)

            # write clipped window of original tif to file without reprojection
            with rio.open(full_dest_path, 'w', num_threads=8, **meta) as out_tiff: #, num_threads=8
                out_tiff.write(raster_window, indexes=list(range(1, number_of_bands+1)))

# helper method to parse file path into in and out paths and file names
def parse_file_path(folder):
    in_path = folder
    in_file = folder.split('\\')[-1]
    prefix = in_file.replace('.tif','')
    out_file = prefix + '_{}-{}.tif'
    clip_full_raster(in_file, in_path, out_file, arg_outpath, dst_crs)

# method to handle multiprocessing pool
def pool_handler(pool_me):
    num_processors = 8 #int(mp.cpu_count() / 2)
    p = Pool(processes=num_processors)
    p.map(parse_file_path, pool_me)

if __name__ == "__main__":
    image_folders = [folder for folder in os.listdir(arg_inpath) if not folder.__contains__('.ipynb')]
    for folder in image_folders:
        get_tif_path(os.path.join(arg_inpath, folder))

    # print(list_of_tifs)
    pool_handler(list_of_tifs)







###----------------------------------------------------------------###

### SCRIPT FOR REPROJECTING TO DESTINATION CRS ###
#       # attempting to reproject to EPSG 4326 while applying windows
#         with rasterio.open(outpath, 'w', **meta) as out_tiff:
#             for i in range(1, in_tiff.count+1):
#                 reproject(
#                     source=rio.band(in_tiff, i),
#                     destination=rio.band(out_tiff, i),
#                     src_transform=in_tiff.transform,
#                     src_crs=in_tiff.crs,
#                     dst_transform=transform,
#                     dst_crs=dst_crs,
#                     resampling=Resampling.nearest)