from osgeo import gdal, gdalconst, osr
import numpy as np, sys
import rasterio
sys.path.append('../')
from CPR.configs import data_path

img = '4444_LC08_044032_20170222_1'
pctl = 50
#=======================================================================================================================
# Get flood layer

img_path = data_path / 'images' / img
stack_path = img_path / 'stack' / 'stack.tif'

# load cloudmasks
clouds_dir = data_path / 'clouds'
clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))

# Check for any features that have all zeros/same value and remove. This only matters with the training data
# Get local image
with rasterio.open(str(stack_path), 'r') as ds:
    data = ds.read()
    data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
    data[data == -999999] = np.nan
    data[np.isneginf(data)] = np.nan
    mask = data[:, :, 15]

#=======================================================================================================================
def raster_buffer(raster_filepath, dist=1000):
    """This function creates a distance buffer around the given raster file with non-zero values.
     The value in output raster will have value of the cell to which it is close to."""
    d = gdal.Open(raster_filepath)
    if d is None:
        print("Error: Could not open image " + raster_filepath)
        sys.exit(1)
    global proj, geotrans, row, col
    proj = d.GetProjection()
    geotrans = d.GetGeoTransform()
    geotrans = (0.0002777777777777778, 0, -180.0001388888889, 0, -0.0002777777777777778, 60.00013888888889)
    row = d.RasterYSize
    col = d.RasterXSize
    inband = d.GetRasterBand(16)
    in_array = inband.ReadAsArray(0, 0, col, row).astype(int)
    Xcell_size = int(abs(geotrans[1]))
    Ycell_size = int(abs(geotrans[5]))
    cell_size = (Xcell_size + Ycell_size) / 2
    cell_dist = dist / cell_size
    in_array[in_array == (inband.GetNoDataValue() or 0 or -999)] = 0
    out_array = np.zeros_like(in_array)
    temp_array = np.zeros_like(in_array)
    i, j, h, k = 0, 0, 0, 0
    print("Running distance buffer...")
    while h < col:
        k = 0
        while k < row:
            if in_array[k][h] >= 1:
                i = h - cell_dist
                while (i < cell_dist + h) and i < col:
                    j = k - cell_dist
                    while j < (cell_dist + k) and j < row:
                        if ((i - h) ** 2 + (j - k) ** 2) <= cell_dist ** 2:
                            if temp_array[j][i] == 0 or temp_array[j][i] > ((i - h) ** 2 + (j - k) ** 2):
                                out_array[j][i] = in_array[k][h]
                                temp_array[j][i] = (i - h) ** 2 + (j - k) ** 2
                        j += 1
                    i += 1
            k += 1
        h += 1
    d, temp_array, in_array = None, None, None
    return out_array


def export_array(in_array, output_path):
    """This function is used to produce output of array as a map."""
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output_path, col, row, 1)
    outband = outdata.GetRasterBand(1)
    outband.SetNoDataValue(np.nan)
    outband.WriteArray(in_array)
    # Georeference the image
    outdata.SetGeoTransform(geotrans)
    # Write projection information
    outdata.SetProjection(proj)
    outdata.FlushCache()
    outdata = None


raster_buffer_array = raster_buffer(str(stack_path), 1000)


export_array(raster_buffer_array, "Output//buffer.tif")
print("Done")


#=======================================================================================================================

"""This function creates a distance buffer around the given raster file with non-zero values.
   The value in output raster will have value of the cell to which it is close to."""
dist=1000
d = gdal.Open(str(stack_path))
if d is None:
    print("Error: Could not open image " + str(stack_path))
    sys.exit(1)
global proj, geotrans, row, col
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
d.SetProjection(srs.ExportToWkt())
proj = d.GetProjection()
geotrans = d.GetGeoTransform()
geotrans = (0.0002777777777777778, 0, -180.0001388888889, 0, -0.0002777777777777778, 60.00013888888889)
row = d.RasterYSize
col = d.RasterXSize
inband = d.GetRasterBand(16)
in_array = inband.ReadAsArray(0, 0, col, row).astype(int)
Xcell_size = int(abs(geotrans[1]))
Ycell_size = int(abs(geotrans[5]))
cell_size = (Xcell_size + Ycell_size) / 2
cell_dist = dist / cell_size
in_array[in_array == (inband.GetNoDataValue() or 0 or -999)] = 0
out_array = np.zeros_like(in_array)
temp_array = np.zeros_like(in_array)
i, j, h, k = 0, 0, 0, 0
print("Running distance buffer...")
while h < col:
    k = 0
    while k < row:
        if in_array[k][h] >= 1:
            i = h - cell_dist
            while (i < cell_dist + h) and i < col:
                j = k - cell_dist
                while j < (cell_dist + k) and j < row:
                    if ((i - h) ** 2 + (j - k) ** 2) <= cell_dist ** 2:
                        if temp_array[j][i] == 0 or temp_array[j][i] > ((i - h) ** 2 + (j - k) ** 2):
                            out_array[j][i] = in_array[k][h]
                            temp_array[j][i] = (i - h) ** 2 + (j - k) ** 2
                    j += 1
                i += 1
        k += 1
    h += 1
d, temp_array, in_array = None, None, None

#=======================================================================================================================
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
dilated_mask = binary_dilation(mask, iterations=20)
plt.figure()
plt.imshow(mask)
plt.figure()
plt.imshow(dilated_mask)

