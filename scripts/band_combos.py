import os
import sys
sys.path.append('../')
from CPR.configs import data_path
from results_viz import VizFuncs
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure, io
import rasterio

pctls = [10, 30, 50, 70, 90]

# Get all images in image directory
img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test', '4444_LC08_044034_20170222_1',
           '4101_LC08_027038_20131103_2', '4594_LC08_022035_20180404_1', '4444_LC08_043035_20170303_1'}
img_list = [x for x in img_list if x not in removed]

feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

viz_params = {'img_list': img_list,
              'pctls': pctls,
              'data_path': data_path,
              'batch': None,
              'feat_list_new': feat_list_new}

# ======================================================================================================================
# viz = VizFuncs(viz_params)
# viz.cir_image(overwrite=True)
# viz.rgb_image(percent=1.25, overwrite=True)


def linear_stretch(input, percent):
    p_low, p_high = np.percentile(input[~np.isnan(input)], (percent, 100 - percent))
    img_rescale = exposure.rescale_intensity(input, in_range=(p_low, p_high))
    return img_rescale

# RGB images that are too bright after linear stretch
# img_list = ['4080_LC08_028034_20130806_1',
#             '4101_LC08_027038_20131103_1',
#             '4101_LC08_027038_20131103_2',
#             '4115_LC08_021033_20131227_1',
#             '4115_LC08_021033_20131227_2',
#             '4444_LC08_045032_20170301_1',
#             '4468_LC08_022035_20170503_1',
#             '4468_LC08_024036_20170501_2',
#             '4514_LC08_027033_20170826_1',
#             '4516_LC08_017038_20170921_1',
#             '4594_LC08_022034_20180404_1',
#             '4594_LC08_022035_20180404_1']

def hist_equalize_rgb(img, view_hist=False, view_img=False, std_low=1.75, std_high=1.75, save=False):
    band_combo_dir = data_path / 'band_combos'
    rgb_file = band_combo_dir / '{}'.format(img + '_rgb_img' + '.png')

    spectra_stack_path = data_path / 'images' / img / 'stack' / 'spectra_stack.tif'
    band_combo_dir = data_path / 'band_combos'
    rgb_file = band_combo_dir / '{}'.format(img + '_rgb_img' + '.png')

    with rasterio.open(spectra_stack_path, 'r') as f:
        red, green, blue = f.read(4), f.read(3), f.read(2)
        red[red == -999999] = 0
        green[green == -999999] = 0
        blue[blue == -999999] = 0
        rgb = np.dstack((red, green, blue))

    shape = rgb.shape
    rgb_vector = rgb.reshape([rgb.shape[0] * rgb.shape[1], rgb.shape[2]])
    rgb_vector = rgb_vector[~np.isnan(rgb_vector).any(axis=1)]

    # View histogram of RGB values
    if view_hist:
        fig = plt.figure(figsize=(10, 7))
        fig.set_facecolor('white')
        for color, channel in zip('rgb_vector', np.rollaxis(rgb, axis=-1)):
            counts, centers = exposure.histogram(channel)
            plt.plot(centers[1::], counts[1::], color=color)
        plt.show()

    lims = []
    for i in range(3):
        x = np.mean(rgb_vector[:, i])
        sd = np.std(rgb_vector[:, i])
        low = x - (std_low * sd)
        high = x + (std_high * sd)
        if low < 0:
            low = 0
        if high > 1:
            high = 1
        lims.append((low, high))

    r = exposure.rescale_intensity(rgb[:, :, 0], in_range=lims[0])
    g = exposure.rescale_intensity(rgb[:, :, 1], in_range=lims[1])
    b = exposure.rescale_intensity(rgb[:, :, 2], in_range=lims[2])
    rgb_enhanced = np.dstack((r, g, b))
    if view_img:
        plt.figure()
        plt.imshow(rgb_enhanced)

    rgb_img = Image.fromarray((rgb_enhanced * 255).astype(np.uint8()))

    if save:
        rgb_img.save(rgb_file, dpi=(300, 300))

    return rgb_enhanced

from skimage import exposure
band_combo_dir = data_path / 'band_combos'
for img in img_list:
    hist_equalize_rgb(img, view_hist=False, std_low=1.75, std_high=1.75, save=True)

img = '4444_LC08_044033_20170222_3'
rgb = hist_equalize_rgb(img, view_hist=False, std_low=2.1, std_high=2.1, save=False)
rgb_enhanced = exposure.adjust_gamma(rgb, gamma=1, gain=1)
rgb_enhanced = exposure.adjust_sigmoid(rgb_enhanced, cutoff=0.5, gain=7)
rgb_img = Image.fromarray((rgb_enhanced * 255).astype(np.uint8()))
rgb_img.save(band_combo_dir / '{}'.format(img + '_rgb_img' + '.png'), dpi=(300, 300))

img = '4444_LC08_045032_20170301_1'
rgb = hist_equalize_rgb(img, view_hist=False, std_low=2.5, std_high=2.5, save=False)
rgb_enhanced = exposure.adjust_gamma(rgb, gamma=1.2, gain=1)
rgb_enhanced = exposure.adjust_sigmoid(rgb_enhanced, cutoff=0.49, gain=10)
rgb_img = Image.fromarray((rgb_enhanced * 255).astype(np.uint8()))
rgb_img.save(band_combo_dir / '{}'.format(img + '_rgb_img' + '.png'), dpi=(300, 300))

img = '4468_LC08_022035_20170503_1'
hist_equalize_rgb(img, view_hist=False, std_low=2.5, std_high=2.7, save=True)

img = '4516_LC08_017038_20170921_1'
rgb = hist_equalize_rgb(img, view_hist=False, std_low=2.7, std_high=2.7, save=False)
rgb_enhanced = exposure.adjust_gamma(rgb, gamma=1, gain=1)
rgb_enhanced = exposure.adjust_sigmoid(rgb_enhanced, cutoff=0.5, gain=8)
rgb_img = Image.fromarray((rgb_enhanced * 255).astype(np.uint8()))
rgb_img.save(band_combo_dir / '{}'.format(img + '_rgb_img' + '.png'), dpi=(300, 300))

img = '4594_LC08_022034_20180404_1'
rgb = hist_equalize_rgb(img, view_hist=False, std_low=2.4, std_high=2.4, save=False)
rgb_enhanced = exposure.adjust_gamma(rgb, gamma=1.2, gain=1)
rgb_enhanced = exposure.adjust_sigmoid(rgb_enhanced, cutoff=0.5, gain=7)
rgb_img = Image.fromarray((rgb_enhanced * 255).astype(np.uint8()))
rgb_img.save(band_combo_dir / '{}'.format(img + '_rgb_img' + '.png'), dpi=(300, 300))

img = '4444_LC08_043035_20170303_1'
rgb = hist_equalize_rgb(img, view_hist=False, std_low=1.75, std_high=1.75, save=False)
rgb_enhanced = exposure.adjust_sigmoid(rgb, cutoff=0.55, gain=6)
rgb_enhanced = exposure.adjust_gamma(rgb_enhanced, gamma=1.1, gain=1)
rgb_img = Image.fromarray((rgb_enhanced * 255).astype(np.uint8()))
rgb_img.save(band_combo_dir / '{}'.format(img + '_rgb_img' + '.png'), dpi=(300, 300))

img = '4594_LC08_022035_20180404_1'
rgb = hist_equalize_rgb(img, view_hist=False, std_low=2.4, std_high=2.4, save=False)
rgb_enhanced = exposure.adjust_gamma(rgb, gamma=1.2, gain=1)
rgb_enhanced = exposure.adjust_sigmoid(rgb_enhanced, cutoff=0.5, gain=10)
rgb_img = Image.fromarray((rgb_enhanced * 255).astype(np.uint8()))
rgb_img.save(band_combo_dir / '{}'.format(img + '_rgb_img' + '.png'), dpi=(300, 300))

img = '4115_LC08_021033_20131227_2'
rgb = hist_equalize_rgb(img, view_hist=False, std_low=2.5, std_high=2.5, save=False)
rgb_enhanced = exposure.adjust_gamma(rgb, gamma=1.2, gain=1)
rgb_enhanced = exposure.adjust_sigmoid(rgb_enhanced, cutoff=0.5, gain=6)
rgb_img = Image.fromarray((rgb_enhanced * 255).astype(np.uint8()))
rgb_img.save(band_combo_dir / '{}'.format(img + '_rgb_img' + '.png'), dpi=(300, 300))
plt.imshow(rgb_enhanced)

img = '4468_LC08_024036_20170501_1'
rgb = hist_equalize_rgb(img, view_hist=False, std_low=2.4, std_high=2.4, save=False)
rgb_enhanced = exposure.adjust_gamma(rgb, gamma=1.2, gain=1)
rgb_img = Image.fromarray((rgb_enhanced * 255).astype(np.uint8()))
rgb_img.save(band_combo_dir / '{}'.format(img + '_rgb_img' + '.png'), dpi=(300, 300))

img = '4444_LC08_044034_20170222_1'
hist_equalize_rgb(img, view_hist=False, std_low=2.4, std_high=2.4, save=True)

img = '4444_LC08_044033_20170222_4'
hist_equalize_rgb(img, view_hist=False, std_low=2.7, std_high=2.7, save=True)

img = '4444_LC08_044032_20170222_1'
hist_equalize_rgb(img, view_hist=False, std_low=2.6, std_high=2.6, save=True)

img = '4444_LC08_044033_20170222_2'
hist_equalize_rgb(img, view_hist=False, std_low=2.6, std_high=2.6, save=True)

img = '4080_LC08_028033_20130806_1'
rgb = hist_equalize_rgb(img, view_hist=False, std_low=2.6, std_high=2.6, save=False)
rgb_enhanced = exposure.adjust_gamma(rgb, gamma=1.1, gain=1)
rgb_enhanced = exposure.adjust_sigmoid(rgb_enhanced, cutoff=0.5, gain=10)
rgb_img = Image.fromarray((rgb_enhanced * 255).astype(np.uint8()))
rgb_img.save(band_combo_dir / '{}'.format(img + '_rgb_img' + '.png'), dpi=(300, 300))

img = '4101_LC08_027038_20131103_2'
rgb = hist_equalize_rgb(img, view_hist=False, view_img=True, std_low=2.6, std_high=2.6, save=False)
rgb_enhanced = exposure.adjust_gamma(rgb, gamma=1.2, gain=1)
rgb_img = Image.fromarray((rgb_enhanced * 255).astype(np.uint8()))
rgb_img.save(band_combo_dir / '{}'.format(img + '_rgb_img' + '.png'), dpi=(300, 300))

img = '4101_LC08_027038_20131103_1'
rgb = hist_equalize_rgb(img, view_hist=False, view_img=True, std_low=2.6, std_high=2.6, save=False)
rgb_enhanced = exposure.adjust_gamma(rgb, gamma=1.2, gain=1)
rgb_img = Image.fromarray((rgb_enhanced * 255).astype(np.uint8()))
rgb_img.save(band_combo_dir / '{}'.format(img + '_rgb_img' + '.png'), dpi=(300, 300))

img = '4101_LC08_027039_20131103_1'
rgb = hist_equalize_rgb(img, view_hist=False, view_img=True, std_low=2.6, std_high=2.6, save=False)
rgb_enhanced = exposure.adjust_gamma(rgb, gamma=1.2, gain=1)
rgb_img = Image.fromarray((rgb_enhanced * 255).astype(np.uint8()))
rgb_img.save(band_combo_dir / '{}'.format(img + '_rgb_img' + '.png'), dpi=(300, 300))

img = '4468_LC08_024036_20170501_2'
rgb = hist_equalize_rgb(img, view_hist=False, view_img=True, std_low=2.6, std_high=2.6, save=False)
rgb_enhanced = exposure.adjust_gamma(rgb, gamma=1.2, gain=1)
rgb_img = Image.fromarray((rgb_enhanced * 255).astype(np.uint8()))
rgb_img.save(band_combo_dir / '{}'.format(img + '_rgb_img' + '.png'), dpi=(300, 300))

img = '4089_LC08_034032_20130917_1'
rgb = hist_equalize_rgb(img, view_hist=False, view_img=True, std_low=2, std_high=2, save=False)
rgb_enhanced = exposure.adjust_gamma(rgb, gamma=1.2, gain=1)
rgb_enhanced = exposure.adjust_sigmoid(rgb_enhanced, cutoff=0.5, gain=5.5)
rgb_img = Image.fromarray((rgb_enhanced * 255).astype(np.uint8()))
rgb_img.save(band_combo_dir / '{}'.format(img + '_rgb_img' + '.png'), dpi=(300, 300))

img = '4444_LC08_044033_20170222_1'
rgb = hist_equalize_rgb(img, view_hist=False, view_img=True, std_low=1.8, std_high=1.8, save=False)
rgb_enhanced = exposure.adjust_gamma(rgb, gamma=1.2, gain=1)
rgb_enhanced = exposure.adjust_sigmoid(rgb_enhanced, cutoff=0.5, gain=6)
rgb_img = Image.fromarray((rgb_enhanced * 255).astype(np.uint8()))
rgb_img.save(band_combo_dir / '{}'.format(img + '_rgb_img' + '.png'), dpi=(300, 300))

img = '4337_LC08_026038_20160325_1'
rgb = hist_equalize_rgb(img, view_hist=False, view_img=True, std_low=4, std_high=4, save=False)
rgb_enhanced = exposure.adjust_gamma(rgb, gamma=1.1, gain=1)
rgb_enhanced = exposure.adjust_sigmoid(rgb_enhanced, cutoff=0.55, gain=6.5)
plt.imshow(rgb_enhanced)
rgb_img = Image.fromarray((rgb_enhanced * 255).astype(np.uint8()))
rgb_img.save(band_combo_dir / '{}'.format(img + '_rgb_img' + '.png'), dpi=(300, 300))

img = '4477_LC08_022033_20170519_1'
rgb = hist_equalize_rgb(img, view_hist=False, view_img=True, std_low=3.5, std_high=3.5, save=False)
rgb_enhanced = exposure.adjust_gamma(rgb, gamma=1.1, gain=1)
rgb_enhanced = exposure.adjust_sigmoid(rgb_enhanced, cutoff=0.5, gain=6.5)
rgb_img = Image.fromarray((rgb_enhanced * 255).astype(np.uint8()))
rgb_img.save(band_combo_dir / '{}'.format(img + '_rgb_img' + '.png'), dpi=(300, 300))

plt.close('all')

batch=None
feat_list_new = ['GSW_distSeasonal', 'aspect', 'curve', 'elevation', 'hand', 'slope',
                 'spi', 'twi', 'sti', 'GSW_perm', 'flooded']

feat_list_all = ['developed', 'forest', 'planted', 'wetlands', 'openspace', 'carbonate', 'noncarbonate', 'akl_intrusive',
                 'silicic_resid', 'silicic_resid', 'extrusive_volcanic', 'colluvial_sed', 'glacial_till_clay',
                 'glacial_till_loam', 'glacial_till_coarse', 'glacial_lake_sed_fine', 'glacial_outwash_coarse',
                 'hydric', 'eolian_sed_coarse', 'eolian_sed_fine', 'saline_lake_sed', 'alluv_coastal_sed_fine',
                 'coastal_sed_coarse', 'GSW_distSeasonal', 'aspect', 'curve', 'elevation', 'hand', 'slope', 'spi',
                 'twi', 'sti', 'GSW_perm', 'flooded']

img_list = ['4337_LC08_026038_20160325_1']

viz_params = {'img_list': img_list,
              'pctls': pctls,
              'data_path': data_path,
              'batch': batch,
              'feat_list_new': feat_list_new}
viz = VizFuncs(viz_params)
viz.cir_image(overwrite=True)

# Doesn't work very well for CIR images - water and dry earth are both turqoise.
def hist_equalize_cir(img, view_hist=False, view_img=False, std_low=1.75, std_high=1.75, save=False):
    spectra_stack_path = data_path / 'images' / img / 'stack' / 'spectra_stack.tif'
    band_combo_dir = data_path / 'band_combos'
    cir_file = band_combo_dir / '{}'.format(img + '_cir_img' + '.png')

    with rasterio.open(spectra_stack_path, 'r') as f:
        nir, red, green = f.read(5), f.read(4), f.read(3)
        nir[nir == -999999] = np.nan
        red[red == -999999] = np.nan
        green[green == -999999] = np.nan
        cir = np.dstack((nir, red, green))

    cir_vector = cir.reshape([cir.shape[0] * cir.shape[1], cir.shape[2]])
    cir_vector = cir_vector[~np.isnan(cir_vector).any(axis=1)]

    # View histogram of RGB values
    if view_hist:
        fig = plt.figure(figsize=(10, 7))
        fig.set_facecolor('white')
        for color, channel in zip('rgb_vector', np.rollaxis(cir_vector, axis=-1)):
            counts, centers = exposure.histogram(channel)
            plt.plot(centers[1::], counts[1::], color=color)
        plt.show()

    lims = []
    for i in range(3):
        x = np.mean(cir_vector[:, i])
        sd = np.std(cir_vector[:, i])
        low = x - (std_low * sd)
        high = x + (std_high * sd)
        if low < 0:
            low = 0
        if high > 1:
            high = 1
        lims.append((low, high))

    r = exposure.rescale_intensity(cir[:, :, 0], in_range=lims[0])
    g = exposure.rescale_intensity(cir[:, :, 1], in_range=lims[1])
    b = exposure.rescale_intensity(cir[:, :, 2], in_range=lims[2])
    cir_enhanced = np.dstack((r, g, b))
    if view_img:
        plt.figure()
        plt.imshow(cir_enhanced)

    cir_img = Image.fromarray((cir_enhanced * 255).astype(np.uint8()))

    if save:
        cir_img.save(cir_file, dpi=(300, 300))
