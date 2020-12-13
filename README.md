Predictive flood mapping in cloud-obscured satellite imagery 
==============================
Uses Python, TensorFlow, Google Earth Engine

Set up notes
------------
 - A small test image (with flood conditioning features) is available for download [here](https://drive.google.com/drive/folders/1gACNKEvGl90Npgwi-NpwSVqOifPDIAQD?usp=sharing). The image should be extracted into `data/images`. 
    - Additional Landsat 8 images can be downloaded from Google Earth Engine (GEE) using the JavaScript API scripts in `scripts/gee`.
    - Precipitation and soil classification layers are unavailable on GEE and must be downloaded separately. 
        - For soil, use the `soil.py` script to find US states that intersect your images, then download the corresponding geodatabases from the [USDA NRCS Box](https://nrcs.app.box.com/v/soils/folder/17971946225) under "2020 gSSURGO by State". Then run `soil.py` to burn the soil class vectors into rasters.
        - For precipitation, run `get_precip_data.py` to download and aggregate daily station data using the NOAA
         API. These data need to be interpolated into a raster — I did this using Empirical Bayesian Kriging in
         ArcGIS with `precip_interpolation.arcpy` but this is not open-source and another method might be required.
 - Cloud masks in `CPR/data/images` were procedurally generated using the Perlin noise algorithm in `utils.py`
 - Train and predict using the scripts in `scripts/model_run` (see Directory, below, for which model iterations were the most successful)
 
Directory:
------------
<details>
  <summary>Description of scripts</summary>
  This is a description of all the useful scripts in this project. The unlisted scripts were only used during exploration and testing but are kept in the repository in case I need to borrow code in the future.

  1. scripts
     * **models.py:** Model architecture
     * **loss_functions.py:** Loss functions for Bayesian neural network
     * **training.py:** Functions to train models
     * **prediction.py:** Functions for prediction using trained models
     * **LR_conf_intervals.py:** Generates confidence intervals for logistic regression models
     * **figures.py:** Creates figures for presentation
     * **RCTS_viz.py:** Creates figures for random cloud trials
     * **band_combos.py:** Generates histogram-equalized RGB .png files of Landsat images
     * **examine_images.py:** Examines ROIs of Landsat band combos or flood conditioning features
     * **get_precip_data.py:** Downloads precipitation station data for the extent of image using NOAA API
     * **precip_interpolation_arcpy.py:** ArcPy (Python 2.7) script that interpolates precipitation station data
     * **soil.py:** Finds states that image intersects, downloads gSSURGO soil data, and creates soil texture raster
  2. random_cloud_trials
     * **dissimilarity.py:** For each random cloud trial, computes mean, var, entropy of train/test sets
     * **dissimilarity_viz.py:** Visualizes dissimilarity of train/test sets
     * **random_cloud_analysis.py:** Visualizes prediction metrics and statistics ofrandom cloud trials
  3. model_run
     * **BNN_kwon.py:** Trains, predicts, and estimates uncertainty using Bayesian neural network from Kwon et al. (2018)
     * **LR.py:** Trains, predicts, and estimates uncertainty using logistic regression model
     * **NN.py:** Trains, predicts, and estimates uncertainty using neural network model (not Bayesian)
     * **RF.py:** Trains and predicts using random forest classification
  4. gee
     * **image_ledger.js:** List of all the Landsat 8 images used in this project
     * **flood_search.js:** Finds and examines Landsat images that cover a given Dartmouth Flood Observatory event
     * **image_batching.js:** Examine images from flood_search.js to identify visible flooding and subset for export.
     * **features.js:** Visualization of all flood conditioning features
     * **export_to_drive.js:** Stacks image and features and exports to Google Drive
     * **utils.js:** All utilities for processing image and calculating flood conditioning features
     
     
</details>

Background:
------------
Maps of flood inundation derived from satellite imagery during and after a flood event are critical tools for disaster management. Their utility, however, is often limited by optically thick cloud cover that obscures many spaceborne sensors. This study, which was completed for my master's thesis at the University of Washington, explores a data-driven method to predict flooding in cloud-obscured Landsat 8 images using flood conditioning features, like topography and land use. 

This problem presents a unique challenge because the missing data from cloud gaps prevents the use of CNN architectures commonly used in computer vision tasks. To get around this, I reshape the images into 2D vectors and train and compare a number of models shown to accurately predict flood hazard in the literature. 

Methods:
------------
Relatively clear images from flood events were located with the Dartmouth Flood Observatory database and downloaded from Google Earth Engine. These images were then masked with artificial cloud cover generated using Perlin noise.

For an obscured image, models were trained on the visible pixels using 30m flood conditioning features. These flood conditioning features have been used extensively in data-driven flood modeling research (Tehrany et al. 2014 and 2017, Choubin et al. 2019, Mojaddadi et al. 2017). The trained models then predicted flooding in the cloud-covered pixels of that same image. Logistic regression (LR), random forest (RF), and neural networks (NN) were evaluated. To obtain uncertainty estimates associated with flood predictions, a Bayesian neural network (BNN) using Monte Carlo dropout was trained and compared to LR confidence intervals. 

![alt image](https://github.com/ianpdavies/CPR/blob/master/figs/features.png)
> Flood conditioning features used to predict flooding in cloud-covered pixels

A number of experiments were run using this framework:
 - Masking images with 10-90% cloud cover to identify the threshold of clear pixels needed for prediction. 
 - Training using only a subset of the image: randomly sampled points; points within permanent water buffers
 - Randomly generating different cloud masks to identify the role of randomness and pixels of outsize importance on model performance 
 - Reclassifying continuous features into discrete bins
 - Weighting features using weights of evidence and frequency analysis
 
Results:
------------
 
Overall, the logistic regression and neural network models had consistently high accuracy (x̄=0.96) and AUC (x̄=0.86) because they could successfully rule out areas that were highly unlikely to flood. However, they could not consistently identify the few flooded pixels in an image, with average recall of 0.35 across all images. It is unclear why the models performed so poorly compared to other studies using the same flood conditioning features. Many of those studies used <500 sample points in test/train datasets, compared to >10^6 in this study, so it is possible that prediction breaks down at large scales.

Although the models were generally robust to cloud coverage (no difference in performance with 10% cloud cover vs. 90%), the placement of those clouds had a significant impact on performance, with a recall variance as high as 0.20 between runs with different cloud masks. 

<img src="https://github.com/ianpdavies/CPR/blob/master/figs/model_comparison_plot.png" width="70%" height="70%">
 
> Mean performance of different models. Each point represents the average score for that model at a given percent of cloud cover. Average across all cloud covers is noted in the corner.
  
With such variability in predictions, it was crucial that the models provide some measure of their uncertainty. The Bayesian neural network tracked with prediction errors much better than logistic regression confidence intervals. Uncertainty measures are glaringly absent from most flood prediction research, and their inclusion could be a useful tool to bolster user confidence in the results.

<img src="https://github.com/ianpdavies/CPR/blob/master/figs/aleatoric_epistemic_uncertainty.png" width="80%" height="80%">
 
>Uncertainty measures (top) and predictions (bottom) for the BNN (left) and LR confidence intervals (right) models in a segment of a sample image. While uncertainty is high for all predictions of flooding in the BNN, it is highest for false positives and false negatives. The LR confidence intervals were not able to discriminate error types as well.
  
<img src="https://github.com/ianpdavies/CPR/blob/master/figs/BNN_uncertainty.png" width="80%" height="80%">.
  
>Histograms of relative prediction type binned by uncertainty of BNN (left) and LR confidence interval (right) 

-----------

#### Cited Research:

Choubin, B., Moradi, E., Golshan, M., Adamowski, J., Sajedi-Hosseini, F., & Mosavi, A. (2019). An ensemble prediction of flood susceptibility using multivariate discriminant analysis, classification and regression trees, and support vector machines. Science of the Total Environment, 651, 2087–2096. https://doi.org/10.1016/j.scitotenv.2018.10.064

Mojaddadi, H., Pradhan, B., Nampak, H., Ahmad, N., & Ghazali, A. H. bin. (2017). Ensemble machine-learning-based geospatial approach for flood risk assessment using multi-sensor remote-sensing data and GIS. Geomatics, Natural Hazards and Risk, 8(2), 1080–1102. https://doi.org/10.1080/19475705.2017.1294113

Shafapour Tehrany, M., Shabani, F., Neamah Jebur, M., Hong, H., Chen, W., & Xie, X. (2017). GIS-based spatial prediction of flood prone areas using standalone frequency ratio, logistic regression, weight of evidence and their ensemble techniques. Geomatics, Natural Hazards and Risk, 8(2), 1538–1561. https://doi.org/10.1080/19475705.2017.1362038

Tehrany, M. S., Pradhan, B., & Jebur, M. N. (2014). Flood susceptibility mapping using a novel ensemble weights-of-evidence and support vector machine models in GIS. Journal of Hydrology, 512, 332–343. https://doi.org/10.1016/J.JHYDROL.2014.03.008




