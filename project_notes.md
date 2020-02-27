Project Notes
---

### To Do

1. Examine images for errors: false positives, missing data
    **Images**
    - 4101_LC08_027038_20131103_1: lots of FN at 10%, FPs at 90%
    - 4101_LC08_027038_20131103_2: lots of FN at 10%
    - 4115_LC08_021033_20131227_1: lots of FN at 10%
    - 4115_LC08_021033_20131227_1: lots of FN at 10%; many FPs
    - 4115_LC08_021033_20131227_2: Good flood prediction
    - 4337_LC08_026038_20160325_1: Looks like perm water wasn't masked in lake. High FNs
    - 4444_LC08_043034_20170303_1: lots of FNs, alternating with cloud cover
    - 4444_LC08_043035_20170303_1: weird FP clumps in hills
    - 4444_LC08_044032_20170222_1: big increase in FPs from 30-50%
    - 4444_LC08_044033_20170222_2: 10 and 90% both have lots of FNs
    
    **Image Notes**
     - Looks like 10 and 90% masks got flipped. See 4444_LC08_044033_20170222_1
       - However, both 10 and 90% have lots of FNs. Why does low cloud cover have high FNs?
         - One possibility: with more clear pixels, less likely that the remaining pixels are closely related to the 
         cloudy pixels, could be spatially distant. 
     - Model tends to be conservative? More FNs than FPs
  
    1a. Check for correct cloud masking (i.e. 10%=10%)

2. Why such large variance in random cloud  trials?
    - Run PCA on unmasked image, then plot each trial in the PC1 vs. PC2 space, and find Mahalanobis distance
    - Mahalanobis might not be appropriate - perhaps just compare the distributions of test and train data for each trial
      - KL-Divergence
      - Plot of overlapping KDE curves
    - Create map overlaying TP/FP on the flood layer. See if there is some buffer calculation? If cloudy area is flooded,
    it might only be predicted if nearby clear area is also flooded.
    - Visually examine images with good predictions and those with poor - what features stand out? Diffuse flooding that
    isn't just around a river? 
        - Look at KDEs of high performing images.
        - Manually put clouds over these areas and run experiment.

3. Visualization
    - Can clouds be just borders with transparent fill? To better visualize overlay.

4. ☒ Logistic regression, SVM, RF results
    - Run log reg and RF on all images at all pctls
    - Take the best performing one and tune hyperparameters
    - Compare to NN
    - Logistic regression
        - Performs better than NN, and faster (I think)
        - But can't be run in parallel, so there is a solid floor to training/prediction time
    - RF
        - Appears to perform poorly using default hyperparams - high precision and low recall, super conservative estimates
        - Might be able to improve with tuning 

5. ☒ Train on small dataset to reduce sparseness of the data
     -  Use binary dilation to buffer flooded pixels
        - Train with varying dilation iterations 10-50 around (a) all water and (b) just flooded pixels
        - Train with dilation iterations around all water + random pixels selected throughout image
        - Depending on the result above, train model with buffers around clouds vs. only around water/flood
            - binary_dilation buffers around NaNs, so will have to turn those to 0 while making the mask to remove cloud
            buffers
        - BUFFER EXPERIMENT:
            - Run the above model permutations with half of images and compare to log reg with half of images

6. Random cloud trials
    - According to Max (CSSS consulting) I should run 30-100 trials. Time permitting, I can do this on the QERM servers.

7. Uncertainty
     - ☒ Get uncertainty estimates from logistic regression
     - Compare these uncertainty estimates to those from the NN. 
     - ☒ Test remove_perm=True/False again with log reg. The log reg buffer tests

------
- Rerun cloud trials with more samples (30-100)
- Rerun val data vs. no val data to see which is better
- Visually inspect images to see why some perform better
- Compare log reg probs vs. Bayesian NN
- Run on more images with final best performing model

Experiments
1. Val data vs. no val data
2. Perm water masking
3. RF vs. LR vs. NN
4. Random cloud trials on best performing model from 3.
5. Visual inspection of images to explain variance
6. (maybe) experiment with randomly taking out patches of image with pixels that might be useful, suggested by 5.
6. Uncertainty: LR probs vs. Bayesian NN

To do:
 - ☒ Make cloud borders
    - Remove setdata() section of false_map() function, make it like border creation to save time
 - ☒ Rerun v39 because perm water isn't being removed properly - still showing up as TP in false maps
 - Rerun NN using val data/perm water settings
 - Train on 2/3 of images and test on remaining 1/3
 - Which does better - LR or NN?
    - Make AUC/ROC plots
 - Inspect images
 - Does Bayesian NN perform as well as NN?
 - Compare BNN to LR probs
    - Is BNN aleatoric just predicting GSW layer? 
 - Delete old models and test scripts
    - Run only the models needed for experiments that will go in thesis


Models to run:
- ☒ LR with different permanent water masks, 10-90% CC
- RF using best data versions from val data/perm water experiments
- LR using best data versions from val data/perm water experiments
- NN using best data versions from val data/perm water experiments
- Random cloud trials using whichever model performed best (LR/RF/NN).
    - At 10-90% CC or fixed at one? Or perhaps only 25/50/75%? Save time and space
- 
-Line 211 in results_viz where I mask out perm water in predictions img - need to see if that new operation is needed 
elsewhere
- Feature engineering?

- Should probably remove 4516 because it's coastal floading
Y.Gal's concrete dropout has this at the end of aleatoric uncertainty .. -0.5 * loss  # return ELBO up to const.
What does that mean? Is that necessary to get good uncertainty?

https://www.inovex.de/blog/uncertainty-quantification-deep-learning/

Custom loss function for NN? Macro F1 that uses probs instead of 0/1 so the loss function is differentiable
https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
should maybe  change perm water mask from permanent to permanent+new_permanent (transitions layer) or maybe the seasonal layer
CNN?

use autoencoder for dimensionality reduction?
relu vs sigmoid vs leaky relu vs ELU?
redo buffer - forgot to actually sample dry pixels that weren't in the buffer!

https://bamos.github.io/2016/08/09/deep-completion/

Found updated DFO database file but no polygon extent, only lat/long. Can add buffer around it to get images in similar way.

what about a CNN used for fixing missing band, but instead of spectral band the band is actually flood
https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11139/111390E/Reconstruct-missing-pixels-of-Landsat-land-surface-temperature-product-using/10.1117/12.2529462.full?SSO=1


### Image examination
   - Low recall/F1
     - 4080_LC08_028034_20130806_1:
        - Very few preds
        - Weaker flood corrs (esp. max extent) vs. average (0.53)
        - Along north river, most TPs are from max extent
            - Max extent should be in the river, but isn't
            y_top = 400
            x_left = 1380
            y_bottom = 1100
            x_right = 2100
     - 4080_LC08_028033_20130806_1
        - Very few preds
        - Unc map has weird spackling near FNs
        - Weaker max extent corr (0.31)
     - 4594_LC08_022035_20180404_1
        - Very few preds
        - Lots of real clouds
     - 4444_LC08_044034_20170222_1 --> REMOVE
        - Cloud shadows account for some of the FNs, might be throwing off training
        - Max extent well correlated with flooding (0.42)
     - 4444_LC08_043034_20170303_1
        - Very few preds
        - Max extent well correlated with flooding (0.6)
        - Reservoir is FN - should remove
        - Weird patches of large squares (TP) around river (FN)
            - These are from TWI and SPI ?
     - 4101_LC08_027039_20131103_1
        - Pretty sparse flooding far away from rivers - is it even flooding?
     - 4444_LC08_043035_20170303_1
     - 4444_LC08_044034_20170222_1
     - 4468_LC08_022035_20170503_1
     - 4514_LC08_027033_20170826_1
     - 4594_LC08_022035_20180404_1
   - High performing
     - '4101_LC08_027038_20131103_1',
     - '4101_LC08_027038_20131103_2',
     - '4101_LC08_027039_20131103_1',
     - '4115_LC08_021033_20131227_1',
     - '4115_LC08_021033_20131227_2'
   - Good batch comparison images (high LR, middle NN, low RF)
     - 4444_LC08_044033_20170222_4
        - Good one to use, 30% bottom middle farmland
     - 4468_LC08_024036_20170501_2
        - 70%, top left, farmland and around river bend
     - 4115_LC08_021033_20131227_1
        - 90%, top right
     - 4444_LC08_045032_20170301_1
        - 90%, bottom center
   - Greatest variance in RCTs
     - 4101_LC08_027038_20131103_2
        - There isn't really any flooding. Consider REMOVING
     - 4477_LC08_022033_20170519_1
        - 
     - 4337_LC08_026038_20160325_1
     - 4594_LC08_022034_20180404_1
     - 4469_LC08_015036_20170502_1
     - 4468_LC08_022035_20170503_1
        - Very few preds. Feature data just didn't track well with flood
            y_top = 1050
            x_left = 280
            y_bottom = 1500
            x_right = 670
        - Even if feature data does track well, the model may not have had enough positive examples to 
        learn this pattern if clouds covered most of them
            y_top = 1400
            x_left = 850
            y_bottom = 1730
            x_right = 1300
     - 4444_LC08_044033_20170222_4
        - Trials 2/3 differ greatly. 90% cloud cover area in the bottom is good to compare in 4 different trials

### Removed or changed images list
- 4444_LC08_044034_20170222_1 (removed for FPs)
- 4337_LC08_026038_20160325_1 (changed because perm mask didn't have reservoir in it)
- 4101_LC08_027038_20131103_2 (removed, no flooding)
- 4594_LC08_022035_20180404_1 (not much flooding, tons of clouds and some FPs)

------
 
### How to set up QERM servers

- Clone git repository
- Build conda env from environment.yml file in repo
- Download data (images + cloud files) from Google Drive
    - Move  into Data folder in cloned repo
- Run scripts from terminal
- When finished, upload data to Google Drive
- Note: Don't develop or edit scripts on server since you will need to set up a git key to commit any changes.