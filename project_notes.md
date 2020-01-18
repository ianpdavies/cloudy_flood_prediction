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
 
4. Thin clouds vs thick clouds? At same % cover.

5. Logistic regression, SVM, RF results
 - Run log reg and RF on all images at all pctls
 - Take the best performing one and tune hyperparameters
 - Compare to NN
 - Logistic regression
    - Performs better than NN, and faster (I think)
    - But can't be run in parallel, so there is a solid floor to training/prediction time
 - RF
    - Appears to perform poorly using default hyperparams - high precision and low recall, super conservative estimates
    - Might be able to improve with tuning 

6. Train on small dataset to reduce sparseness of the data
 -  Use binary dilation to buffer flooded pixels
    - Train with varying dilation iterations 10-50 around (a) all water and (b) just flooded pixels
    - Train with dilation iterations around all water + random pixels selected throughout image
    - Depending on the result above, train model with buffers around clouds vs. only around water/flood
        - binary_dilation buffers around NaNs, so will have to turn those to 0 while making the mask to remove cloud
        buffers
    - BUFFER EXPERIMENT:
        - Run the above model permutations with half of images and compare to log reg with half of images   
7. Random cloud trials
 - According to Max (CSSS consulting) I should run 30-100 trials. Time permitting, I can do this on the QERM servers.

8. Uncertainty
 - Get uncertainty estimates from logistic regression
 - Compare these uncertainty estimates to those from the NN. 
 MAKE SURE CLOUD THRESHOLDING IS WORKING CORRECTLY AND THAT ISN'T WHY LOG REG IS DOING SO WELL!!!
 Test remove_perm=True/False again with log reg. The log reg buffer tests 
 ------
 
### How to set up QERM servers

- Clone git repository
- Build conda env from environment.yml file in repo
- Download data (images + cloud files) from Google Drive
    - Move  into Data folder in cloned repo
- Run scripts from terminal
- When finished, upload data to Google Drive
- Note: Don't develop or edit scripts on server since you will need to set up a git key to commit any changes.