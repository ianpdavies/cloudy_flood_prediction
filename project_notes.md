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
 
Custom loss function for NN? Macro F1 that uses probs instead of 0/1 so the loss function is differentiable
https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric

CNN?
https://bamos.github.io/2016/08/09/deep-completion/

Found updated DFO database file but no polygon extent, only lat/long. Can add buffer around it to get images in similar way.

------
 
### How to set up QERM servers

- Clone git repository
- Build conda env from environment.yml file in repo
- Download data (images + cloud files) from Google Drive
    - Move  into Data folder in cloned repo
- Run scripts from terminal
- When finished, upload data to Google Drive
- Note: Don't develop or edit scripts on server since you will need to set up a git key to commit any changes.