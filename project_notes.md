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

3. Visualization
 - Can clouds be just borders with transparent fill? To better visualize overlay.
 
4. Thin clouds vs thick clouds? At same % cover.

5. Logistic regression, SVM, RF results
 - 