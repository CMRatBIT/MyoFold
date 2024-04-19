
import numpy as np

def mean_std_ROI(x, mask,Tag):
    meanstdValsArr = [ [0 for i in range(x.shape[0])] for i in range(2)]
    for i in range(0, x.shape[0]):
        xs = x[i,]
        roiVals = xs[np.nonzero(mask[i,])]
        if Tag=='T1':
            roiVals = roiVals[roiVals < 3000]
        elif Tag == 'T2':
            roiVals = roiVals[roiVals < 250]

        if roiVals.size ==0:
            meanstdValsArr[0][i] = 0
            meanstdValsArr[1][i] = 0
        else:
            meanstdValsArr[0][i] = roiVals.mean()
            meanstdValsArr[1][i] = roiVals.std()

    return meanstdValsArr
# r = mean(|(x/y-1)*100})
def meanAbsoluterRelativeErr(x,y, mask):
    meanAbsoluterRelativeErrArr = [0 for i in range(x.shape[0])]
    for i in range(0, x.shape[0]):
        xs = x[i,]
        ys = y[i,]
        roiValsX = xs[np.nonzero(mask[i,])]
        roiValsY = ys[np.nonzero(mask[i,])]
        #relativeErr = np.abs((roiValsX-roiValsY))
        relativeErr = np.abs((roiValsX / roiValsY - 1) * 100)
        meanAbsoluterRelativeErrArr[i] = relativeErr.mean()
    return meanAbsoluterRelativeErrArr