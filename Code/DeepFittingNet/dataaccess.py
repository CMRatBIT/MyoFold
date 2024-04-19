from scipy.io import loadmat
import numpy as np
import h5py


def loadInvivoDataYoung(matFileName=''):
    print('Start loading invivo dataset..... ')
    print(matFileName);
    data = loadmat(matFileName)
    try:
        fitA = np.array(data['fitA'])
    except:
        fitA = []
    try:
        fitB = np.array(data['fitB'])
    except:
        fitB = []
    try:
        fitTxMap =  np.array(data['fitTxMap'])
    except:
        fitTxMap = []
    try:
        sigsTisTsatsTEs = np.array(data['SigsTinvsTsatsTEs'])
    except:
        sigsTisTsatsTEs = []
    try:
        roiMyoMask = np.array(data['ROIMyoMask'])
    except:
        roiMyoMask = []

    try:
        roiBPMask = np.array(data['ROIBPMask'])
    except:
        roiBPMask = []

    try:
        roiSepMask = np.array(data['ROISepMask'])
    except:
        roiSepMask = []

    try:
        AHAMasks = np.array(data['ROIAHAMask'])
    except:
        AHAMasks = []

    try:
        subjectsLists = data['subjectsLists']
        subjectsLists = subjectsLists.tolist()
    except:
        subjectsLists = []

    return sigsTisTsatsTEs, fitA, fitB,fitTxMap, roiMyoMask,roiBPMask,roiSepMask,AHAMasks, subjectsLists


def loadInvivoDataOlder(matFileName=''):
    print('Start loading invivo dataset..... ')
    print(matFileName);
    # data = loadmat(matFileName)
    matFile = h5py.File(matFileName)
    data = matFile
    try:
        fitA = np.array(data['fitA'])
        fitA = np.transpose(fitA, (3, 2, 1,0))
    except:
        fitA = []
    try:
        fitB = np.array(data['fitB'])
        fitB = np.transpose(fitB, (3, 2, 1, 0))
    except:
        fitB = []
    try:
        fitTxMap =  np.array(data['fitTxMap'])
        fitTxMap = np.transpose(fitTxMap, (3, 2, 1, 0))
    except:
        fitTxMap = []
    try:
        sigsTisTsatsTEs = np.array(data['SigsTinvsTsatsTEs'])
        sigsTisTsatsTEs = np.transpose(sigsTisTsatsTEs, (4,3, 2, 1, 0))
    except:
        sigsTisTsatsTEs = []
    try:
        roiMyoMask = np.array(data['ROIMyoMask'])
        roiMyoMask = np.transpose(roiMyoMask, ( 3, 2, 1, 0))
    except:
        roiMyoMask = []

    try:
        roiBPMask = np.array(data['ROIBPMask'])
        roiBPMask = np.transpose(roiBPMask, (3, 2, 1, 0))
    except:
        roiBPMask = []

    try:
        roiSepMask = np.array(data['ROISepMask'])
        roiSepMask = np.transpose(roiSepMask, (3, 2, 1, 0))
    except:
        roiSepMask = []

    try:
        AHAMasks = np.array(data['ROIAHAMask'])
        AHAMasks = np.transpose(AHAMasks, (4,3, 2, 1, 0))
    except:
        AHAMasks = []

    try:
        subjectsLists=[]
        subjectsListsLoad = np.array(data['subjectsLists'])
        # subjectsLists = subjectsLists.tolist()
        subjectsListsLoad = np.transpose(subjectsListsLoad, (1, 0))

        for ix in range(0,subjectsListsLoad.shape[0] ):
            tempChar=''
            for ij in range(0, subjectsListsLoad.shape[1]):
                tempChar+=(chr(subjectsListsLoad[ix][ij]))
            subjectsLists.append(tempChar)

    except:
        subjectsLists = []

    return sigsTisTsatsTEs, fitA, fitB,fitTxMap, roiMyoMask,roiBPMask,roiSepMask,AHAMasks, subjectsLists






