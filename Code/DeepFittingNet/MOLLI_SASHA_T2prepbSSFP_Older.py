import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
from dataaccess import loadInvivoDataOlder

import xlsxwriter
import os
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utilities import mean_std_ROI

TimeScaling = 1000;
TimeScalingFactor =1/TimeScaling

cwdpath=Path(os.getcwd())

def TestingInvivo(loadDataPath,TxDispMax, worksheet , xlsxRowrow,xlsxCol , seqTag = 'MOLLI53',LLCorrection=False, Para='T1'):
    try:
        stRow = xlsxRow
        stCol = xlsxCol
        #write 1st Row
        worksheet.write(stRow,stCol,seqTag)  #Col #A
        stRow += 1
        stCol += 1
        worksheet.write(stRow,stCol, 'LV Myo')  #Col #C
        stRow += 1
        worksheet.write(stRow,stCol, 'Name')  #Col #C
        worksheet.write(stRow,stCol+1, 'fitTxMean')  #Col #D
        worksheet.write(stRow,stCol + 2, 'fitTxSD')

        worksheet.write(stRow-1, stCol+8, 'Blood pool')
        worksheet.write(stRow, stCol+8, 'Name')
        worksheet.write(stRow, stCol+9, 'fitTxMean')
        worksheet.write(stRow, stCol + 10, 'fitTxSD')

        worksheet.write(stRow-1, stCol+16, 'Sep Myo')
        worksheet.write(stRow, stCol+16, 'Name')
        worksheet.write(stRow, stCol+17, 'fitTxMean')
        worksheet.write(stRow, stCol + 18, 'fitTxSD')

        #load data
        sigsTisTsatsTEs_, fitA_, fitB_, fitTxMap_, roiMyoMask_, roiBPMask_, roiSepMask_, AHAMask_, subjectsLists_ = loadInvivoDataOlder(
            loadDataPath)

        #for storing the resutls from DL net
        Tx_MyoMeanandStdLst = [[],[],[],[]]
        Tx_BPMeanandStdLst = [[],[],[],[]]
        Tx_SepMeanandStdLst = [[],[],[],[]]

        SubjectsLists = list()

        allResults = {}

        predT1_ = np.array( fitTxMap_ )

        MyoT1MUandSD = np.array( mean_std_ROI(predT1_ * TimeScaling, roiMyoMask_,Para))
        BpT1MUandSD = np.array( mean_std_ROI(predT1_ * TimeScaling, roiBPMask_,Para))
        SepT1MUandSD = np.array( mean_std_ROI(predT1_ * TimeScaling, roiSepMask_,Para))

        Tx_MyoMeanandStdLst[0].extend(MyoT1MUandSD[0,:])
        Tx_MyoMeanandStdLst[1].extend(MyoT1MUandSD[1,:])

        Tx_BPMeanandStdLst[0].extend(BpT1MUandSD[0,:])
        Tx_BPMeanandStdLst[1].extend(BpT1MUandSD[1,:])

        Tx_SepMeanandStdLst[0].extend(SepT1MUandSD[0,:])
        Tx_SepMeanandStdLst[1].extend(SepT1MUandSD[1,:])

        allResults['Tx_MyoMeanandStdLst'] = np.array(Tx_MyoMeanandStdLst).transpose()

        allResults['Tx_BPMeanandStdLst'] = np.array(Tx_BPMeanandStdLst).transpose()

        allResults['Tx_SepMeanandStdLst'] = np.array(Tx_SepMeanandStdLst).transpose()

        allResults['subjectsLists'] = subjectsLists_

        rangeSt = stRow+2
        for ix in range(0, MyoT1MUandSD.shape[1]):
            stRow += 1
            worksheet.write(stRow, stCol, subjectsLists_[ix])
            worksheet.write(stRow, stCol + 1, Tx_MyoMeanandStdLst[0][ix])
            worksheet.write(stRow, stCol + 2, Tx_MyoMeanandStdLst[1][ix])

            worksheet.write(stRow, stCol + 9,  Tx_BPMeanandStdLst[0][ix])
            worksheet.write(stRow, stCol + 10, Tx_BPMeanandStdLst[1][ix])

            worksheet.write(stRow, stCol + 17, Tx_SepMeanandStdLst[0][ix])
            worksheet.write(stRow, stCol + 18, Tx_SepMeanandStdLst[1][ix])


        rangeEd = stRow+1
        #
        stRow += 2
        worksheet.write(stRow,stCol, 'Mean')
        worksheet.write(stRow,stCol+1, '=AVERAGE(C'+str(rangeSt)+':C'+ str(rangeEd)+')')        #C col
        worksheet.write(stRow,stCol + 2, '=AVERAGE(D'+str(rangeSt)+':D'+ str(rangeEd)+')')      #D col

        worksheet.write(stRow, stCol+9, '=AVERAGE(K'+str(rangeSt)+':K'+ str(rangeEd)+')')       #K col
        worksheet.write(stRow, stCol + 10, '=AVERAGE(L'+str(rangeSt)+':L'+ str(rangeEd)+')')    #L col

        worksheet.write(stRow, stCol+17, '=AVERAGE(S'+str(rangeSt)+':S'+ str(rangeEd)+')')      #S col
        worksheet.write(stRow, stCol + 18, '=AVERAGE(T'+str(rangeSt)+':T'+ str(rangeEd)+')')    #T col

        stRow += 1
        worksheet.write(stRow,stCol, 'SD')
        worksheet.write(stRow,stCol+1, '=STDEV(C'+str(rangeSt)+':C'+ str(rangeEd)+')')        #C col
        worksheet.write(stRow,stCol + 2, '=STDEV(D'+str(rangeSt)+':D'+ str(rangeEd)+')')      #D col

        worksheet.write(stRow, stCol+9, '=STDEV(K'+str(rangeSt)+':K'+ str(rangeEd)+')')       #K col
        worksheet.write(stRow, stCol + 10, '=STDEV(L'+str(rangeSt)+':L'+ str(rangeEd)+')')    #L col

        worksheet.write(stRow, stCol+17, '=STDEV(S'+str(rangeSt)+':S'+ str(rangeEd)+')')      #S col
        worksheet.write(stRow, stCol + 18, '=STDEV(T'+str(rangeSt)+':T'+ str(rangeEd)+')')    #T col

        # export AHA results
        stRow += 5
        worksheet.write(stRow, stCol, 'AHA-24 Segments T1')  # Col #C
        stRow += 1
        worksheet.write(stRow, stCol, 'Name')  # Col #C
        worksheet.write(stRow, stCol + 1, '1')  # Col #D
        worksheet.write(stRow, stCol + 2, '2')
        worksheet.write(stRow, stCol + 3, '3')
        worksheet.write(stRow, stCol + 4, '4')
        worksheet.write(stRow,stCol + 5, '5')
        worksheet.write(stRow,stCol + 6, '6')
        worksheet.write(stRow,stCol + 7, '7')
        worksheet.write(stRow, stCol + 8, '8')
        worksheet.write(stRow, stCol + 9, '9')
        worksheet.write(stRow, stCol + 10, '10')
        worksheet.write(stRow, stCol + 11, '11')
        worksheet.write(stRow, stCol + 12, '12')
        worksheet.write(stRow, stCol + 13, '13')
        worksheet.write(stRow, stCol + 14, '14')
        worksheet.write(stRow, stCol + 15, '15')
        worksheet.write(stRow, stCol + 16, '16')
        worksheet.write(stRow, stCol + 17, '17')
        worksheet.write(stRow, stCol + 18, '18')
        worksheet.write(stRow, stCol + 19, '19')
        worksheet.write(stRow, stCol + 20, '20')
        worksheet.write(stRow, stCol + 21, '21')
        worksheet.write(stRow, stCol + 22, '22')
        worksheet.write(stRow, stCol + 23, '23')
        worksheet.write(stRow, stCol+ 24, '24')

        for ix in range(0, MyoT1MUandSD.shape[1]):
            stRow += 1
            TxMap =predT1_[ix,0,]* TimeScaling
            AHAmask_ix = AHAMask_[ix,0,]
            worksheet.write(stRow, stCol, subjectsLists_[ix])
            for ij in range(0,24):
               roiVals = TxMap[np.nonzero( AHAmask_ix[:,:,ij])]
               if Para=='T1':
                   roiVals = roiVals[roiVals<3000]
               elif Para =='T2':
                   roiVals = roiVals[roiVals < 250]
               try:
                   try:
                       meanVal = roiVals.mean()
                   except:
                       meanVal = 0
                   worksheet.write(stRow, stCol + 1+ij, meanVal)
                   try:
                       stdVal = roiVals.std()
                   except:
                       stdVal = 0
                   worksheet.write(stRow, stCol + 1 + ij+30, stdVal)
               except:
                   print('Empty segment')


        #AP col
        # save maps as mat and png format file
        stRow += 2


        for ix in range(0,MyoT1MUandSD.shape[1]):
            subjectName = subjectsLists_[ix]

            predT1 = predT1_[ix,:,:,:].squeeze() * TimeScaling

            TxMap=np.zeros((predT1.shape[0],predT1.shape[1],1,2))
            TxMap[:,:,0,0] = predT1

            fig, axs = plt.subplots()

            axs.set_title(seqTag+' Map')
            plt1=axs.imshow(predT1, cmap='jet', vmin=0, vmax=TxDispMax)
            axs.axis('off')
            divider1 = make_axes_locatable(axs)
            colorbar_axes = divider1.append_axes("right",
                                                size="10%",
                                                pad=0.1)

            axs.axis('off')

            fig.colorbar(plt1, cax=colorbar_axes)

            # save png file
            fig.savefig( os.path.normpath(cwdpath.parent.parent)+'//Results//Older//' + seqTag+'//'+subjectName+'.png',dpi = 300)
            plt.close(fig)
    except Exception as e:
        print(e)
    return stRow
workbook = xlsxwriter.Workbook( os.path.normpath(cwdpath.parent.parent)+'//Results//Older//'+'MOLLI_SASHA_T2prepbSSFP.xlsx')
worksheet = workbook.add_worksheet()

xlsxRow = 0
xlsxCol = 0
#MOLLI
xlsxRow = TestingInvivo( os.path.normpath(cwdpath.parent.parent)+'//data//Older//MOLLI.mat',2500,worksheet, xlsxRow,xlsxCol,'MOLLI',False,'T1')
#SASHA
xlsxRow = TestingInvivo(os.path.normpath(cwdpath.parent.parent)+'//data//Older//SASHA.mat',2500,worksheet, xlsxRow,xlsxCol,'SASHA',False, 'T1')
#T2prepbSSFP
xlsxRow = TestingInvivo(os.path.normpath(cwdpath.parent.parent)+'//data//Older//T2prepbSSFP.mat',100,worksheet, xlsxRow,xlsxCol,'T2prepbSSFP',False, 'T2')

workbook.close()

