# models
import sys
from torch import optim
import torch
from torch.autograd import Variable
print('is CUDA available:')
print(torch.cuda.is_available())
import numpy as np
from Model import DeepFittingNet
import matplotlib.pyplot as plt
import h5py
from dataaccess import loadInvivoDataYoung
from utilities import mean_std_ROI
import xlsxwriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from pathlib import Path
#scaling factor for input inverion time and output T1,
TimeScaling = 1000;
TimeScalingFactor =1/TimeScaling


print(torch.cuda.is_available())
#loading trained model
myoRelaxationNet = DeepFittingNet()
if torch.cuda.is_available():
    myoRelaxationNet = torch.nn.DataParallel(myoRelaxationNet).cuda()
else:
    myoRelaxationNet.to(params.device)
cwdpath=Path(os.getcwd())


model = torch.load(os.path.normpath(cwdpath.parent.parent)+'//Model//MODEL_EPOCH2032.pth' )
myoRelaxationNet.load_state_dict(model['state_dict'])



def Predication(net,sigsTisTsatsTEs):
    predA = []
    predB = []
    predT1 = []
    predT2 = []
    net.eval()
    bs = 1
    try:
        with torch.no_grad():

            val_N = sigsTisTsatsTEs.shape[0]
            val_lst = list(range(0,val_N))

            for idx in range(0, val_N, bs):

                preSubjectsLst = []
                postSubjectsLst = []

                try:

                    X = Variable(torch.FloatTensor(sigsTisTsatsTEs[val_lst[idx:idx + bs]])).to('cuda:0')
                    xs = X.shape
                    X = X.permute((0, 3, 4, 1, 2)).reshape((xs[0] * xs[3] * xs[4], xs[1], xs[2]))
                    # predicated by net
                    y_pred = net(X.to('cuda:0')).to('cuda:0')

                    T1T2 = y_pred.cpu().data.numpy()

                    predT1.append(T1T2[:, 0].reshape((xs[0], xs[3], xs[4])))
                    predT2.append(T1T2[:, 1].reshape((xs[0], xs[3], xs[4])))

                except Exception as e:
                    print(e)
    except Exception as e:
        print(e)

    return predT1, predT2


def TestingInvivo(net, loadDataPath,TxDispMax, worksheet , xlsxRowrow,xlsxCol , seqTag = 'MOLLI53',LLCorrection=False):
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
        worksheet.write(stRow,stCol+3, 'netTxMean')
        worksheet.write(stRow,stCol + 4, 'netTxSD')

        worksheet.write(stRow-1, stCol+8, 'Blood pool')
        worksheet.write(stRow, stCol+8, 'Name')
        worksheet.write(stRow, stCol+9, 'fitTxMean')
        worksheet.write(stRow, stCol + 10, 'fitTxSD')
        worksheet.write(stRow, stCol+11, 'netTxMean')
        worksheet.write(stRow, stCol + 12, 'netTxSD')

        worksheet.write(stRow-1, stCol+16, 'Sep Myo')
        worksheet.write(stRow, stCol+16, 'Name')
        worksheet.write(stRow, stCol+17, 'fitTxMean')
        worksheet.write(stRow, stCol + 18, 'fitTxSD')
        worksheet.write(stRow, stCol+19, 'netTxMean')
        worksheet.write(stRow, stCol + 20, 'netTxSD')

        #load data
        sigsTisTsatsTEs_, fitA_, fitB_, fitTxMap_, roiMyoMask_, roiBPMask_, roiSepMask_, AHAMask_, subjectsLists_ = loadInvivoDataYoung(
            loadDataPath)

        #for storing the resutls from DL net
        Tx_MyoMeanandStdLst = [[],[],[],[]]
        Tx_BPMeanandStdLst = [[],[],[],[]]
        Tx_SepMeanandStdLst = [[],[],[],[]]

        SubjectsLists = list()

        allResults = {}
        #
        predT1_,predT2_ = Predication(net,sigsTisTsatsTEs_)
        predT1_ = np.array( predT1_ )
        predT2_ = np.array( predT2_ )

        MyoT1MUandSD = np.array( mean_std_ROI(np.squeeze(predT1_) * TimeScaling, roiMyoMask_[:,0,:,:],'T1'))
        BpT1MUandSD = np.array( mean_std_ROI(np.squeeze(predT1_) * TimeScaling, roiBPMask_[:,0,:,:],'T1'))
        SepT1MUandSD = np.array( mean_std_ROI(np.squeeze(predT1_) * TimeScaling, roiSepMask_[:,0,:,:],'T1'))

        Tx_MyoMeanandStdLst[0].extend(MyoT1MUandSD[0,:])
        Tx_MyoMeanandStdLst[1].extend(MyoT1MUandSD[1,:])

        Tx_BPMeanandStdLst[0].extend(BpT1MUandSD[0,:])
        Tx_BPMeanandStdLst[1].extend(BpT1MUandSD[1,:])

        Tx_SepMeanandStdLst[0].extend(SepT1MUandSD[0,:])
        Tx_SepMeanandStdLst[1].extend(SepT1MUandSD[1,:])

        MyoT2MUandSD = np.array(mean_std_ROI(np.squeeze(predT2_) * TimeScaling, roiMyoMask_[:,1,:,:],'T2'))
        BpT2MUandSD = np.array(mean_std_ROI(np.squeeze(predT2_) * TimeScaling, roiBPMask_[:,1,:,:],'T2'))
        SepT2MUandSD = np.array(mean_std_ROI(np.squeeze(predT2_) * TimeScaling, roiSepMask_[:,1,:,:],'T2'))

        Tx_MyoMeanandStdLst[2].extend(MyoT2MUandSD[0, :])
        Tx_MyoMeanandStdLst[3].extend(MyoT2MUandSD[1, :])
        allResults['Tx_MyoMeanandStdLst'] = np.array(Tx_MyoMeanandStdLst).transpose()

        Tx_BPMeanandStdLst[2].extend(BpT2MUandSD[0, :])
        Tx_BPMeanandStdLst[3].extend(BpT2MUandSD[1, :])
        allResults['Tx_BPMeanandStdLst'] = np.array(Tx_BPMeanandStdLst).transpose()

        Tx_SepMeanandStdLst[2].extend(SepT2MUandSD[0, :])
        Tx_SepMeanandStdLst[3].extend(SepT2MUandSD[1, :])
        allResults['Tx_SepMeanandStdLst'] = np.array(Tx_SepMeanandStdLst).transpose()


        allResults['subjectsLists'] = subjectsLists_

        rangeSt = stRow+2
        for ix in range(0, MyoT1MUandSD.shape[1]):
            stRow += 1
            worksheet.write(stRow, stCol, subjectsLists_[ix])
            worksheet.write(stRow, stCol + 1, Tx_MyoMeanandStdLst[0][ix])
            worksheet.write(stRow, stCol + 2, Tx_MyoMeanandStdLst[1][ix])
            worksheet.write(stRow, stCol + 3, Tx_MyoMeanandStdLst[2][ix])
            worksheet.write(stRow, stCol + 4, Tx_MyoMeanandStdLst[3][ix])

            worksheet.write(stRow, stCol + 9,  Tx_BPMeanandStdLst[0][ix])
            worksheet.write(stRow, stCol + 10, Tx_BPMeanandStdLst[1][ix])
            worksheet.write(stRow, stCol + 11, Tx_BPMeanandStdLst[2][ix])
            worksheet.write(stRow, stCol + 12, Tx_BPMeanandStdLst[3][ix])

            worksheet.write(stRow, stCol + 17, Tx_SepMeanandStdLst[0][ix])
            worksheet.write(stRow, stCol + 18, Tx_SepMeanandStdLst[1][ix])
            worksheet.write(stRow, stCol + 19, Tx_SepMeanandStdLst[2][ix])
            worksheet.write(stRow, stCol + 20, Tx_SepMeanandStdLst[3][ix])

        rangeEd = stRow+1
        #
        stRow += 2
        worksheet.write(stRow,stCol, 'Mean')
        worksheet.write(stRow,stCol+1, '=AVERAGE(C'+str(rangeSt)+':C'+ str(rangeEd)+')')        #C col
        worksheet.write(stRow,stCol + 2, '=AVERAGE(D'+str(rangeSt)+':D'+ str(rangeEd)+')')      #D col
        worksheet.write(stRow,stCol+3, '=AVERAGE(E'+str(rangeSt)+':E'+ str(rangeEd)+')')        #E col
        worksheet.write(stRow,stCol + 4,'=AVERAGE(F'+str(rangeSt)+':F'+ str(rangeEd)+')')       #F col

        worksheet.write(stRow, stCol+9, '=AVERAGE(K'+str(rangeSt)+':K'+ str(rangeEd)+')')       #K col
        worksheet.write(stRow, stCol + 10, '=AVERAGE(L'+str(rangeSt)+':L'+ str(rangeEd)+')')    #L col
        worksheet.write(stRow, stCol+11, '=AVERAGE(M'+str(rangeSt)+':M'+ str(rangeEd)+')')      #M col
        worksheet.write(stRow, stCol + 12, '=AVERAGE(N'+str(rangeSt)+':N'+ str(rangeEd)+')')    #N col

        worksheet.write(stRow, stCol+17, '=AVERAGE(S'+str(rangeSt)+':S'+ str(rangeEd)+')')      #S col
        worksheet.write(stRow, stCol + 18, '=AVERAGE(T'+str(rangeSt)+':T'+ str(rangeEd)+')')    #T col
        worksheet.write(stRow, stCol+19, '=AVERAGE(U'+str(rangeSt)+':U'+ str(rangeEd)+')')      #U col
        worksheet.write(stRow, stCol + 20,'=AVERAGE(v'+str(rangeSt)+':V'+ str(rangeEd)+')')     #V col

        stRow += 1
        worksheet.write(stRow,stCol, 'SD')
        worksheet.write(stRow,stCol+1, '=STDEV(C'+str(rangeSt)+':C'+ str(rangeEd)+')')        #C col
        worksheet.write(stRow,stCol + 2, '=STDEV(D'+str(rangeSt)+':D'+ str(rangeEd)+')')      #D col
        worksheet.write(stRow,stCol+3, '=STDEV(E'+str(rangeSt)+':E'+ str(rangeEd)+')')        #E col
        worksheet.write(stRow,stCol + 4,'=STDEV(F'+str(rangeSt)+':F'+ str(rangeEd)+')')       #F col

        worksheet.write(stRow, stCol+9, '=STDEV(K'+str(rangeSt)+':K'+ str(rangeEd)+')')       #K col
        worksheet.write(stRow, stCol + 10, '=STDEV(L'+str(rangeSt)+':L'+ str(rangeEd)+')')    #L col
        worksheet.write(stRow, stCol+11, '=STDEV(M'+str(rangeSt)+':M'+ str(rangeEd)+')')      #M col
        worksheet.write(stRow, stCol + 12, '=STDEV(N'+str(rangeSt)+':N'+ str(rangeEd)+')')    #N col

        worksheet.write(stRow, stCol+17, '=STDEV(S'+str(rangeSt)+':S'+ str(rangeEd)+')')      #S col
        worksheet.write(stRow, stCol + 18, '=STDEV(T'+str(rangeSt)+':T'+ str(rangeEd)+')')    #T col
        worksheet.write(stRow, stCol+19, '=STDEV(U'+str(rangeSt)+':U'+ str(rangeEd)+')')      #U col
        worksheet.write(stRow, stCol + 20,'=STDEV(v'+str(rangeSt)+':V'+ str(rangeEd)+')')     #V col

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
               roiVals = roiVals[roiVals < 3000]
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

        stRow += 5
        worksheet.write(stRow, stCol, 'AHA-24 Segments T2')  # Col #C
        stRow += 1
        worksheet.write(stRow, stCol, 'Name')  # Col #C
        worksheet.write(stRow, stCol + 1, '1')  # Col #D
        worksheet.write(stRow, stCol + 2, '2')
        worksheet.write(stRow, stCol + 3, '3')
        worksheet.write(stRow, stCol + 4, '4')
        worksheet.write(stRow, stCol + 5, '5')
        worksheet.write(stRow, stCol + 6, '6')
        worksheet.write(stRow, stCol + 7, '7')
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
        worksheet.write(stRow, stCol + 24, '24')

        for ix in range(0, MyoT1MUandSD.shape[1]):
            stRow += 1
            TxMap = predT2_[ix, 0,] * TimeScaling
            AHAmask_ix = AHAMask_[ix, 1,]
            worksheet.write(stRow, stCol, subjectsLists_[ix])
            for ij in range(0, 24):
                roiVals = TxMap[np.nonzero(AHAmask_ix[:, :, ij])]

                roiVals = roiVals[roiVals<250]
                try:
                    try:
                        meanVal = roiVals.mean()
                    except:
                        meanVal = 0
                    worksheet.write(stRow, stCol + 1 + ij , meanVal)
                    try:
                        stdVal = roiVals.std()
                    except:
                        stdVal = 0
                    worksheet.write(stRow, stCol + 1 + ij + 30, stdVal)


                except:
                    print('Empty segment')

        stRow += 2


        for ix in range(0,sigsTisTsatsTEs_.shape[0]):
            subjectName = subjectsLists_[ix]

            predT1 = predT1_[ix,:,:,:].squeeze() * TimeScaling
            predT2 = predT2_[ix, :, :, :].squeeze() * TimeScaling


            TxMap=np.zeros((predT1.shape[0],predT1.shape[1],1,2))
            TxMap[:,:,0,0] = predT1
            TxMap[:, :, 0, 1] = predT2

            fig, axs = plt.subplots(1, 2)

            axs[0].set_title('MyoFold T1 Map')
            plt1=axs[0].imshow(predT1, cmap='jet', vmin=0, vmax=2500)
            axs[0].axis('off')
            divider1 = make_axes_locatable(axs[0])
            colorbar_axes = divider1.append_axes("right",
                                                size="10%",
                                                pad=0.1)
            axs[1].set_title('MyoFold T2 Map')
            plt2 = axs[1].imshow(predT2, cmap='jet', vmin=0, vmax=100)
            axs[1].axis('off')
            divider2 = make_axes_locatable(axs[1])
            fig.colorbar(plt1, cax=colorbar_axes)
            colorbar_axes = divider2.append_axes("right",
                                                 size="10%",
                                                 pad=0.1)
            fig.colorbar(plt2, cax=colorbar_axes)

            # save png file
            fig.savefig(  os.path.normpath(cwdpath.parent.parent)+'//Results//Young//' + seqTag+'//'+subjectName+'.png',dpi = 300)
            plt.close(fig)
    except Exception as e:
        print(e)
    return stRow
workbook = xlsxwriter.Workbook( os.path.normpath(cwdpath.parent.parent)+'//Results//Young//TestingInvivo_MyoFold.xlsx')
worksheet = workbook.add_worksheet()

xlsxRow = 0
xlsxCol = 0

xlsxRow = TestingInvivo(myoRelaxationNet,os.path.normpath(cwdpath.parent.parent)+'//data//Young//MyoFold.mat',2000,worksheet, xlsxRow,xlsxCol,'MyoFold',LLCorrection=False)

workbook.close()

