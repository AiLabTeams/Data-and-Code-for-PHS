from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
import WavePackage
import PressPackage
import math
from scipy.optimize import curve_fit
from scipy.fftpack import fft,ifft
import random
import Gramme
import WavePackage
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch
from torch import nn
mod = #"resnet20230320.410162445"
LightWaveAmplitudeListsRaw = []
LightWaveAmplitudeListsFilter = []
LightWaveAmplitudeLists = []
LightWaveLengthList = []
LightWaveTimeList = []
LightWaveTimeSecondList = []
PressTimeList = []
PressAmplitudeList = []
NewPressAmplitudeList = []
PressAmplitudelogList = []
lightfilecount=0
PressTimeToFix=0
LightTimeToFix=0
GASFMatrixListRaw = []
GADFMatrixListRaw = []
GASFMatrixList = []
GADFMatrixList = []
AllGASFMatrixList = []
AllGADFMatrixList = []

Allfftlist=[]
AllPressTimeToFix=[]
AllLightTimeToFix=[]
AllPressTimeList = []
AllPressAmplitudelogList = []
AllLightWaveTimeSecondList=[]
AllLightWaveAmplitudeLists=[]

fftabslistlist = []
fftanglelist = []

Allxlist=[]
Allylist=[]
xtrainlist=[]
ytrianlist=[]
yprelist=[]
xprelist=[]
timedis=300
channel=4
batch=1
lr, num_epochs = 0.01 , 200
channelmut=8
refuc=nn.Sigmoid
class MyNet(nn.Module):
   def __init__(self):
       super(MyNet, self).__init__()
       self.convs = nn.Sequential(
           nn.Conv2d(1, 2 *channel, 5, 1),  # in_channels, out_channels,kernel_size
           refuc(),
           nn.BatchNorm2d(2 * channel),
       )
       self.convd = nn.Sequential(
           nn.Conv2d(1, 2 *channel, 5, 1),  # in_channels, out_channels,kernel_size
           refuc(),
           nn.BatchNorm2d(2 * channel),
       )
       self.makelayers1 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3
           nn.BatchNorm2d(channel),
           refuc(),
       )
       self.makelayers2 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3
           nn.BatchNorm2d(channel),
           refuc(),
       )
       self.makelayers3 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3
           nn.BatchNorm2d(channel),
           refuc(),
       )
       self.makelayers4 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3
           nn.BatchNorm2d(channel),
           refuc(),
       )
       self.makelayers5 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3
       )
       self.makelayerd1 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3
           nn.BatchNorm2d(channel),
           refuc(),
       )
       self.makelayerd2 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3
           nn.BatchNorm2d(channel),
           refuc(),
       )
       self.makelayerd3 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3
           nn.BatchNorm2d(channel),
           refuc(),
       )
       self.makelayerd4 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3
           nn.BatchNorm2d(channel),
           refuc(),
       )
       self.makelayerd5 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3

       )
       self.downlayers1=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.downlayers2=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.downlayers3=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.downlayers4=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.downlayers5=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.downlayerd1=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.downlayerd2=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.downlayerd3=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.downlayerd4=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.downlayerd5=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.fcs = nn.Sequential(
           nn.Linear(channel, 1),
       )
       self.fcd = nn.Sequential(
           nn.Linear(channel, 1),
       )
       self.fc = nn.Sequential(
           nn.Linear(2, 1)
       )
   def forward(self, imgs,imgd):
       features = self.convs(imgs)
       mlayers1 = self.makelayers1(features)
       dlayers1 = self.downlayers1(features)
       mlayers2 = self.makelayers2(torch.cat((mlayers1,dlayers1),1))
       dlayers2 = self.downlayers2(torch.cat((mlayers1,dlayers1),1))
       mlayers3 = self.makelayers3(torch.cat((mlayers2, dlayers2), 1))
       dlayers3 = self.downlayers3(torch.cat((mlayers2, dlayers2), 1))
       mlayers4 = self.makelayers4(torch.cat((mlayers3, dlayers3), 1))
       dlayers4 = self.downlayers4(torch.cat((mlayers3, dlayers3), 1))
       mlayers5 = self.makelayers5(torch.cat((mlayers4,dlayers4),1))
       featured = self.convd(imgd)
       mlayerd1 = self.makelayerd1(featured)
       dlayerd1 = self.downlayerd1(featured)
       mlayerd2 = self.makelayerd2(torch.cat((mlayerd1, dlayerd1), 1))
       dlayerd2 = self.downlayerd2(torch.cat((mlayerd1, dlayerd1), 1))
       mlayerd3 = self.makelayerd3(torch.cat((mlayerd2, dlayerd2), 1))
       dlayerd3 = self.downlayerd3(torch.cat((mlayerd2, dlayerd2), 1))
       mlayerd4 = self.makelayerd4(torch.cat((mlayerd3, dlayerd3), 1))
       dlayerd4 = self.downlayerd4(torch.cat((mlayerd3, dlayerd3), 1))
       mlayerd5 = self.makelayerd5(torch.cat((mlayerd4, dlayerd4), 1))
       outputs = self.fcs(mlayers5.view(imgs.shape[0], -1))
       outputd = self.fcd(mlayerd5.view(imgs.shape[0], -1))
       output = self.fc(torch.cat((outputs,outputd),1))
       return output
def Import_light_file(fpath):
    global LightWaveAmplitudeListsRaw
    global LightWaveAmplitudeListsFilter
    global LightWaveAmplitudeLists
    global LightWaveLengthList
    global LightWaveTimeList
    global LightWaveTimeSecondList
    global lightfilecount
    global LightTimeToFix
    try:
        t = WavePackage.Wave(fpath)

        if lightfilecount == 0:
            LightTimeStart = t.TimeStart
            LightTimeToFix = t.IntWaveTimeList[0] % 100000000000 / 1000
            LightWaveLengthList = t.FloatWaveLengthList
            LightWaveAmplitudeListsRaw += t.FloatWaveAmplitudeLists

            LightWaveTimeList += t.IntWaveTimeList
            LightWaveTimeSecondList += t.FloatWaveTimeSecondList
            lightfilecount = lightfilecount + 1
            LightTimeEnd = t.TimeEnd
        else:
            LightWaveAmplitudeListsRaw += t.FloatWaveAmplitudeLists
            LightWaveTimeList += t.IntWaveTimeList
            temp = len(LightWaveTimeSecondList)
            LightWaveTimeSecondList += [i + LightWaveTimeSecondList[1] - LightWaveTimeSecondList[0] +
                                             LightWaveTimeSecondList[temp - 1]
                                             for i in t.FloatWaveTimeSecondList]
            lightfilecount = lightfilecount + 1
            LightTimeEnd = t.TimeEnd
        LightWaveAmplitudeLists = LightWaveAmplitudeListsRaw
        print('光谱数据读入成功')
        print('当前载入光谱文件数量: ' + str(lightfilecount))
    except:
        print('光谱数据读入失败')
    pass
def LightWavefilter():
    global LightWaveAmplitudeListsFilter
    global LightWaveAmplitudeLists
    LightWaveAmplitudeListsFilter=LightWaveAmplitudeLists.copy()
    templist=[0 for i in range(len(LightWaveAmplitudeLists))]
    for i in range(len(LightWaveAmplitudeLists[0])):
        for iii in range(len(LightWaveAmplitudeLists)):
            templist[iii]=LightWaveAmplitudeLists[iii][i]
        #templist=savgol_filter(templist, 5, 1)
        for iii in range(len(LightWaveAmplitudeLists)):
               LightWaveAmplitudeListsFilter[iii][i]=templist[iii]
    LightWaveAmplitudeLists=LightWaveAmplitudeListsFilter.copy()

def Import_press_file(fpath):
    global PressTimeList
    global PressAmplitudeList
    global NewPressAmplitudeList
    global PressAmplitudelogList
    global PressTimeToFix
    try:
        t = PressPackage.Press(fpath)
        PressTimeStart = t.TimeStart
        PressTimeToFix = t.IntPressTimeList[0] % 100000000000 / 1000
        PressTimeList = t.IntPressTimeSecondList

        PressAmplitudeList = t.IntPressDataList
        PressTimeEnd = t.TimeEnd
        PressMove = 0.013
        temp = 0
        pmin = min(PressAmplitudeList)
        PressAmplitudeList=[i-pmin for i in PressAmplitudeList]
        for i in range(len(PressAmplitudeList)):
            temp = temp + PressMove
            if i > 5 and PressAmplitudeList[i] - PressAmplitudeList[i - 5] > 10:
                temp = temp - PressMove
            if i > 5 and PressAmplitudeList[i] - PressAmplitudeList[i - 5] < -10:
                temp = 0
            NewPressAmplitudeList.append(max(PressAmplitudeList[i] - temp, 2))
            #NewPressAmplitudeList.append(min(PressAmplitudeList[i], PressAmplitudeList[len(PressAmplitudeList)-1]*1.01))
        PressAmplitudelogList = [math.log(max(2, i), 10) for i in NewPressAmplitudeList]
        print('气压数据读入成功')
    except:
        print('气压数据读入失败')

def fftfuc():
    for i in range(len(LightWaveAmplitudeLists)):
        ffty = fft(LightWaveAmplitudeLists[i])
        fftyabs = [abs(i) for i in ffty]
        fftyangle = [i.imag for i in ffty]
        fftabslistlist.append(fftyabs[0:5])
        fftanglelist.append(fftyangle[0:5])
    pass
def SGDfuc():
    for i in range(len(LightWaveAmplitudeLists)):
        LightWaveAmplitudeLists[i]=savgol_filter(LightWaveAmplitudeLists[i],len(LightWaveAmplitudeLists[i])//2,5)
    pass
def GASFFunc():
    global GASFMatrixListRaw
    global GASFMatrixList
    for i in LightWaveAmplitudeLists:
        listtemp = []
        for ii in range(0, len(i), 15):
            if(LightWaveLengthList[ii]>600 and LightWaveLengthList[ii]<900):
                            listtemp.append(i[ii])
        temp=Gramme.GASFExchange(listtemp,flag=0)
        GASFMatrixListRaw.append(temp)
    GASFMatrixList=GASFMatrixListRaw.copy()
    print('GASF 成功')
def GADFFunc():
    global GADFMatrixListRaw
    global GADFMatrixList
    for i in LightWaveAmplitudeLists:
        listtemp = []
        for ii in range(0, len(i), 15):
            if(LightWaveLengthList[ii]>600 and LightWaveLengthList[ii]<900):
                            listtemp.append(i[ii])
        temp=Gramme.GADFExchange(listtemp,flag=0)
        GADFMatrixListRaw.append(temp)
    GADFMatrixList=GADFMatrixListRaw.copy()
    print('GADF 成功')


def Dataget(plog):
    global Allxlist
    global Allylist
    xlist=[]
    ylist=[]
    for lt in range(len(LightWaveTimeSecondList) - 100, len(LightWaveTimeSecondList)):
        templist = [[], []]
        templist[0].extend([GASFMatrixList[lt]])
        templist[1].extend([GADFMatrixList[lt]])
        xlist.append(templist.copy())
        ylist.append(plog)

    Allxlist.append(xlist.copy())
    Allylist.append(ylist.copy())
    print('dataget 成功')




def zero():
    global LightWaveAmplitudeLists
    templen=20
    tempexlist=[0 for i in range(len(LightWaveAmplitudeLists[0]))]
    for ii in range(len(LightWaveAmplitudeLists[0])):
        tempsum=0.0
        for iii in range(templen):
            tempsum=LightWaveAmplitudeLists[iii+10][ii]+tempsum
        tempexlist[ii]=tempsum/templen
    for i in range(0,len(LightWaveAmplitudeLists)):
        for ii in range(len(LightWaveAmplitudeLists[0])):
            LightWaveAmplitudeLists[i][ii] = LightWaveAmplitudeLists[i][ii] - tempexlist[ii]
def next():
    global LightWaveAmplitudeListsRaw
    global LightWaveAmplitudeListsFilter
    global LightWaveAmplitudeLists
    global LightWaveLengthList
    global LightWaveTimeList
    global LightWaveTimeSecondList
    global lightfilecount
    global PressTimeList
    global PressAmplitudeList
    global NewPressAmplitudeList
    global PressAmplitudelogList
    global GASFMatrixListRaw
    global GADFMatrixListRaw
    global GASFMatrixList
    global GADFMatrixList
    global AllGASFMatrixList
    global AllGADFMatrixList
    global AllPressTimeToFix
    global AllLightTimeToFix
    global AllPressTimeList
    global AllPressAmplitudelogList
    global AllLightWaveTimeSecondList
    global fftabslistlist
    global fftanglelist
    AllPressTimeToFix.append(PressTimeToFix)
    AllLightTimeToFix.append(LightTimeToFix)
    AllPressTimeList.append(PressTimeList.copy())
    AllPressAmplitudelogList.append(PressAmplitudelogList.copy())
    AllLightWaveTimeSecondList.append(LightWaveTimeSecondList)
    AllGASFMatrixList.append(GASFMatrixList.copy())
    AllGADFMatrixList.append(GADFMatrixList.copy())
    LightWaveAmplitudeListsRaw = []
    LightWaveAmplitudeListsFilter = []
    LightWaveAmplitudeLists = []
    LightWaveLengthList = []
    LightWaveTimeList = []
    LightWaveTimeSecondList = []
    PressTimeList = []
    PressAmplitudeList = []
    NewPressAmplitudeList = []
    PressAmplitudelogList = []
    fftabslistlist = []
    fftanglelist = []
    GASFMatrixListRaw = []
    GADFMatrixListRaw = []
    GASFMatrixList = []
    GADFMatrixList = []
    lightfilecount = 0
    print('清除缓存')
def loaddata(lpaths,p):
    for i in range(len(lpaths)):
        Import_light_file('mydata/'+lpaths[i]+'.txt')
    plog=math.log10(p)
    zero()
    fftfuc()
    SGDfuc()
    GADFFunc()
    GASFFunc()
    Dataget(plog)
    next()
loaddata(['22-pd20'],22)#1
loaddata(['25-pd20'],25)#1
loaddata(['28-pd20'],28)#1
loaddata(['31-pd20'],31)#1
loaddata(['34-pd20'],34)#1
loaddata(['37-pd20'],37)#1
loaddata(['39-pd20'],39)#1
loaddata(['40-pd20'],40)#1
loaddata(['41-pd20'],41)#1
loaddata(['43-pd20'],43)#1
loaddata(['47-pd20'],47)#1
loaddata(['50-pd20'],50)#1
loaddata(['55-pd20'],55)#1
loaddata(['60-pd20'],60)#1
loaddata(['65-pd20'],65)#1
loaddata(['70-pd20'],70)#1
loaddata(['75-pd20'],75)#1
loaddata(['80-pd20'],80)#1
loaddata(['85-pd20'],85)#1
loaddata(['90-pd20'],90)#1
loaddata(['95-pd20'],95)#1
loaddata(['100-pd20'],100)#1





xtrainlistraw=[]
ytrianlistraw=[]
for i in range(len(Allxlist)):
    xtrainlistraw.extend(Allxlist[i])
    ytrianlistraw.extend(Allylist[i])
cntlist=[i for i in range(len(xtrainlistraw))]
random.shuffle(cntlist)
for i in range(len(cntlist)):
    xtrainlist.append(xtrainlistraw[cntlist[i]])
    ytrianlist.append([ytrianlistraw[cntlist[i]]])
cnnmodel=MyNet()

print('开始训练')
loss = nn.MSELoss()
x = [i for i in range(len(xtrainlist))]
losslist=[]
acclist = []
lrcnt=0
for epoch in range(1, num_epochs + 1):
    random.shuffle(x)
    lsum = 0
    for i in range(int(len(x)//batch)) :
        nl1,nl2,np=torch.tensor([xtrainlist[x[i*batch+j]][0] for j in range(batch)]),torch.tensor([xtrainlist[x[i*batch+j]][1] for j in range(batch)]),\
                   torch.tensor([ytrianlist[x[i*batch+j]] for j in range(batch)]),
        r = lr / (10 + 1.2*epoch)
        optimizer = torch.optim.Adam(cnnmodel.parameters(), lr=r)
        y_hat= cnnmodel(nl1,nl2)
        l = loss(y_hat, np).mean()
        optimizer.zero_grad()
        l.backward()
        lsum = lsum + l.data.item()
        optimizer.step()
        if i % 100 == 0:
            print("epoch:{},loss:{}".format(epoch, l))
    losslist.append(lsum)
    acc = 0
    for i in range(len(Allxlist)):
        ypreloglist = [
            cnnmodel(torch.tensor([Allxlist[i][ii][0]]), torch.tensor([Allxlist[i][ii][1]]))[0][0].data.item() for
            ii in range(len(Allxlist[i]))]
        yprelist = [10 ** ii for ii in ypreloglist]
        ytruelist = [10 ** ii for ii in Allylist[i]]
        stdlist = [abs(ytruelist[i] - yprelist[i]) / ytruelist[i] for i in range(len(yprelist))]
        std = 0
        for ii in range(len(stdlist) // 2, len(stdlist)):
            std = std + stdlist[ii]
        acc = 1 - std / (len(stdlist) // 2) + acc
    acc = acc / len(Allxlist)
    acclist.append(acc)
print('训练完成')
net = cnnmodel.to(torch.device('cpu'))
torch.save(net, 'mydata/' + mod + 'net.pkl')
print(losslist)
print(acclist)
plt.plot(acclist)
plt.plot(losslist)
plt.show()
