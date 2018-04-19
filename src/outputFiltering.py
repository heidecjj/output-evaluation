from pandas import DataFrame as _DataFrame
from pandas import concat as _concat

def LowPassFilter(df, dt, windowSize=1):
    windowSize = int(windowSize/dt)
    return df.rolling(windowSize, center=True).max()


def PacketDetect(df, time, thresh=0, windowSize=1):
    dt = time[1] - time[0]

    df = LowPassFilter(df, dt, windowSize=windowSize)    

    # Filter out non-peaks
    df[(df < thresh)] = 0
    df[(df != 0)] = 1
    
    df = df.diff()
    
    df = _DataFrame(df)
    df.insert(0, 'time', time)
    
    return df

def PacketAnalysis(df):
    # Only keep non-zero entries
    df = df.dropna()
    df = df[(df.iloc[:,1:].T != 0).any()]

    dfValue = df.iloc[:,1]

    dfStart = df[dfValue > 0]['time']
    dfEnd = df[dfValue < 0]['time'].reset_index(drop=True)

    startList = []
    endList = []

    for burstStart in dfStart:
        laterEnds = dfEnd[(dfEnd > burstStart)]

        if laterEnds.shape[0] > 0:
            burstEnd = laterEnds.sort_values().iloc[0]
        else:
            burstEnd = None

        startList.append(burstStart)
        endList.append(burstEnd)

    return _DataFrame({'start':startList, 'end':endList}, 
        columns=['start', 'end'])