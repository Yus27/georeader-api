#-*- coding: utf-8 -*-

import numpy as np
import os
import app.utils

def importFromGeoScan(FileName, IsOldVersion = False):
    # try:
    import struct
    with open(FileName, "rb") as f:
        # Чтение заголовка файла
        label, IDVersion, MainNumber, SerNumber, ProfSerNumber, State, NTraces, NSamples, NTextLabels, Tall, Eps, Ddxmm, \
        StartPosition, StartX, StartY, StartZ = struct.unpack(
            "IIIIIIIIIIfiqqqq", f.read(struct.calcsize("IIIIIIIIIIfiqqqq")))

        if IsOldVersion:
            NSamples -= 2

        f.read(8 + 8)
        # CreateTime_low, CreateTime_high = struct.unpack("II", f.read(struct.calcsize("II")))
        # print(CreateTime_low, CreateTime_high)
        # ManipulationTime_low, ManipulationTime_high = struct.unpack("II", f.read(struct.calcsize("II")))
        # print(ManipulationTime_low, ManipulationTime_high)

        La, Tstart, Tspp, SppTreshold, Kraz, WinSize, HorWinSize, WhiteProcent, BlackProcent, ScanMode, NSum, NTpz = struct.unpack(
            "IiiffIIiiIII", f.read(struct.calcsize("IiiffIIiiIII")))

        F4_TEXT_STRING_SIZE = 40
        AntenName = f.read(F4_TEXT_STRING_SIZE).decode("windows-1251")
        Operator = f.read(F4_TEXT_STRING_SIZE).decode("windows-1251")
        Object_ = f.read(F4_TEXT_STRING_SIZE).decode("windows-1251")
        Tips1 = f.read(F4_TEXT_STRING_SIZE).decode("windows-1251")
        Tips2 = f.read(F4_TEXT_STRING_SIZE).decode("windows-1251")
        Tips3 = f.read(F4_TEXT_STRING_SIZE).decode("windows-1251")
        CreatingUserNumber, LastUserNumber, ZeroZone, ShiftProcent = struct.unpack("llii", f.read(struct.calcsize("llii")))

        GPRUnit = "ОКО-2"
        import re
        try:
            Frequency = float("".join(re.findall("\d", AntenName)))
        except ValueError:
            Frequency = 1000

        xState = State % 0x1000
        digit3 = xState // 0x100
        if digit3 >= 0x8:
            # print("Файл был сжат...")
            isCompress = True
        else:
            isCompress = False

        f.seek(512)

        if isCompress:
            compress_data = f.read()
            import bz2
            decompress_data = bz2.decompress(compress_data)
            f.close()

            AFilePath = os.path.dirname(FileName)
            AFileName = os.path.basename(FileName)
            NewFileName = AFilePath + "//___" + AFileName
            # print(NewFileName)
            f = open(NewFileName, "wb")
            f.write(decompress_data)
            f.close()
            f = open(NewFileName, "rb")

        # Чтение массива цветов
        ColorArray = []
        COLOR_ARRAY_LENGTH = 256
        for i in range(COLOR_ARRAY_LENGTH):
            c, = struct.unpack("L", f.read(struct.calcsize("L")))
            ColorArray.append(c)
        # print(ColorArray)

        # Чтение массива коэффициентов выравнивания
        MArray = np.zeros((NSamples))
        for i in range(NSamples):
            c, = struct.unpack("f", f.read(struct.calcsize("f")))
            MArray[i] = c
        # unPaint.drawPlot(np.arange(NSamples), MArray)

        # Чтение массива трасс
        Data = np.zeros((NTraces, NSamples))
        Info = []

        TimeCollecting = []
        Labels = []
        for i in range(NTraces):
            t = f.read(8)
            intervals = int.from_bytes(t, byteorder='little')

            TimeCollecting.append(intervals)

            Position, X, Y, Z, IAnt, LabelID, LabelPos = struct.unpack("iiiiiII", f.read(struct.calcsize("iiiiiII")))
            info = {"Position": Position, "X": X, "Y": Y, "Z": Z, "IAnt": IAnt, "LabelID": LabelID,
                    "LabelPos": LabelPos}
            if LabelID > 0:
                Labels.append(i)

            t = f.read(8)


            try:
                Data[i,:] = np.fromfile(f, count=NSamples, dtype="float32")
            except ValueError:
                if i == NTraces-1:
                    Data[i, :] = Data[i-1, :]
            # if i == 0:
            #     unPaint.drawPlot(np.arange(NSamples), Data[i,:])

            # for j in range(NSamples):
            #     A, = struct.unpack("f", f.read(struct.calcsize("f")))
            #     Data[i][j] = A

            Info.append(info)

        try:
            Data = Data.astype("float64")
        except MemoryError:
            return None
        # for i in range(NTraces):
        #     if TimeCollecting[i] == 0:
        #         if i == 0:
        #             TimeCollecting[i] = TimeCollecting[i+1]
        #         elif i == NTraces-1:
        #             TimeCollecting[i] = TimeCollecting[i-1]
        #         else:
        #             TimeCollecting[i] = (TimeCollecting[i+1] + TimeCollecting[i-1])/2


        TimeCollecting = np.array(TimeCollecting)
        Years = []
        for i,interval in enumerate(TimeCollecting):
            # print(interval)
            DateTime = app.utils.DateTransforms.getDateTime(interval)
            if DateTime is None:
                # print("Времена записи %d трассы некорректны!" % i)
                TimeCollecting[i] = 0
            else:
                Years.append(DateTime.year)
        Years = np.array(Years)
        if len(Years) == 0:
            TimeCollecting = None
        else:
            freq = np.bincount(Years)
            currYear = np.argmax(freq)
            for i,interval in enumerate(TimeCollecting):
                DateTime = app.utils.DateTransforms.getDateTime(interval)
                if (DateTime is not None) and (np.abs(DateTime.year - currYear) > 1):
                    # print("Времена записи %d трассы некорректны! %d" % (i, DateTime.year))
                    TimeCollecting[i] = 0

            if (len(TimeCollecting) <= 1) or (TimeCollecting[0] == 0) or (TimeCollecting[1] < TimeCollecting[0]) or (TimeCollecting[-1] == 0) or (TimeCollecting[-1] < TimeCollecting[-2]) :
                TimeCollecting = None
            else:
                for i in range(1, len(TimeCollecting)-1):
                    if TimeCollecting[i] != 0:
                        i1 = i-1
                        i2 = i+1
                        while (TimeCollecting[i1]==0):
                            i1 -= 1
                        while (TimeCollecting[i2]==0):
                            i2 += 1
                        if not (TimeCollecting[i1] <= TimeCollecting[i] <= TimeCollecting[i2]):
                            TimeCollecting[i] = 0

                from scipy.interpolate import interp1d
                for i in range(1, len(TimeCollecting) - 1):
                    if TimeCollecting[i] == 0:
                        i_last = i-1
                        i_next = i+1
                        while TimeCollecting[i_next] == 0:
                            i_next += 1
                        t_last = TimeCollecting[i_last]
                        t_next = TimeCollecting[i_next]
                        func = interp1d([i_last, i_next], [t_last, t_next], kind='linear', fill_value='extrapolate')
                        TimeCollecting[i_last:i_next+1] = func(range(i_last, i_next+1))

            if TimeCollecting is not None:
                TimeCollecting = TimeCollecting.astype("int64")


        # Чтение массива текстов меток
        LabelTexts = []
        for i in range(NTextLabels):
            LabelID, LabelClr = struct.unpack("IL", f.read(struct.calcsize("IL")))
            try:
                Text = f.read(F4_TEXT_STRING_SIZE).decode("windows-1251")
            except UnicodeDecodeError:
                Text = "ERROR"
            LabelTexts.append({"LabelID": LabelID, "LabelClr": LabelClr, "Text": Text})

        f.close()
        if isCompress:
            try:
                os.remove(NewFileName)
            except:
                pass

        Stage = Ddxmm/1000
        if Stage <= 0: return None
        TimeBase = Tall
        if TimeBase <=0: return None
        AntDist = La/1000
        from app.gpr_calculations import EpsToVelocity
        DefaultV = EpsToVelocity(Eps)
        if DefaultV > 0.3: DefaultV = 0.3
        StartPosition = StartPosition/1000
        GainCoefs = MArray/np.max(MArray)

        if np.all(Data == 0.0): return None
        if (Data.shape[0] < 2) or (Data.shape[1] < 2): return None
        return (Data, Stage, TimeBase, AntDist, DefaultV, StartPosition, Tspp, Kraz, WinSize, HorWinSize,
                GainCoefs, TimeCollecting, GPRUnit, AntenName, Frequency, Labels)
    # except:
    #     return None


def importFromSGY(FileName, endian='little'):
    """Импорт радарограммы из SGY"""
    try:
        import segyio
        with segyio.open(FileName, mode="r", endian=endian, ignore_geometry=True, strict=False) as f:
            Data = segyio.tools.collect(f.trace[:])
            Data=Data.astype(float)
            return Data
    except:
        return None


def importFromDZT(FileName):
    try:
        import readgssi
        header, Data = readgssi.readgssi(FileName)
        GPRUnit = "GSSI"
        AntenName = header['rh_antname']
        try:
            Frequency = readgssi.ANT[AntenName]
        except KeyError:
            Frequency = None
        TracesPerMeter = header['rhf_spm']
        if TracesPerMeter == 0:
            TracesPerMeter = 1.0
        Stage = 1.0 / TracesPerMeter
        eps = header['rhf_epsr']
        if eps!=0:
            DefaultV = 0.3 / np.sqrt(eps)
        else:
            DefaultV = 0.1
        TimeBase = header['rhf_range']
        Data = np.transpose(Data)
        Data = Data.astype("float64")
        if (Data.shape[0] < 2) or (Data.shape[1] < 2): return None
        return (Data, Stage, TimeBase, DefaultV, GPRUnit, AntenName, Frequency)
    except:
        return None


def importFromRD3(FileNameRD3, FileNameRad, DefaultStage):
    try:
        import pandas as pd
        with open(FileNameRad) as fRad:
            df = pd.read_csv(fRad, sep=':', encoding='utf-8',  index_col=0, names=["values",])
            SamplesCount = int(df["values"]["SAMPLES"])
            TracesCount = int(df["values"]["LAST TRACE"])
            Stage = float(df["values"]["DISTANCE INTERVAL"])
            if Stage == 0:
                Stage = DefaultStage
            TimeBase = float(df["values"]["TIMEWINDOW"])
            AntDist = float(df["values"]["ANTENNA SEPARATION"])
            GPRUnit = "MALA"
            AntenName = df["values"]["ANTENNAS"]
            import re
            S = re.findall("\d+", AntenName)
            if S[0] == "":
                Frequency = None
            else:
                Frequency = float(S[0])

        with open(FileNameRD3, 'rb') as fRD3:
            Data = np.fromfile(fRD3, count=TracesCount * SamplesCount, dtype="int16")
            try:
                Data = np.reshape(Data, (TracesCount, SamplesCount))
            except ValueError:
                TracesCount = TracesCount - 1
                Data = np.reshape(Data, (TracesCount, SamplesCount))
            Data = Data.astype("float64")
            if (Data.shape[0] < 2) or (Data.shape[1] < 2): return None
            return (Data, Stage, TimeBase, AntDist, GPRUnit, AntenName, Frequency)
    except:
        return None


def importFromDT1(FileNameDT1, FileNameHD, DateFormat):
    try:
        import pandas as pd
        info = {}
        with open(FileNameHD) as fHD:
            i = 0
            for line in fHD.readlines():
                line = line.strip()
                if line != "":
                    i += 1
                    if i == 3:
                        Date = None
                        # Date = line
                        # try:
                        #     Date = utils.DateTransforms.parserDate(Date, DateFormat)
                        # except ValueError:
                        #     import logging
                        #     logging.error("Формат даты некорректен")
                        #     print("Формат даты некорректен")
                        #     Date = None
                    else:
                        pair = line.split("=")
                        if len(pair) == 2:
                            left = pair[0].strip()
                            right = pair[1].strip()
                            info[left] = right
            TracesCount = int(info["NUMBER OF TRACES"])
            SamplesCount = int(info["NUMBER OF PTS/TRC"])
            Stage = float(info["STEP SIZE USED"])
            TimeBase = float(info["TOTAL TIME WINDOW"])
            AntDist = float(info["ANTENNA SEPARATION"])
            GPRUnit = "Sensors & Software"
            AntenName = ""
            Frequency = float(info["NOMINAL FREQUENCY"])
            TimeZeroSample = float(info["TIMEZERO AT POINT"])

            if Date is not None:
                diff = app.utils.DateTransforms.dateTimeToInterval(Date)

            with open(FileNameDT1, 'rb') as fDT1:
                import struct
                Data = np.zeros((TracesCount, SamplesCount))
                Data = Data.astype("int16")
                if Date is not None:
                    TimeCollecting = np.zeros((TracesCount,))
                else:
                    TimeCollecting = None
                for i in range(TracesCount):
                    trace_number, position, number_of_points_per_trace, topodata, _, _, _, stacks, time_window, _, _, _, _, _,\
                    receiver_x, receiver_y, receiver_z, transmitter_x, transmitter_y, transmitter_z, time_zero, zero_flag,\
                    _, time_of_day, comment_flag = struct.unpack("f"*25, fDT1.read(struct.calcsize("f"*25)))
                    comment = fDT1.read(7*4)
                    # print(trace_number, position, number_of_points_per_trace, topodata, _, _, _, stacks, time_window, _, _, _, _, _,\
                    # receiver_x, receiver_y, receiver_z, transmitter_x, transmitter_y, transmitter_z, time_zero, zero_flag,\
                    # _, time_of_day, comment_flag, comment)
                    if int(number_of_points_per_trace) != SamplesCount:
                        return None

                    Data[i, :] = np.fromfile(fDT1, count=SamplesCount, dtype="int16")
                    if Date is not None:
                        TimeCollecting[i] = int(diff + time_of_day*10000000)
                if Date is not None:
                    TimeCollecting = TimeCollecting.astype("int64")

                Data = Data.astype("float64")
                if (Data.shape[0] < 2) or (Data.shape[1] < 2): return None
                return (Data, Stage, TimeBase, AntDist, GPRUnit, AntenName, Frequency, TimeZeroSample, TimeCollecting)
    except:
        import traceback
        traceback.print_exc()
        return None


def importFromCSV(FileName, sep = ";", IsTranspose = False):
    """Импорт радарограммы из CSV"""
    try:
        Data = []
        with open(FileName) as f1:
            for i, line in enumerate(f1):
                if i>=3:
                    dat = line.split(sep)
                    del dat[0]
                    dat = [float(x) for x in dat]
                    Data.append(dat)
        Data = np.array(Data)
        if IsTranspose:
            Data = np.transpose(Data)
        if (Data.shape[0] < 2) or (Data.shape[1] < 2): return None
        return Data
    except:
        return None


def importFromASCII(FileName):
    """Импорт радарограммы из ASCII"""
    try:
        import pandas as pd
        with open(FileName, 'r') as f:
            # df = pd.read_csv(f, delim_whitespace=True, names = ["l", "t", "A"])
            df = pd.read_csv(f, sep=',', names = ["l", "t", "A"])


        Times = df[df["l"] == 0]["t"].values
        dT = np.mean(Times[1:]-Times[:-1])
        samplesCount = len(Times)
        TimeBase = np.max(Times)

        L = df["l"].unique()
        dL = np.mean(L[1:]-L[:-1])
        tracesCount = len(L)

        A = df["A"].values
        Data = A.reshape((tracesCount, samplesCount))
        if (Data.shape[0] < 2) or (Data.shape[1] < 2): return None

        return (Data, dL, TimeBase)

    except:
        return None


def importFromTXT(FileName):
    """Импорт радарограммы из TXT (формат Крот)"""
    try:
        with open(FileName, 'r') as f:
            import pandas as pd
            df = pd.read_csv(f, sep=';', names=["trace","sample","amplitude"], skiprows=[0], index_col=False)
            try:
                df["sample"] = df["sample"].astype(float)
            except ValueError:
                df["sample"] = [x.replace(',', '.') for x in df["sample"]]
                df["sample"] = df["sample"].astype(float)

            # try:
            dt = df["sample"].values[1] - df["sample"].values[0]
            # except TypeError:
            #     df["sample"] = [x.replace(',', '.') for x in df["sample"]]
            #     dt = df["sample"].values[1] - df["sample"].values[0]
            df["sample"] = df["sample"]/dt
            df["trace"] = df["trace"].astype(int)
            df["sample"] = df["sample"].astype(int)
            df["amplitude"] = df["amplitude"].astype(float)
            maxX = df["trace"].max()
            maxT = df["sample"].max()

            Data = np.zeros((maxX + 1, maxT + 1))
            Data[df["trace"].values, df["sample"].values] = df["amplitude"]
            Data = Data[~np.all(Data == 0, axis=1)]
            if (Data.shape[0] < 2) or (Data.shape[1] < 2): return None
            return Data
    except:
        return None


def importFromTXTWithCoords(FileName):
    """Импорт радарограммы из TXT (формат Крот)"""
    try:
        with open(FileName, 'r') as f:
            import pandas as pd
            df = pd.read_csv(f, sep=';', names=["X", "Y", "Z", "T", "amplitude", "TimeCollecting"], skiprows=[0], index_col=False, decimal=",")
            df["X"] = df["X"].astype(float)
            df["Y"] = df["Y"].astype(float)
            df["Z"] = df["Z"].astype(float)
            df["T"] = df["T"].astype(float)
            df["amplitude"] = df["amplitude"].astype(float)

            dt = df["T"].values[1] - df["T"].values[0]
            df["sample"] = df["T"]/dt
            df["sample"] = df["sample"].astype(int)

            maxSample = df["sample"].max()
            samplesCount = int(maxSample+1)
            tracesCount = int(len(df.index)//samplesCount)

            tracesSeries = pd.Series(np.arange(tracesCount))
            tracesForEachSampleSeries = tracesSeries.repeat(samplesCount)
            tracesForEachSampleSeries.index = df.index
            df["trace"] = tracesForEachSampleSeries

            Data = np.zeros((tracesCount, samplesCount))
            Data[df["trace"].values, df["sample"].values] = df["amplitude"]
            if (Data.shape[0] < 2) or (Data.shape[1] < 2):
                return None
            ZData = np.zeros((tracesCount, samplesCount))
            ZData[df["trace"].values, df["sample"].values] = df["Z"]

            firstSampleFromEachTrace = df[df["sample"]==0].copy()
            import datetime
            DateTimeParserFormat = "%H:%M:%S,%f"
            firstSampleFromEachTrace["TimeCollecting"] = firstSampleFromEachTrace["TimeCollecting"].apply(datetime.datetime.strptime, args=(DateTimeParserFormat, ))
            import utils
            firstSampleFromEachTrace["TimeCollecting"] = firstSampleFromEachTrace["TimeCollecting"].apply(utils.DateTransforms.dateTimeToInterval)
            X = firstSampleFromEachTrace["X"].values
            Y = firstSampleFromEachTrace["Y"].values
            Z = firstSampleFromEachTrace["Z"].values
            CoordTraces = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
            TimeCollecting = firstSampleFromEachTrace["TimeCollecting"].values
            Trajectory = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'DateTime': TimeCollecting})

            return Data, ZData, CoordTraces, Trajectory, TimeCollecting, dt
    except:
        return None


def getListOfRewriteRDR(ImportFileName):
    import os
    RewriteFileNames = []
    BaseFileName = os.path.splitext(ImportFileName)[0]
    FileName = BaseFileName + ".rdr"
    if os.path.exists(FileName):
        RewriteFileNames.append(os.path.basename(FileName))
    for i in range(100):
        FileName = "{:s}_{:0>3d}.rdr".format(BaseFileName, i)
        if os.path.exists(FileName):
            RewriteFileNames.append(os.path.basename(FileName))
    return RewriteFileNames




