import numpy as np
from numba import jit


class Geometry:
    @staticmethod
    def offsetSegment(x1, y1, x2, y2, offset):
        l = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        x1p = x1 + offset * (y2 - y1) / l
        x2p = x2 + offset * (y2 - y1) / l
        y1p = y1 + offset * (x1 - x2) / l
        y2p = y2 + offset * (x1 - x2) / l
        return x1p, y1p, x2p, y2p

    @staticmethod
    def DistPoint(p1, p2):
        """Расстояние между двумя точками"""
        return np.hypot(p1[0]-p2[0], p1[1]-p2[1])

    @staticmethod
    def point_to_line_dist(point, line):
        """Calculate the distance between a point and a line segment.

        To calculate the closest distance to a line segment, we first need to check
        if the point projects onto the line segment.  If it does, then we calculate
        the orthogonal distance from the point to the line.
        If the point does not project to the line segment, we calculate the
        distance to both endpoints and take the shortest distance.

        :param point: Numpy array of form [x,y], describing the point.
        :type point: numpy.core.multiarray.ndarray
        :param line: list of endpoint arrays of form [P1, P2]
        :type line: list of numpy.core.multiarray.ndarray
        :return: The minimum distance to a point.
        :rtype: float
        """
        # unit vector
        unit_line = line[1] - line[0]
        norm_unit_line = unit_line / np.linalg.norm(unit_line)

        # compute the perpendicular distance to the theoretical infinite line
        segment_dist = (
            np.linalg.norm(np.cross(line[1] - line[0], line[0] - point)) /
            np.linalg.norm(unit_line)
        )

        diff = (
            (norm_unit_line[0] * (point[0] - line[0][0])) +
            (norm_unit_line[1] * (point[1] - line[0][1]))
        )

        x_seg = (norm_unit_line[0] * diff) + line[0][0]
        y_seg = (norm_unit_line[1] * diff) + line[0][1]

        endpoint_dist = min(
            np.linalg.norm(line[0] - point),
            np.linalg.norm(line[1] - point)
        )

        # decide if the intersection point falls on the line segment
        lp1_x = line[0][0]  # line point 1 x
        lp1_y = line[0][1]  # line point 1 y
        lp2_x = line[1][0]  # line point 2 x
        lp2_y = line[1][1]  # line point 2 y
        is_betw_x = lp1_x <= x_seg <= lp2_x or lp2_x <= x_seg <= lp1_x
        is_betw_y = lp1_y <= y_seg <= lp2_y or lp2_y <= y_seg <= lp1_y
        if is_betw_x and is_betw_y:
            return segment_dist
        else:
            # if not, then return the minimum distance to the segment endpoints
            return endpoint_dist

    @staticmethod
    def intersectLines(pt1, pt2, ptA, ptB):
        """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

            returns a tuple: (xi, yi, valid, r, s), where
            (xi, yi) is the intersection
            r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
            s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
                valid == 0 if there are 0 or inf. intersections (invalid)
                valid == 1 if it has a unique intersection ON the segment    """

        import math
        DET_TOLERANCE = 0.00000001

        # the first line is pt1 + r*(pt2-pt1)
        # in component form:
        x1, y1 = pt1;
        x2, y2 = pt2
        dx1 = x2 - x1;
        dy1 = y2 - y1

        # the second line is ptA + s*(ptB-ptA)
        x, y = ptA;
        xB, yB = ptB;
        dx = xB - x;
        dy = yB - y;

        # we need to find the (typically unique) values of r and s
        # that will satisfy
        #
        # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
        #
        # which is the same as
        #
        #    [ dx1  -dx ][ r ] = [ x-x1 ]
        #    [ dy1  -dy ][ s ] = [ y-y1 ]
        #
        # whose solution is
        #
        #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
        #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
        #
        # where DET = (-dx1 * dy + dy1 * dx)
        #
        # if DET is too small, they're parallel
        #
        DET = (-dx1 * dy + dy1 * dx)

        if math.fabs(DET) < DET_TOLERANCE: return None #(0, 0, 0, 0, 0)

        # now, the determinant should be OK
        DETinv = 1.0 / DET

        # find the scalar amount along the "self" segment
        r = DETinv * (-dy * (x - x1) + dx * (y - y1))

        # find the scalar amount along the input line
        s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

        # return the average of the two descriptions
        xi = (x1 + r * dx1 + x + s * dx) / 2.0
        yi = (y1 + r * dy1 + y + s * dy) / 2.0

        # x1, y1 = pt1;
        # x2, y2 = pt2
        #
        # x, y = ptA;
        # xB, yB = ptB;

        if ((math.fabs(x1-x2) < DET_TOLERANCE) or (x1 <= xi <= x2) or (x2 <= xi <= x1)) \
                and ((math.fabs(y1-y2) < DET_TOLERANCE) or (y1 <= yi <= y2) or (y2 <= yi <= y1)) \
                and ((math.fabs(x-xB) < DET_TOLERANCE) or (x <= xi <= xB) or (xB <= xi <= x)) \
                and ((math.fabs(y-yB) < DET_TOLERANCE) or (y <= yi <= yB) or (yB <= yi <= y)):
            return (xi, yi)
        else:
            return None
        # return (xi, yi, 1, r, s)


class Picketazh:
    @staticmethod
    def transformDistToPicket(L):
        "Преобразование расстояния в пикетаж"
        PK = int(np.trunc(L / 100))
        Plus = L - 100 * PK
        return PK, Plus


    @staticmethod
    def transformPicketToDist(PK, Plus):
        "Преобразование пикетажа в расстояние"
        if PK < 0:
            return
        if Plus < 0:
            return
        if Plus >= 100:
            return
        return 100 * PK + Plus


    @staticmethod
    def stringToPicket(S):
        Elements = S.split("+")
        if len(Elements) != 2:
            return
        try:
            PK = int(Elements[0])
        except:
            return None
        try:
            Plus = float(Elements[1])
        except:
            return None
        return 100 * PK + Plus


class Calculations:
    @staticmethod
    def Interpolate(x1, y1, x2, y2, x):
        """Линейная интерполяция"""
        try:
            return (x - x1) * (y2 - y1) / (x2 - x1) + y1
        except ZeroDivisionError:
            return None

    @staticmethod
    def calcPrecision(X):
        """Расчет требуемого количества знаков после запятой"""
        dX = np.abs(X[1:] - X[:-1])
        dx = np.min(dX)
        if dx <= 1e-6:
            import decimal
            S = "{value:.{prec}g}".format(value=X[0], prec=6)
            d = decimal.Decimal(S)
            digits = np.abs(d.as_tuple().exponent)
            return min(digits, 6)
        k = 0
        while np.round(dx) == 0:
            k += 1
            dx *= 10
        return min(k, 6)


    @staticmethod
    def savitzky_golay(y, window_size, order, deriv=0, rate=1):
        r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters
        ----------
        y : array_like, shape (N,)
            the values of the time history of the signal.
        window_size : int
            the length of the window. Must be an odd integer number.
        order : int
            the order of the polynomial used in the filtering.
            Must be less then `window_size` - 1.
        deriv: int
            the order of the derivative to compute (default = 0 means only smoothing)
        Returns
        -------
        ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.
        Examples
        --------
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
           Data by Simplified Least Squares Procedures. Analytical
           Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
           W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
           Cambridge University Press ISBN-13: 9780521880688
        """
        import numpy as np
        from math import factorial

        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order + 1)
        half_window = (window_size - 1) // 2
        # precompute coefficients
        b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
        m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode='valid')


    @staticmethod
    @jit(nopython=True)
    def cross(x, y, lc):
        # x = np.array(x)
        # y = np.array(y)
        lx = len(x)
        x = np.concatenate((x, np.zeros(lc)))
        cc = np.zeros(lc)
        for i in range(lc):
            cc[i] = np.sum(x[i:i + lx] * y)
        return cc


    @staticmethod
    @jit(nopython=True)
    def ricker(f, dt):
        # Ricker wavelet of central frequency f
        #  IN   f : central freq. in Hz (f <<1/(2dt) )
        #       dt: sampling interval in sec
        #  OUT  w:  the Ricker wavelet
        #       tw: time axis

        nw = 2.2 / f / dt
        nw = int(2 * np.floor(nw/2) + 1)
        nc = int(np.floor(nw/2))
        k = np.arange(nw)
        alpha = (nc-k+1)*f*dt*np.pi
        beta = alpha**2
        w = (1 - beta*2) * np.exp(-beta)
        tw = -(nc + 1 - np.arange(nw)) * dt
        return w #, tw


    @staticmethod
    # @jit(nopython=True)
    def blackharrispulse(fr, t):
        # This function computes the derivative of a Blackman-Harris window given a time vector
        # and the desired dominant frequency.  See Chen et al. (1997), Geophysics 62, p. 1733 for
        # details.  Note that their formulation has been changed here to T = 1.14/fr, such that fr
        # represents the approximate dominant frequency of the resulting pulse.
        # Syntax:  p = blackharrispulse(fr,t)
        # where fr = dominant frequency (Hz)
        #       t  = time vector (s)
        a = np.array([0.35322222, -0.488, 0.145, -0.010222222])
        T = 1.14 / fr
        window = np.zeros(t.shape)
        for n in range(4):
            window = window + a[n] * np.cos(2*n*np.pi*t/T)
        window[t>=T] = 0
        p = np.diff(window)
        p = p/np.max(np.abs(p))
        return p


    @staticmethod
    # @jit(nopython=True)
    def convmtx(v, n):
        """Generates a convolution matrix
        Usage: X = convm(v,n)
        Given a vector v of length N, an N+n-1 by n convolution matrix is
        generated of the following form:
                  |  v(0)  0      0     ...      0    |
                  |  v(1) v(0)    0     ...      0    |
                  |  v(2) v(1)   v(0)   ...      0    |
             X =  |   .    .      .              .    |
                  |   .    .      .              .    |
                  |   .    .      .              .    |
                  |  v(N) v(N-1) v(N-2) ...  v(N-n+1) |
                  |   0   v(N)   v(N-1) ...  v(N-n+2) |
                  |   .    .      .              .    |
                  |   .    .      .              .    |
                  |   0    0      0     ...    v(N)   |
        And then it's trasposed to fit the MATLAB return value.
        That is, v is assumed to be causal, and zero-valued after N.
        """
        N = len(v) + 2 * n - 2
        xpad = np.concatenate([np.zeros(n - 1), v[:], np.zeros(n - 1)])
        X = np.zeros((len(v) + n - 1, n))
        # Construct X column by column
        for i in range(n):
            X[:, i] = xpad[n - i - 1:N - i]
        return X.transpose()


   
class DateTransforms:
    @staticmethod
    def dateTimeToInterval(dateTime):
        import datetime
        null_date = datetime.datetime(1601, 1, 1, 0, 0, 0)
        dateTime = dateTime.to_pydatetime()
        diff = dateTime - null_date
        return diff.total_seconds() * 10000000

    @staticmethod
    def getDateTime(interval, isGPSHoursDelta=False, gps_hours_delta=0):
        if (interval is None) or (interval == 0):
            return None
        try:
            import pandas as pd
            import datetime
            FILETIME_null_date = datetime.datetime(1601, 1, 1, 0, 0, 0)

            if isGPSHoursDelta:
                time = FILETIME_null_date + datetime.timedelta(microseconds=interval / 10) + datetime.timedelta(
                    hours=gps_hours_delta)
            else:
                time = FILETIME_null_date + datetime.timedelta(microseconds=interval / 10)
            date = pd.to_datetime(time)
            return date
        except (ValueError, OverflowError):
            return None

    # @staticmethod
    # def parserDate(DateTime, DateTimeParserFormat):
    #     import pandas as pd
    #     # datetime = pd.Timestamp.strptime(DateTime, DateTimeParserFormat)
    #     import datetime
    #     Datetime = datetime.datetime.strptime(DateTime, DateTimeParserFormat)
    #     # Datetime = pd.Timestamp.strptime(DateTime, DateTimeParserFormat)
    #     # datetime = pd.to_datetime(DateTime, DateTimeParserFormat)
    #
    #     if (Datetime.year == 1900) and (Global.Project.Rad.TimeCollecting is not None):
    #         new_date = DateTransforms.getDateTime(Global.Project.Rad.TimeCollecting[0])
    #         datetime = pd.Timestamp(year=new_date.year, month=new_date.month, day=new_date.day,
    #                                 hour=Datetime.hour, minute=Datetime.minute, second=Datetime.second,
    #                                 microsecond=Datetime.microsecond)
    #     return Datetime



def in_directory(file, directory):
    #make both absolute
    import os.path
    directory = os.path.join(os.path.realpath(directory), '')
    file = os.path.realpath(file)

    #return true, if the common prefix of both is equal to directory
    #e.g. /a/b/c/d.rst and directory is /a/b, the common prefix is /a/b
    return os.path.commonprefix([file, directory]) == directory