class WindowedAdaptation():
    def __init__(self, warmup, initbuffer = 75, termbuffer = 100, windowsize = 25):
        self._windowsize = windowsize
        self._warmup = warmup
        self._firstwindow = initbuffer
        self._closewindow = self._firstwindow + self._windowsize
        self._lastwindow = self._warmup - termbuffer

    def calculate_next_window(self):
        self._windowsize *= 2
        nextclosewindow = self._closewindow + self._windowsize
        if self._closewindow + 2 * self._windowsize > self._lastwindow:
            self._closewindow = self._lastwindow
        else:
            if nextclosewindow <= self._lastwindow:
                self._closewindow = nextclosewindow
            else:
                self._closewindow = self._lastwindow

    def firstwindow(self):
        return self._firstwindow

    def lastwindow(self):
        return self._lastwindow

    def closewindow(self):
        return self._closewindow
