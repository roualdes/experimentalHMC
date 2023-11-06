class WindowedAdaptation():
    def __init__(self, warmup, initbuffer = 75, termbuffer = 100, windowsize = 25):
        self.windowsize = windowsize
        self.warmup = warmup
        self.firstwindow = initbuffer
        self.closewindow = self.firstwindow + self.windowsize
        self.lastwindow = self.warmup - termbuffer

    def calculate_next_window(self):
        self.windowsize *= 2
        nextclosewindow = self.closewindow + self.windowsize
        if self.closewindow + 2 * self.windowsize > self.lastwindow:
            self.closewindow = self.lastwindow
        else:
            if nextclosewindow <= self.lastwindow:
                self.closewindow = nextclosewindow
            else:
                self.closewindow = self.lastwindow

    def firstwindow(self):
        return self.firstwindow

    def lastwindow(self):
        return self.lastwindow

    def closewindow(self):
        return self.closewindow
