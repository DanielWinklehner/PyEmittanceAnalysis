import gi
gi.require_version('Gtk', '3.0')  # nopep8
from gi.repository import Gtk, GObject
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon
from scipy.interpolate import griddata
import csv
from matplotlib import colors as mplcol
from matplotlib import cm
from matplotlib import rc
from matplotlib import rc_context

rc('xtick.major', pad=10)
from matplotlib import rcdefaults
from matplotlib.colors import LinearSegmentedColormap
# from dans_pymodules import *
from mpl_canvas_wrapper import MPLCanvasWrapper
from filedialog_gtk import FileDialog
from particles import IonSpecies
import os
import sys

saveExe = False

# --- Constants -------------------------------------------------------------- #
pi = 3.14159265358979323  # Pi
amu = 1.66053873e-27  # Atomic Mass Unit (kg)
clight = 2.99792458e+8  # Speed of light in vacuum (m/s)
echarge = 1.602176462e-19  # Proton charge (C)
emass = 9.10938188e-31  # Electron mass (kg)
eps_0 = 8.854187817620e-12  # vacuum permittivity (F/m)
largepos = 1.0e36  # Very large positive number
# ---------------------------------------------------------------------------- #


class MyColors(object):

    def __init__(self):
        """
        Constructor
        """
        self.colors = ['#4B82B8',
                       '#B8474D',
                       '#95BB58',
                       '#234B7C',
                       '#8060A9',
                       '#53A2CB',
                       '#FC943B']

    def __getitem__(self, item):
        return self.colors[int(item % 7)]


class Polygon2D:
    """
    Simple class to handle polygon operations such as point in polygon or
    orientation of rotation (cw or ccw), area, etc.
    """

    def add_point(self, p=None):
        """
        Append a point to the polygon
        """

        if p is not None:

            if isinstance(p, tuple) and len(p) == 2:

                self.poly.append(p)

            else:
                print("Error in add_point of Polygon: p is not a 2-tuple!")

        else:
            print("Error in add_point of Polygon: No p given!")

        return 0

    def add_polygon(self, poly=None):
        """
        Append a polygon object to the end of this polygon
        """

        if poly is not None:

            if isinstance(poly.poly, list) and len(poly.poly) > 0:

                if isinstance(poly.poly[0], tuple) and len(poly.poly[0]) == 2:
                    self.poly.extend(poly.poly)

        return 0

    def area(self):
        """
        Calculates the area of the polygon. only works if there are no crossings

        Taken from http://paulbourke.net, algorithm written by Paul Bourke, 1998

        If area is positive -> polygon is given clockwise
        If area is negative -> polygon is given counter clockwise
        """

        area = 0
        poly = self.poly
        nPts = len(poly)

        j = nPts - 1
        i = 0

        for point in poly:
            p1 = poly[i]
            p2 = poly[j]
            area += (p1[0] * p2[1])
            area -= p1[1] * p2[0]
            j = i
            i += 1

        area /= 2

        return area

    def centroid(self):
        """
        Calculate the centroid of the polygon

        Taken from http://paulbourke.net, algorithm written by Paul Bourke, 1998
        """
        poly = self.poly
        nPts = len(poly)
        x = 0
        y = 0
        j = nPts - 1
        i = 0

        for point in poly:
            p1 = poly[i]
            p2 = poly[j]
            f = p1[0] * p2[1] - p2[0] * p1[1]
            x += (p1[0] + p2[0]) * f
            y += (p1[1] + p2[1]) * f
            j = i
            i += 1

        f = self.area() * 6

        return x / f, y / f

    def clockwise(self):
        """
        Returns True if the polygon points are ordered clockwise

        If area is positive -> polygon is given clockwise
        If area is negative -> polygon is given counter clockwise
        """

        if self.area() > 0:
            return True
        else:
            return False

    def closed(self):
        """
        Checks whether the polygon is closed (i.e first point == last point)
        """

        if self.poly[0] == self.poly[-1]:

            return True

        else:

            return False

    def nvertices(self):
        """
        Returns the number of vertices in the polygon
        """

        return len(self.poly)

    def point_in_poly(self, p=None):
        """
        Check if a point p (tuple of x,y) is inside the polygon
        This is called the "ray casting method": If a ray cast from p crosses
        the polygon an even number of times, it's outside, otherwise inside

        From: http://www.ariel.com.au/a/python-point-int-poly.html

        Note:   Points directly on the edge or identical with a vertex are not
                        considered "inside" the polygon!
        """

        if p is None: return None

        poly = self.poly
        x = p[0]
        y = p[1]
        n = len(poly)
        inside = False

        p1x, p1y = poly[0]

        for i in range(n + 1):

            p2x, p2y = poly[i % n]

            if y > min(p1y, p2y):

                if y <= max(p1y, p2y):

                    if x <= max(p1x, p2x):

                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x

                        if p1x == p2x or x <= xinters:
                            inside = not inside

            p1x, p1y = p2x, p2y

        return inside

    def remove_last(self):
        """
        Remove the last tuple in the ploygon
        """

        self.poly.pop(-1)

        return 0

    def reverse(self):
        """
        Reverses the ordering of the polygon (from cw to ccw or vice versa)
        """

        temp_poly = []
        nv = self.nvertices()

        for i in range(self.nvertices() - 1, -1, -1):
            temp_poly.append(self.poly[i])

        self.poly = temp_poly

        return temp_poly

    def rotate(self, index):
        """
        rotates the polygon, so that the point with index 'index' before now has
        index 0
        """

        if index > self.nvertices() - 1: return 1

        for i in range(index): self.poly.append(self.poly.pop(0))

        return 0

    def __init__(self, poly=None):
        """
        construct a polygon object
        If poly is not specified, an empty polygon is created
        if poly is specified, it has to be a list of 2-tuples!
        """
        self.poly = []

        if poly is not None:

            if isinstance(poly, list) and len(poly) > 0:

                if isinstance(poly[0], tuple) and len(poly[0]) == 2:
                    self.poly = poly

    def __getitem__(self, index):

        return self.poly[index]

    def __setitem__(self, index, value):

        if isinstance(value, tuple) and len(value) == 2:
            self.poly[index] = value

    def __len__(self):

        return len(self.poly)


class BCSEmittanceAnalysis(object):
    """
    """

    def add_roi(self, widget):
        """
        """
        if widget.get_active():

            self.ROIS.append(Polygon2D())

            self.recordROI = True

            widget.set_label("Use mouse for ROI, then click button again!")

        else:

            self.recordROI = False

            widget.set_label("Add ROI to subtract...")

            if not self.ROIS[-1].closed():
                self.ROIS[-1].add_point(self.ROIS[-1][0])

            self.update_gui(self)

        return 0

    def buttonPress(self, widget, event):
        """
        """

        if event.button == 1 and self.recordROI:
            self.insert_point_at_cursor(self)

        return True

    def clear_rois(self, widget):
        """
        """

        self.ROIS = []
        self.update_gui(self)

        return 0

    @staticmethod
    def generate_data_struct():
        """
        """
        raw_data = {"position": np.array([], 'd'),
                    "voltage": np.array([], 'd'),
                    "current": np.array([], 'd')}

        data = {"raw_data": raw_data}

        return data

    def get_connections(self, widget):
        """
        """

        con = {"on_mainWindow_destroy": Gtk.main_quit,
               "on_open_mi_activate": self.open_file,
               "on_save_mi_activate": self.save_files,
               "on_quit_mi_activate": self.quit_all,
               "on_about_mi_activate": self.about,
               "on_recalculate_b_clicked": self.update_gui,
               "on_clear_rois_b_clicked": self.clear_rois,
               "on_add_roi_b_toggled": self.add_roi,
               "on_mainWindow_key_press_event": self.keypress,
               "main_update": self.update_gui,
               "radiobutton_callback": self.radiobutton_callback}

        return con

    def insert_point_at_cursor(self, widget):
        """
        """

        if self.mainPlot.inaxes:

            self.ROIS[-1].add_point((self.mainPlot.mouse_x, self.mainPlot.mouse_y))
            points = np.array(self.ROIS[-1].poly)
            temp_poly = Polygon(points, fill=False)
            if self.temp_roi is not None: self.temp_roi.remove()
            self.temp_roi = self.mainPlot.axis.add_artist(temp_poly)
            self.mainPlot.draw_idle()

        else:

            mymsg = "Mouse has to be inside canvas to create a point!"
            self.status.push(1, mymsg)

        return 0

    @staticmethod
    def keypress(widget, event):
        """
        """
        # keyname = gtk.gdk.keyval_name(event.keyval)
        # print keyname

        return 0

    def about(self, widget):
        print("No About Dialog implemented. GitHub Repo:" \
              "https://github.com/DanielWinklehner/PyEmittanceAnalysis")

        return 0

    def open_file(self, widget):
        """
        """
        of = FileDialog()
        inFileName = of.get_filename(action='open', old_path=self.remember_path(self))

        if inFileName is None: return 1

        self.recordROI = False
        self.ROIS = []
        self.temp_roi = None

        # --- Initialize some lists for the read-in data --- #
        currents = []  # I mean
        plate_voltages = []  # V target
        x_pos = []  # Step Pos

        # --- Read in data --- #
        self.save_path(inFileName)
        with open(inFileName, 'r') as infile:

            data = infile.readlines()

        if data[0].strip() == "# Emittance scan results":
            # mist-1 data file
            numpoints = int(data[3].strip().split(":")[1].strip())
            currents = np.zeros(numpoints)  # I mean
            plate_voltages = np.zeros(numpoints)  # V target
            x_pos = np.zeros(numpoints)  # Step Pos
            for i, line in enumerate(data[5:]):
                _, x_pos[i], plate_voltages[i], currents[i], _ = [item for item in line.strip().split(',')]
            plate_voltages *= 1000.0  # kV --> V

        else:
            # BCS data file
            for item in csv.reader(data, dialect="excel-tab"):

                if "NaN" not in item:

                    currents.append(float(item[2]))
                    plate_voltages.append(float(item[4]))
                    x_pos.append(float(item[9]))

                else:

                    mymsg = "Cave, one or more data entries are NaN!"
                    self.status.push(1, mymsg)

        # --- Converting values into numpy arrays --- #
        self.data["raw_data"]["voltage"] = np.array(plate_voltages, 'd')  # V
        self.data["raw_data"]["current"] = np.array(currents, 'd')  # A
        self.data["raw_data"]["position"] = np.array(x_pos, 'd')  # mm

        self.update_gui(self)

        mymsg = "Opened file %s" % inFileName
        self.status.push(0, mymsg)

        return 0

    def radiobutton_callback(self, widget):
        """
        """
        if widget.get_active():
            self.update_gui(widget)

        return 0

    def remember_path(self, widget):
        """
        Try and open the Path file that stores the last used path for file-
        opening
        """
        oldPath = None

        if os.path.isfile("Path.txt"):

            with open("Path.txt", "rb") as pathFile:

                tempPath = pathFile.readline()

                if os.path.isdir(tempPath):
                    oldPath = tempPath

        return oldPath

    def save_files(self, widget):
        """
        """
        of = FileDialog()
        outFileName = of.get_filename(action='save', old_path=self.remember_path(self))

        if outFileName is None: return 1

        if len(self.data["raw_data"]["current"]) == 0:
            mymsg = "Please load some data first!"
            self.status.push(1, mymsg)
            return 1

        outFileNameImg = os.path.splitext(outFileName)[0] + ".pdf"
        outFileNameDat = os.path.splitext(outFileName)[0] + ".dat"

        # --- Save text file --- "
        with open(outFileNameDat, 'w') as outfile:

            self.save_path(outFileNameDat)
            outfile.write(self.outputTextBuffer.get_text(self.outputTextBuffer.get_start_iter(),
                                                         self.outputTextBuffer.get_end_iter(), False))

        # --- Save Image file --- #
        if not saveExe:

            with rc_context(rc={'text.usetex': True,
                                'font.family': 'serif',
                                'font.serif': 'Computer Modern Roman',
                                'font.weight': 150,
                                'font.size': 18}):

                self.mainPlot.figure.savefig(outFileNameImg, format='pdf', bbox_inches='tight', dpi=150)

        else:

            self.mainPlot.figure.savefig(outFileNameImg, format='png', bbox_inches='tight', dpi=150)

        outFileNameImg = os.path.split(outFileNameImg)[1]
        outFileNameDat = os.path.split(outFileNameDat)[1]

        mymsg = "Files saved as %s and %s" % (outFileNameImg, outFileNameDat)
        self.status.push(0, mymsg)

        return 0

    def save_path(self, filename=None):

        if filename is not None:
            with open("Path.txt", "w") as pathFile:
                pathFile.write(os.path.split(filename)[0])

        return 0

    def update_gui(self, widget, parameter1=None):
        """
        """

        plotSettings = self.mainPlot.get_settings()
        self.mainPlot.clear()
        self.temp_roi = None

        if len(self.data["raw_data"]["current"]) == 0:
            mymsg = "Please load some data first!"
            self.status.push(1, mymsg)
            return 1

        # --- Get some values from GUI and self.data --- #
        raw_data = self.data["raw_data"]
        plateVoltages = raw_data["voltage"]
        xPos = raw_data["position"]
        currents = raw_data["current"]

        sourceHV = int(self.wtree.get_object("hv_sb").get_value())
        mass = float(self.wtree.get_object("mass_sb").get_value())
        chargeState = int(self.wtree.get_object("chargestate_sb").get_value())
        clipNegative = self.wtree.get_object("clip_negative_cb").get_active()
        threshold = float(self.wtree.get_object("threshold_sb").get_value())

        # --- Determine scan mode and adjust labels --- #
        if self.wtree.get_object("horz_rb").get_active():

            plotSettings["entry"]["x_label"] = "x (mm)"
            plotSettings["entry"]["y1_label"] = "x' (mrad)"
            positionSign = 1.0
            angleSign = 1.0

        else:

            plotSettings["entry"]["x_label"] = "y (mm)"
            plotSettings["entry"]["y1_label"] = "y' (mrad)"
            positionSign = 1.0
            angleSign = 1.0

        logarithmicScale = self.wtree.get_object("lognorm_cb").get_active()
        methods = ["nearest", "linear", "cubic"]
        interpolationMethod = methods[int(self.wtree.get_object("interpolation_cb").get_active())]
        mycolormap = self.mycolormaps[int(self.wtree.get_object("colormap_cb").get_active())]
        bgcolors = [(0.0, 0.0, 0.5, 1.0),
                    (0.0, 0.0, 0.0, 1.0),
                    (0.0, 0.0, 0.0, 1.0),
                    (0.0, 0.0, 0.3, 1.0)]
        self.mainPlot.axis.set_facecolor(bgcolors[int(self.wtree.get_object("colormap_cb").get_active())])
        # print mycolormap(0.0) # uncomment this when adding more colormaps to get the background of the plot
        # ---------------------------------------------------------------------------- #

        # --- Calculate some non-relativistic values --- #
        charge = chargeState * echarge  # (C)
        energy = sourceHV * chargeState  # (eV)
        energyJoules = energy * echarge  # (J)
        energyPerAmu = energy / mass  # (eV/amu)
        massKg = mass * amu
        velocity = np.sqrt(2.0 * energyJoules / massKg)  # (m/s)
        beta = velocity / clight
        gamma = 1.0 / np.sqrt(1.0 - beta * beta)

        info = "Beam:\n"
        info += "-----\n"
        info += "Mass: %.4f amu / %.4e kg\n" % (mass, massKg)
        info += "Charge: %i e / %.4e C\n" % (chargeState, charge)
        info += "Energy: %.4e eV / %.4e eV/amu / %.4e J\n" % (energy, energyPerAmu, energyJoules)
        info += "Velocity: %.4e m/s\n" % velocity
        info += "Relativistic factors: Beta = %.5f, Gamma = %.5f\n" % (beta, gamma)

        # --- Converting plate voltage to angle (rad) --- #
        # BCSI:
        # plateDistance = 4.0  # mm
        # plateLength = 76.0  # mm
        # MIST-1:
        # plateDistance = 5.3  # mm
        # plateLength = 96.0  # mm
        plateDistance = float(self.wtree.get_object("plate_d_sb").get_value())  # mm
        plateLength = float(self.wtree.get_object("plate_len_sb").get_value())  # mm
        convFactor = 1000.0 / energy * plateLength / 4.0 / plateDistance
        angles = plateVoltages * convFactor

        # --- Apply threshold to currents --- #
        if clipNegative:
            currents[np.where(currents < 0.0)] = 0.0

        # --- Arranging into numpy sorted array and sorting by 'x' --- #
        dtype = [('x', float), ('xp', float), ('currents', float)]

        _data = [(_pos, _ang, _cur) for _pos, _ang, _cur in zip(positionSign * xPos, angleSign * angles, currents)]

        data = np.array(_data, dtype=dtype)
        data = np.sort(data, order='x')

        # --- Remove points in user defined ROI's --- #
        counter = 0

        for x_temp, xp_temp in zip(data['x'], data['xp']):

            for poly in self.ROIS:

                if poly.point_in_poly(p=(x_temp, xp_temp)):
                    data["currents"][counter] = 0

            counter += 1

        points = np.array([data['x'], data['xp']]).T

        # --- Limits --- #
        xmin = min(points[:, 0])
        xmax = max(points[:, 0])
        xpmin = min(points[:, 1])
        xpmax = max(points[:, 1])

        # --- Total Current --- #
        totalCurrent = np.sum(data["currents"])

        info += "\nTotal current in scan: %.4e uA\n" % (totalCurrent * 1.0e6)

        # --- Means --- #
        xMean = np.sum(data["currents"] * data["x"]) / totalCurrent
        xpMean = np.sum(data["currents"] * data["xp"]) / totalCurrent

        info += "\nGeometric Information:\n"
        info += "----------------------\n"
        info += "Mean position: %.4f mm \n" % xMean
        info += "Mean angle: %.4f mrad  \n" % xpMean

        # --- RMS values --- #
        xRmsSq = np.sum(data["currents"] * (data["x"] - xMean) * (data["x"] - xMean)) / totalCurrent
        xpRmsSq = np.sum(data["currents"] * (data["xp"] - xpMean) * (data["xp"] - xpMean)) / totalCurrent
        xXpRmsSq = np.sum(data["currents"] * (data["x"] - xMean) * (data["xp"] - xpMean)) / totalCurrent

        info += "RMS beam diameter: %.4f mm \n" % (2.0 * np.sqrt(xRmsSq))
        info += "2RMS beam diameter: %.4f mm\n" % (4.0 * np.sqrt(xRmsSq))

        # --- Emittances --- #
        emittanceRms = np.sqrt(xRmsSq * xpRmsSq - xXpRmsSq * xXpRmsSq)
        emittanceRmsNorm = beta * gamma * emittanceRms
        emittance4Rms = 4.0 * emittanceRms
        emittance4RmsNorm = beta * gamma * emittance4Rms

        info += "\nEmittances:\n"
        info += "-----------\n"
        info += "RMS emittance: %.5f pi-mm-mrad \n" % emittanceRms
        info += "4RMS emittance: %.5f pi-mm-mrad\n" % emittance4Rms
        info += "Normalized RMS emittance: %.5f pi-mm-mrad \n" % emittanceRmsNorm
        info += "Normalized 4RMS emittance: %.5f pi-mm-mrad\n" % emittance4RmsNorm

        # --- Twiss parameters --- #
        twissBeta = xRmsSq / emittanceRms
        twissGamma = xpRmsSq / emittanceRms

        # Define the sign of twissAlpha
        checkXp = np.sum(data["currents"] * (data["x"] - xMean) * (data["xp"] - xpMean))

        if checkXp < 0:

            twissAlpha = np.sqrt(twissBeta * twissGamma - 1.0)  # convergent beam

        else:

            twissAlpha = -np.sqrt(twissBeta * twissGamma - 1.0)  # divergent beam

        # --- Angle of rotation of beam ellipse in phase space from Twiss parameters --- #
        phiEllipse = 0.5 * np.arctan2(-2.0 * twissAlpha, twissGamma - twissBeta) * 180.0 / pi + 90.0

        # --- Generate beam ellipses for plot --- #
        hTemp = 0.5 * (twissBeta + twissGamma)

        # 4RMS
        rootHalfEmittance = np.sqrt(0.5 * emittance4Rms)

        R1 = rootHalfEmittance * (np.sqrt(hTemp + 1) + np.sqrt(hTemp - 1))
        R2 = rootHalfEmittance * (np.sqrt(hTemp + 1) - np.sqrt(hTemp - 1))

        ellipse4Rms = Ellipse([xMean, xpMean], 2.0 * R1, 2.0 * R2, angle=-phiEllipse,
                              fill=False,
                              edgecolor=self.colors[2],
                              linestyle="dashed",
                              linewidth=2.0)

        # 1RMS
        rootHalfEmittance = np.sqrt(0.5 * emittanceRms)

        R1 = rootHalfEmittance * (np.sqrt(hTemp + 1) + np.sqrt(hTemp - 1))
        R2 = rootHalfEmittance * (np.sqrt(hTemp + 1) - np.sqrt(hTemp - 1))

        ellipse1Rms = Ellipse([xMean, xpMean], 2.0 * R1, 2.0 * R2, angle=-phiEllipse,
                              fill=False,
                              edgecolor=self.colors[3],
                              linestyle="dashed",
                              linewidth=1.5)

        # --- Calculate percent of beam inside 4 RMS emittance --- #
        current1Rms = 0
        current4Rms = 0

        for xTemp, xpTemp, currentTemp in data:

            if ellipse1Rms.contains_point([xTemp, xpTemp]):
                current1Rms += currentTemp

            if ellipse4Rms.contains_point([xTemp, xpTemp]):
                current4Rms += currentTemp

        emittance1RmsPercent = 100.0 * current1Rms / totalCurrent  # percent
        emittance4RmsPercent = 100.0 * current4Rms / totalCurrent  # percent

        info += "\n1RMS emittance includes %.1f %% of the beam\n" % emittance1RmsPercent
        info += "4RMS emittance includes %.1f %% of the beam\n" % emittance4RmsPercent

        info += "\n1RMS Twiss parameters:\n"
        info += "----------------------\n"
        info += "Alpha = %.4e        \n" % twissAlpha
        info += "Beta = %.4e mm/mrad \n" % twissBeta
        info += "Gamma = %.4e mrad/mm\n" % twissGamma

        # --- Interpolation --- #
        grid_x, grid_xp = np.mgrid[xmin:xmax:200j, xpmin:xpmax:200j]
        sscn_img = griddata(points, data['currents'] * 1.0e9, (grid_x, grid_xp), method=interpolationMethod)

        # --- Thresholding the interpolated data --- #
        sscn_img[np.where(sscn_img < threshold)] = threshold

        # --- Plotting --- #
        if self.wtree.get_object("ellipses_cb").get_active():
            self.mainPlot.axis.add_artist(ellipse1Rms)
            self.mainPlot.axis.add_artist(ellipse4Rms)

        if logarithmicScale:
            norm = LogNorm()
        else:
            norm = Normalize()

        cmScaling = float(self.wtree.get_object("cm_scaling_entry").get_text())

        if cmScaling == -1:

            vmax = max(data["currents"] * 1.0e9)
            vmin = min(data["currents"] * 1.0e9)

        else:

            vmax = cmScaling
            vmin = min(data["currents"] * 1.0e9)

        # cNorm  = mplcol.Normalize(vmin, vmax)
        # scalarMap = cm.ScalarMappable(norm=cNorm, cmap=mycolormap)

        ticks = np.linspace(vmin, vmax, 6)

        if cmScaling == -1:

            img = self.mainPlot.axis.imshow(sscn_img.T,
                                            aspect='auto',
                                            extent=[xmin, xmax, xpmin, xpmax],
                                            origin='lower',
                                            norm=norm,
                                            cmap=mycolormap)
        else:

            img = self.mainPlot.axis.imshow(sscn_img.T,
                                            aspect='auto',
                                            extent=[xmin, xmax, xpmin, xpmax],
                                            origin='lower',
                                            vmax=vmax,
                                            cmap=mycolormap)

        self.mainPlot.set_settings(plotSettings)

        if self.mainPlot.cb is not None:
            self.mainPlot.cb.remove()

        self.mainPlot.cb = self.mainPlot.figure.colorbar(img, ticks=ticks, format=r"%.1f nA")
        self.mainPlot.cb.ax.tick_params(labelsize=16)
        self.mainPlot.cb.set_clim(vmin=vmin, vmax=vmax)
        # img.set_clim(vmin=vmin, vmax=vmax)
        self.mainPlot.cb.set_ticks(ticks)
        self.mainPlot.cb.update_normal(img)
        self.mainPlot.reset_secondary_axes()

        # else:
        #
        #     self.mainPlot.cb.set_cmap(mycolormap)
        #     img.set_clim(vmin=vmin, vmax=vmax)
        #     # self.mainPlot.cb.set_clim(vmin=vmin, vmax=vmax)
        #     self.mainPlot.cb.set_ticks(ticks)
        #     self.mainPlot.cb.update_normal(img)

        # --- Draw ROI's (if any) --- #
        if self.wtree.get_object("show_rois_cb").get_active():

            for roi in self.ROIS:
                points = np.array(roi.poly)
                self.mainPlot.plot(points[:, 0], points[:, 1], color="black")

        self.mainPlot.set_settings(plotSettings)
        self.mainPlot.draw_idle()

        self.outputTextBuffer.set_text(info)

        return 0

    def quit_all(self, widget):
        """
        Function to be called when quit buttons are pressed
        """
        self.window.destroy()
        Gtk.main_quit()

        return 0

    def __init__(self):
        """
        Constructor
        """
        self.colors = MyColors()
        self.data = self.generate_data_struct()
        self.recordROI = False
        self.ROIS = []
        self.temp_roi = None
        # --- Set up the glade file (GUI) and connect signal handlers --- #
        self.gladefile = "GUI_v3.glade"
        self.wtree = Gtk.Builder()
        self.wtree.add_from_file(self.gladefile)

        self.window = self.wtree.get_object("mainWindow")
        self.status = self.wtree.get_object("mainStatusbar")
        self.wtree.connect_signals(self.get_connections(self))
        self.outputTextView = self.wtree.get_object("results_tv")
        self.outputTextBuffer = self.wtree.get_object("results_tb")

        # Some colors:
        pink = np.array([255, 192, 203], 'd') / 255.0
        plum = np.array([221, 160, 221], 'd') / 255.0
        violet = np.array([238, 130, 238], 'd') / 255.0
        mistyrose = np.array([255, 228, 225], 'd') / 255.0
        col1 = np.array([231, 185, 240], 'd') / 255.0
        col2 = np.array([216, 203, 245], 'd') / 255.0
        col3 = np.array([224, 194, 242], 'd') / 255.0

        r, g, b = col3

        cdict = {'red': ((0.0, 0.0, 0.0),
                         (1.0, r, r)),
                 'green': ((0.0, 0.0, 0.0),
                           (1.0, g, g)),
                 'blue': ((0.0, 0.0, 0.0),
                          (1.0, b, b))}

        mycmap = LinearSegmentedColormap("H2PBeam", cdict)

        # self.wtree.get_object("hv_sb").set_value(60000)
        # mass = float(self.wtree.get_object("mass_sb").get_value())
        # chargeState = int(self.wtree.get_object("chargestate_sb").get_value())
        # clipNegative = self.wtree.get_object("clip_negative_cb").get_active()
        # threshold = float(self.wtree.get_object("threshold_sb").get_value())

        # Override negative limit of threshold in glade file
        self.wtree.get_object("threshold_sb").set_range(-1000.0, 1000.0)

        self.wtree.get_object("interpolation_cb").set_active(2)

        self.mycolormaps = [cm.get_cmap("jet"), mycmap, cm.get_cmap("Spectral"), cm.get_cmap("seismic")]

        self.mainPlot = MPLCanvasWrapper(self.window, 0)
        self.wtree.get_object("plot_alignment").add(self.mainPlot)
        # self.mainPlot.set_aspect("equal")
        self.mainPlot.set_title("Phase Space")
        self.mainPlot.set_xlabel("x (mm)")
        self.mainPlot.set_ylabel("x' (mrad)")
        self.mainPlot.connect("button-press-event", self.buttonPress)

        message = " Good morning user! Default values have been loaded...have fun!"
        self.status.push(0, message)

        self.window.set_size_request(1000, 600)
        self.window.show_all()


if __name__ == "__main__":
    AESGUI = BCSEmittanceAnalysis()
    Gtk.main()
