from pathlib import Path
import shutil
import copy
import os
import json
import time
import numpy as np

from PyQt5.QtWidgets import (
        QAction, QApplication, QAbstractItemView,
        QCheckBox, QComboBox,
        QDialog, QDialogButtonBox,
        QFileDialog, QFrame,
        QGridLayout, QGroupBox,
        QHBoxLayout, 
        QLabel, QLineEdit,
        QMainWindow, QMenuBar, QMessageBox,
        QPlainTextEdit, QPushButton,
        QSizePolicy,
        QSpacerItem, QSpinBox, QDoubleSpinBox,
        QStatusBar, QStyle, QStyledItemDelegate,
        QTableView, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, QToolBar,
        QVBoxLayout, 
        QWidget,
        QSlider,
        QHeaderView,
        )
from PyQt5.QtCore import (
        QAbstractTableModel, QCoreApplication, QObject,
        QSize, QTimer, Qt, qVersion, QSettings,
        )
from PyQt5.QtGui import QPainter, QPalette, QColor, QCursor, QIcon, QPixmap, QImage

from PyQt5.QtSvg import QSvgRenderer

from PyQt5.QtXml import QDomDocument

from tiff_loader import TiffLoader
from data_window import DataWindow, SurfaceWindow
from project import Project, ProjectView
from fragment import Fragment, FragmentsModel, FragmentView
from trgl_fragment import TrglFragment, TrglFragmentView
from base_fragment import BaseFragment, BaseFragmentView
from volume import (
        Volume, VolumesModel, 
        DirectionSelectorDelegate,
        ColorSelectorDelegate)
from ppm import Ppm
from utils import COLORLIST

import maxflow
from structure_tensor import eig_special_3d, structure_tensor_3d
from skimage.morphology import flood
from skimage.segmentation import expand_labels, watershed
from skimage.measure import label as apply_labels
import scipy.optimize as optimize
from collections import deque

MAXFLOW_STRUCTURE = np.array(
    [[[1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]],

    [[1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]],

    [[1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]]]
)

SIGNAL_SCALE = 1.0
PLANARITY_SCALE = 1.0
DIRECTION_SCALE = 1.0

BOX_WIDTH = 41
BOX_PADDING = 15
BOX_RADIUS = BOX_WIDTH // 2
MAX_AUTOFILL_DISTANCE = 200

def compute_perp_dists_to_plane(params, x, y, z):
    a, b, c, d = params
    length = np.sqrt(a ** 2 + b ** 2 + c ** 2)
    return (np.abs(a * x + b * y + c * z + d) / length)

def perp_error(params, xyz):
    x, y, z = xyz
    dists = compute_perp_dists_to_plane(params, x, y, z)
    # Compute the mean squared distance (note we drop the square root since
    # it doesn't matter for minimization purposes)
    return np.mean(dists ** 2)

def fit_plane_to_data(x, y, z):
    """
    Basin-Hopping method, minimize mean absolute values of the
    orthogonal distances.
    """
    def unit_length(params):
        """
        Constrain the vector perpendicular to the plane to be of unit length.
        """
        a, b, c, d = params
        return a**2 + b**2 + c**2 - 1

    # Random initial guess for the a,b,c,d plane coefficients.
    initial_guess = np.random.uniform(-10., 10., 4)

    # Constrain the vector perpendicular to the plane to be of unit length.
    cons = ({'type': 'eq', 'fun': unit_length})
    min_kwargs = {"constraints": cons, "args": [x, y, z]}

    # obtain the best fit coefficients.
    sol = optimize.minimize(
        perp_error, initial_guess, **min_kwargs, method="SLSQP", 
        options={"disp": False},
    )
    return list(sol.x)

def get_new_box_centers(subsection, saved_labels, init_coord, saved_label_id):
    zpos, ypos, xpos = init_coord
    mask = saved_labels == saved_label_id
    
    z, y, x = np.meshgrid(*[np.arange(s) for s in subsection.shape], indexing='ij')
    
    for idx in [0, -1]:
        if np.any(mask[idx, :, :]):
            yield (
                zpos + (BOX_PADDING if idx == -1 else -BOX_PADDING),
                ypos + int(np.median(y[idx, :, :][mask[idx, :, :]])) - BOX_RADIUS,
                xpos + int(np.median(x[idx, :, :][mask[idx, :, :]])) - BOX_RADIUS,
            )
        if np.any(mask[:, idx, :]):
            yield (
                zpos + int(np.median(z[:, idx, :][mask[:, idx, :]])) - BOX_RADIUS,
                ypos + (BOX_PADDING if idx == -1 else -BOX_PADDING),
                xpos + int(np.median(x[:, idx, :][mask[:, idx, :]])) - BOX_RADIUS,
            )
        if np.any(mask[:, :, idx]):
            yield (
                zpos + int(np.median(z[:, :, idx][mask[:, :, idx]])) - BOX_RADIUS,
                ypos + int(np.median(y[:, :, idx][mask[:, :, idx]])) - BOX_RADIUS,
                xpos + (BOX_PADDING if idx == -1 else -BOX_PADDING),
            )

def check_box_for_extension(subsection, saved_labels, saved_label_id):
    local_labels = find_sheets(subsection, saved_labels)
    z, y, x = np.meshgrid(*[np.arange(s) for s in local_labels.shape], indexing='ij')
    linkages = compute_linkages(local_labels, saved_labels)
    valid_new_label_ids = [l for l in linkages if (len(linkages[l]) == 1) and (saved_label_id in linkages[l])]
    if not valid_new_label_ids:
        return None
    saved_mask = saved_labels == saved_label_id
    plane_params = fit_plane_to_data(x[saved_mask], y[saved_mask], z[saved_mask])
    dists = compute_perp_dists_to_plane(plane_params, x, y, z)
    if np.percentile(dists[saved_mask], 95) > 8 or np.sum(saved_mask) < 50:
        # The existing annotations do not fit a plane well enough to confidently extend
        return None
    for new_label_id in valid_new_label_ids:
        new_mask = local_labels == new_label_id
        dist_95perc = np.percentile(dists[new_mask], 95)
        #print(dist_95perc)
        if dist_95perc < 12 and np.sum(new_mask) > 50:
            yield new_mask

def compute_linkages(new_labels, saved_labels):
    """Counts the number of adjacent positions between every two labels.
    """
    linkages = {}
    if saved_labels is None:
        return linkages
    expanded = expand_labels(new_labels)
    for l in [label for label in np.unique(expanded) if label > 0]:
        adjacent = (expanded == l) & (saved_labels > 0)
        links = {}
        for a in np.unique(saved_labels[adjacent]):
            mask = adjacent & (saved_labels == a)
            links[a] = mask.sum()
        linkages[l] = links
    return linkages

def find_sheets(section, saved_labels=None, threshold=30000, minsize=25, minval=2500):
    """Takes a 3D numpy array of uint16s and tries to separate papyrus sheets from 
    background.  First applies a simple threshold, then filters the resulting
    regions for both pixel count and average signal.

    Returns a new numpy array of the same size of integers indicating which 
    label is associated with each pixel.  
    """
    if saved_labels is None:
        mask = (section > threshold)
    else:
        mask = (section > threshold) & (saved_labels == 0)
    section_labels = apply_labels(mask).astype(np.uint16)
    label_ids = np.unique(section_labels)
    linkages = compute_linkages(section_labels, saved_labels)
    for l in label_ids:
        if l == 0:
            # These were below the threshold
            continue
        mask = section_labels == l
        count = mask.sum()
        if saved_labels is not None:
            links = linkages[l]
            for key in links:
                mask |= saved_labels == key
        value = np.mean(section[mask]) - threshold
        if count < minsize or value < minval:
            section_labels[mask] = 0
    # Resort the labels to increase incrementally from 1
    label_ids = sorted(np.unique(section_labels))
    for i, l in enumerate(label_ids):
        if l == 0:
            continue
        mask = section_labels == l
        section_labels[mask] = i
    return section_labels


def split_label_maxflow(
        section_signal,
        section_labels,
        new_labels,
        split_label,
        new_label,
        source_label,
        sink_label,
    ):
    """Tries to split one of the new labels into two separate labels, one connected to the source
    label and one connected to the sink label.  Uses a maximum flow/minimum cut algorithm to 
    try to find the splitting surface that goes through the fewest pixels with the minimum signal.
    """
    int16 = (1 << 16) - 1
    norm_signal = (section_signal / int16).astype(float)
    # ST parameters are still experimental and may need adjusting for different datasets
    S = structure_tensor_3d(norm_signal, sigma=0.5, rho=0.5)
    eigval, eigvec = eig_special_3d(S, full=True)
    # Extract only the eigenvector corresponding to the largest eigenvalue and reverse
    # axis order so that it corresponds to the input matrix shape.
    eigvec = eigvec[2, ::-1, :, :, :]

    # N.B. for some use cases we may want to re-orient to positive z
    # An estimate of planarity:  things are planar when the largest eigenvalue is 
    # significantly larger than the other two
    # note we want *lower* weights for *higher* planarity, so we invert to encourage cuts in planar regions
    planarity = eigval[2] / eigval[1]
    scale = np.log10(1 / planarity)
    scale -= scale.min()

    graph = maxflow.GraphFloat()
    nodes = graph.add_grid_nodes(section_labels.shape)
    weights = np.zeros_like(norm_signal)
    keep_mask = (section_labels == source_label) | (section_labels == sink_label) | (new_labels == split_label)
    weights[keep_mask] = norm_signal[keep_mask]
    weights = np.power(weights, 1.2) * SIGNAL_SCALE * scale * PLANARITY_SCALE
    for i in range(3):
        for j in range(3):
            for k in range(3):
                z, y, x = (i-1, j-1, k-1)
                if MAXFLOW_STRUCTURE[i,j,k]:
                    structure = np.zeros((3, 3, 3))
                    structure[i,j,k] = 1
                    normvec = np.array([z, y, x], dtype=float)
                    normvec /= np.sqrt(np.sum(normvec ** 2))
                    eigvec_scale = 1 - np.abs(np.dot(eigvec.transpose(), normvec).transpose())
                    local_weights = weights * eigvec_scale * DIRECTION_SCALE
                    graph.add_grid_edges(nodes, weights=local_weights, structure=structure, symmetric=False)

    #graph.add_grid_edges(nodes, weights=weights, structure=MAXFLOW_STRUCTURE)
    # Add extremely high capacities to the source & sink nodes
    sourcecaps = np.zeros_like(weights)
    sinkcaps = np.zeros_like(weights)
    # Split by applying source/sinks along the extrema along one axis
    MAXWIDTH = 8
    if source_label == "z":
        zvals = [i for i in range(section_labels.shape[0]) if np.any(keep_mask[i, :, :])]
        minz, maxz = min(zvals), max(zvals)
        width = max(1, min(MAXWIDTH, (maxz - minz) // 3))
        sourcecaps[minz:minz+width, :, :][keep_mask[minz:minz+width, :, :]] = 1e6
        sinkcaps[maxz-width:maxz, :, :][keep_mask[maxz-width:maxz, :, :]] = 1e6
    elif source_label == "y":
        yvals = [i for i in range(section_labels.shape[1]) if np.any(keep_mask[:, i, :])]
        miny, maxy = min(yvals), max(yvals)
        width = max(1, min(MAXWIDTH, (maxy - miny) // 3))
        sourcecaps[:, miny:miny+width, :][keep_mask[:, miny:miny+width, :]] = 1e6
        sinkcaps[:, maxy-width:maxy, :][keep_mask[:, maxy-width:maxy, :]] = 1e6
    elif source_label == "x":    
        xvals = [i for i in range(section_labels.shape[2]) if np.any(keep_mask[:, :, i])]
        minx, maxx = min(xvals), max(xvals)
        width = max(1, min(MAXWIDTH, (maxx - minx) // 3))
        sourcecaps[:, :, minx:minx+width][keep_mask[:, :, minx:minx+width]] = 1e6
        sinkcaps[:, :, maxx-width:maxx][keep_mask[:, :, maxx-width:maxx]] = 1e6
    elif source_label == "n":
        # Split by applying source/sinks along the extrema along the "natural" axis
        # Estimate the natural axis by the best-fit line to the eigenvectors in this region
        zvals = [i for i in range(section_labels.shape[0]) if np.any(keep_mask[i, :, :])]
        minz, maxz = min(zvals), max(zvals)
        yvals = [i for i in range(section_labels.shape[1]) if np.any(keep_mask[:, i, :])]
        miny, maxy = min(yvals), max(yvals)
        xvals = [i for i in range(section_labels.shape[2]) if np.any(keep_mask[:, :, i])]
        minx, maxx = min(xvals), max(xvals)
        col_eigvec = eigvec[:, minz:maxz, miny:maxy, minx:maxx]
        col_eigvec = eigvec[:, keep_mask].T

        def perp_error(params, col_eigvec):
            a, b, c = params
            local_vec = np.array([a, b, c])
            return np.mean((np.cross(local_vec, col_eigvec) ** 2).sum(axis=1))

        def unit_length(params):
            """
            Constrain the vector perpendicular to the plane to be of unit length.
            """
            a, b, c = params
            return a**2 + b**2 + c**2 - 1

        initial_guess = [0, 1, 0]
        cons = ({'type': 'eq', 'fun': unit_length})
        min_kwargs = {"constraints": cons, "args": col_eigvec}
        solution = optimize.minimize(
            perp_error, initial_guess, **min_kwargs, method="SLSQP",
            options={"disp": False},
        )
        meanvec = np.array(list(solution.x))
        print(f"Natural vector: Z {meanvec[0]:.2f} Y {meanvec[1]:.2f} X {meanvec[2]:.2f}")
        z, y, x = np.meshgrid(*[np.arange(s) for s in section_labels.shape], indexing='ij')
        natural_dists = z * meanvec[0] + y * meanvec[1] + x * meanvec[2]
        mind, maxd = np.min(natural_dists[keep_mask]), np.max(natural_dists[keep_mask])
        width = max(1, min(MAXWIDTH, (maxd - mind) // 3))
        sourcemask = (natural_dists < (mind + width)) & keep_mask
        sourcecaps[sourcemask] = 1e6
        sinkmask = (natural_dists > (maxd - width)) & keep_mask
        sinkcaps[sinkmask] = 1e6
    else:
        sourcecaps[section_labels == source_label] = 1e6
        sinkcaps[section_labels == sink_label] = 1e6

    graph.add_grid_tedges(nodes, sourcecaps, sinkcaps)
    graph.maxflow()
    sgm = graph.get_grid_segments(nodes) & (new_labels == split_label)
    new_labels[sgm] = new_label
    return new_labels


class AnnotationWindow(QWidget):

    def __init__(self, main_window, slices=11):
        super(AnnotationWindow, self).__init__()
        self.show()
        self.setWindowTitle("Volume Annotations")
        self.main_window = main_window
        self.volume_view = None
        # This is a copy of the central region of the box, for doing
        # automatic volume segmentation on.
        self.signal_section = None
        self.new_labels = None
        self.saved_labels = None
        self.slices = slices
        # Radius of the central region of the box
        self.radius = [30, 30, 30]
        self.update_annotations = False
        # Signal threshold for auto-labeling
        self.threshold = 30000

        grid = QGridLayout()
        self.depth = [
            DataWindow(self.main_window, 2)
            for i in range(slices)
        ]
        self.inline = [
            DataWindow(self.main_window, 0)
            for i in range(slices)
        ]
        self.xline = [
            DataWindow(self.main_window, 1)
            for i in range(slices)
        ]

        for i in range(slices):
            grid.addWidget(self.xline[i], 0, i)
            grid.addWidget(self.inline[i], 1, i)
            grid.addWidget(self.depth[i], 2, i)

        # Set up the control panel below the images
        panel = QWidget()
        hlayout = QHBoxLayout()
        vlayout = QVBoxLayout()
        label = QLabel("New volume segments")
        label.setAlignment(Qt.AlignLeft)
        vlayout.addWidget(label)
        auto_annotate = QCheckBox("Auto-update annotations")
        auto_annotate.setChecked(self.update_annotations)
        auto_annotate.stateChanged.connect(self.checkAutoUpdate)
        vlayout.addWidget(auto_annotate)

        # Slider for controlling the auto-labeling threshold
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Signal Threshold:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(29000)
        self.threshold_slider.setMaximum(50000)
        self.threshold_slider.setSingleStep(1000)
        self.threshold_slider.setPageStep(1000)
        self.threshold_slider.setValue(self.threshold)
        self.threshold_slider.valueChanged.connect(self.updateSliders)
        hl.addWidget(self.threshold_slider)
        self.threshold_display = QLabel(str(self.threshold))
        hl.addWidget(self.threshold_display)
        vlayout.addLayout(hl)

        # Sliders for controlling the radius of annotation along each axis
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Z Radius:"))
        self.zslider = QSlider(Qt.Horizontal)
        self.zslider.setMinimum(5)
        self.zslider.setMaximum(75)
        self.zslider.setValue(self.radius[0])
        self.zslider.setSingleStep(5)
        self.zslider.setPageStep(5)
        self.zslider.valueChanged.connect(self.updateSliders)
        hl.addWidget(self.zslider)
        self.zrad = QLabel(str(self.radius[0]))
        hl.addWidget(self.zrad)
        vlayout.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Y Radius:"))
        self.yslider = QSlider(Qt.Horizontal)
        self.yslider.setMinimum(5)
        self.yslider.setMaximum(75)
        self.yslider.setValue(self.radius[1])
        self.yslider.setSingleStep(5)
        self.yslider.setPageStep(5)
        self.yslider.valueChanged.connect(self.updateSliders)
        hl.addWidget(self.yslider)
        self.yrad = QLabel(str(self.radius[1]))
        hl.addWidget(self.yrad)
        vlayout.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("X Radius:"))
        self.xslider = QSlider(Qt.Horizontal)
        self.xslider.setMinimum(5)
        self.xslider.setMaximum(75)
        self.xslider.setValue(self.radius[2])
        self.xslider.setSingleStep(5)
        self.xslider.setPageStep(5)
        self.xslider.valueChanged.connect(self.updateSliders)
        hl.addWidget(self.xslider)
        self.xrad = QLabel(str(self.radius[2]))
        hl.addWidget(self.xrad)
        vlayout.addLayout(hl)

        for i in range(5):
            # Trying to pad things out a bit
            vlayout.addWidget(QLabel(""))

        hlayout.addLayout(vlayout, stretch=1)

        # Table of annotations
        vlayout = QVBoxLayout()
        self.saved_table = QTableWidget(panel)
        labels = [
            "Saved Label ID",
            "Color",
            "Clear",
            "Auto-Fill",
        ]
        self.saved_table.setColumnCount(len(labels))
        self.saved_table.setHorizontalHeaderLabels(labels)
        header = self.saved_table.horizontalHeader()
        for i in range(len(labels)):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        vlayout.addWidget(self.saved_table)
        hlayout.addLayout(vlayout, stretch=3)

        # Table of novel annotations
        vlayout = QVBoxLayout()
        self.new_table = QTableWidget(panel)
        labels = [
            "New Label ID",
            "Color",
            "# Pixels",
            "Mean Signal",
            "Linked Labels",
            "ID to Save As",
            "Split IDs",
            "Save",
            "Split",
        ]
        self.new_table.setColumnCount(len(labels))
        self.new_table.setHorizontalHeaderLabels(labels)
        header = self.new_table.horizontalHeader()
        for i in range(len(labels)):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        vlayout.addWidget(self.new_table)
        hlayout.addLayout(vlayout, stretch=3)

        panel.setLayout(hlayout)

        grid.addWidget(panel, 3, 0, 3, self.slices + 1, Qt.AlignmentFlag.AlignTop)

        self.setLayout(grid)

    def checkAutoUpdate(self, checkbox):
        self.update_annotations = checkbox == Qt.Checked
        if self.update_annotations:
            self.drawSlices()


    def updateSliders(self, slider):
        self.threshold = self.threshold_slider.value()
        self.radius[0] = self.zslider.value()
        self.radius[1] = self.yslider.value()
        self.radius[2] = self.xslider.value()
        self.threshold_display.setText(str(self.threshold_slider.value()))
        self.zrad.setText(str(self.zslider.value()))
        self.yrad.setText(str(self.yslider.value()))
        self.xrad.setText(str(self.xslider.value()))
        self.drawSlices()


    def setVolumeView(self, volume_view):
        self.volume_view = volume_view
        if volume_view is None:
            return
        for i in range(self.slices):
            for datawindow in [self.depth[i], self.inline[i], self.xline[i]]:
                datawindow.setVolumeView(volume_view)

    def get_colors(self):
        colors_used = set()
        colormap = {}
        all_labels = []
        if self.saved_labels is not None:
            all_labels.extend([idx for idx in np.unique(self.saved_labels) if idx > 0])
        if self.new_labels is not None:
            all_labels.extend([idx for idx in np.unique(self.new_labels) if idx != 0])
        all_labels = np.unique(all_labels)
        for idx in all_labels:
            color_idx = idx % len(COLORLIST)
            colors_used.add(color_idx)
            colormap[idx] = color_idx
        return colormap

    def update_saved_label_table(self):
        if self.saved_labels is None:
            return
        colormap = self.get_colors()
        idxs = [idx for idx in np.unique(self.saved_labels) if idx > 0]
        self.saved_table.clearContents()
        self.saved_table.setRowCount(len(idxs))
        for row_i, idx in enumerate(idxs):
            #mask = self.saved_labels == idx
            self.saved_table.setItem(row_i, 0, QTableWidgetItem(str(idx)))
            # A cell needs empty text to be able to set the background color
            self.saved_table.setItem(row_i, 1, QTableWidgetItem(""))
            self.saved_table.item(row_i, 1).setBackground(COLORLIST[colormap[idx]])
            # Clear button
            button = QPushButton()
            button.setText("Clear")
            button.clicked.connect(self.clear_button_clicked)
            self.saved_table.setCellWidget(row_i, 2, button)
            # Auto-fill button
            button = QPushButton()
            button.setText("Fill")
            button.clicked.connect(self.autofill_button_clicked)
            self.saved_table.setCellWidget(row_i, 3, button)

    def update_new_label_table(self):
        if self.new_labels is None:
            return
        colormap = self.get_colors()
        linkages = compute_linkages(self.new_labels, self.saved_labels)
        idxs = [idx for idx in np.unique(self.new_labels) if idx != 0]
        self.new_table.clearContents()
        self.new_table.setRowCount(len(idxs))
        for row_i, idx in enumerate(idxs):
            mask = self.new_labels == idx
            self.new_table.setItem(row_i, 0, QTableWidgetItem(str(idx)))
            # A cell needs empty text to be able to set the background color
            self.new_table.setItem(row_i, 1, QTableWidgetItem(""))
            self.new_table.item(row_i, 1).setBackground(COLORLIST[colormap[idx]])
            self.new_table.setItem(row_i, 2, QTableWidgetItem(f"{mask.sum():,}"))
            self.new_table.setItem(row_i, 3, 
                                   QTableWidgetItem(f"{int(np.mean(self.signal_section[mask])):,}")
                                                    )
            # Linkages
            links = linkages[idx] if idx in linkages else {}
            self.new_table.setItem(row_i, 4, QTableWidgetItem(str(links)))

            # The label IDs to save to
            line = QLineEdit()
            if len(links) == 1:
                line.setText(str(list(links.keys())[0]))
            self.new_table.setCellWidget(row_i, 5, line)

            # Split IDs
            line = QLineEdit()
            if len(links) == 2:
                ids = list(links.keys())
                line.setText(f"{ids[0]},{ids[1]}")
            self.new_table.setCellWidget(row_i, 6, line)

            # Save button
            button = QPushButton()
            button.setText("Save")
            button.clicked.connect(self.save_button_clicked)
            self.new_table.setCellWidget(row_i, 7, button)

            # Split button
            button = QPushButton()
            button.setText("Split")
            button.clicked.connect(self.split_button_clicked)
            self.new_table.setCellWidget(row_i, 8, button)

    def save_button_clicked(self):
        button = self.sender()
        col_idx = 7
        row_idx = None
        # TODO: this is a terrible hack
        for i in range(self.new_table.rowCount()):
            if self.new_table.cellWidget(i, col_idx) == button:
                row_idx = i
        if row_idx is None:
            print("Button sender not found")
            return
        if self.main_window.annotation is None:
            print("Cannot save data without annotation file loaded")
            return
        label_id = int(self.new_table.item(row_idx, 0).text())
        new_label_id = self.new_table.cellWidget(row_idx, 5).text()
        if not new_label_id:
            print("No saved label provided")
            return
        try:
            new_label_id = int(new_label_id)
        except:
            print("Non-integer saved label provided")
            return
        mask = self.new_labels == label_id
        self.saved_labels[mask] = new_label_id
        vv = self.volume_view
        vol = self.volume_view.volume
        it, jt, kt = vol.transposedIjkToIjk(vv.ijktf, vv.direction)
        islice = slice(
            max(0, it - self.radius[2]), 
            min(vol.data.shape[2], it + self.radius[2] + 1),
            None,
        )
        jslice = slice(
            max(0, jt - self.radius[1]), 
            min(vol.data.shape[1], jt + self.radius[1] + 1),
            None,
        )
        kslice = slice(
            max(0, kt - self.radius[0]), 
            min(vol.data.shape[0], kt + self.radius[0] + 1),
        )
        self.main_window.annotation.write_annotations(self.saved_labels, islice, jslice, kslice)
        self.drawSlices()

    def split_button_clicked(self):
        button = self.sender()
        col_idx = 8
        row_idx = None
        for i in range(self.new_table.rowCount()):
            if self.new_table.cellWidget(i, col_idx) == button:
                row_idx = i
        if row_idx is None:
            print("Button sender not found")
            return
        label_id = int(self.new_table.item(row_idx, 0).text())
        new_label_id = np.max(self.new_labels) + 1
        split_str = self.new_table.cellWidget(row_idx, 6).text().lower()
        if not split_str:
            print("Defaulting to natural axis split")
            #print("No split labels provided")
            #return
            split_str = "n"
        if split_str in ["x", "y", "z", "n"]:
            split1, split2 = split_str, None
        else:
            try:
                split1, split2 = tuple([int(s) for s in split_str.split(",")])
            except:
                print("Invalid split labels provided")
                return
        self.new_labels = split_label_maxflow(
            self.signal_section,
            self.saved_labels,
            self.new_labels,
            label_id,
            new_label_id,
            split1,
            split2,
        )
        self.update_new_label_table()
        self.drawSlices(update_labels=False)

    def clear_button_clicked(self):
        button = self.sender()
        col_idx = 2
        row_idx = None
        for i in range(self.saved_table.rowCount()):
            if self.saved_table.cellWidget(i, col_idx) == button:
                row_idx = i
        if row_idx is None:
            print("Button sender not found")
            return
        label_id = int(self.saved_table.item(row_idx, 0).text())
        mask = self.saved_labels == label_id
        # Set back to 0 to clear the annotation
        self.saved_labels[mask] = 0
        vv = self.volume_view
        vol = self.volume_view.volume
        it, jt, kt = vol.transposedIjkToIjk(vv.ijktf, vv.direction)
        islice = slice(
            max(0, it - self.radius[2]), 
            min(vol.data.shape[2], it + self.radius[2] + 1),
            None,
        )
        jslice = slice(
            max(0, jt - self.radius[1]), 
            min(vol.data.shape[1], jt + self.radius[1] + 1),
            None,
        )
        kslice = slice(
            max(0, kt - self.radius[0]), 
            min(vol.data.shape[0], kt + self.radius[0] + 1),
        )
        self.main_window.annotation.write_annotations(self.saved_labels, islice, jslice, kslice)
        self.drawSlices()

    def autofill_button_clicked(self):
        if not self.update_annotations:
            print("Autofill not enabled without annotations")
            return
        button = self.sender()
        col_idx = 3
        row_idx = None
        for i in range(self.saved_table.rowCount()):
            if self.saved_table.cellWidget(i, col_idx) == button:
                row_idx = i
        if row_idx is None:
            print("Button sender not found")
            return
        saved_label_id = int(self.saved_table.item(row_idx, 0).text())
        vv = self.volume_view
        vol = self.volume_view.volume
        # Pull out the initial positions to make sure we can reset to here after we're finished
        it, jt, kt = vol.transposedIjkToIjk(vv.ijktf, vv.direction)
        print(f"Auto-filling label {saved_label_id} around position ({it}, {jt}, {kt})")

        # Get shortcuts for the global-level signal & annotation data
        section = vol.data
        labels = self.main_window.annotation.volume

        position_queue = deque()
        xpos, ypos, zpos = it, jt, kt
        start_coord = (zpos, ypos, xpos)
        zslice = slice(max(0, zpos - BOX_RADIUS), min(section.shape[0] - 1, zpos + BOX_RADIUS), None)
        yslice = slice(max(0, ypos - BOX_RADIUS), min(section.shape[1] - 1, ypos + BOX_RADIUS), None)
        xslice = slice(max(0, xpos - BOX_RADIUS), min(section.shape[2] - 1, xpos + BOX_RADIUS), None)
        subsection = section[zslice, yslice, xslice]
        saved_labels = labels[zslice, yslice, xslice]
        for new_coord in get_new_box_centers(subsection, saved_labels, start_coord, saved_label_id):
            position_queue.append(new_coord)
        total_pixels = 0
        total_masks = 0
        i = 0
        while len(position_queue) > 0:
            i += 1
            print(f"Processing coord {i}", end="\r")
            new_coord = position_queue.popleft()
            zpos, ypos, xpos = new_coord
            zslice = slice(max(0, zpos - BOX_RADIUS), min(section.shape[0] - 1, zpos + BOX_RADIUS), None)
            yslice = slice(max(0, ypos - BOX_RADIUS), min(section.shape[1] - 1, ypos + BOX_RADIUS), None)
            xslice = slice(max(0, xpos - BOX_RADIUS), min(section.shape[2] - 1, xpos + BOX_RADIUS), None)
            subsection = section[zslice, yslice, xslice]
            saved_labels = labels[zslice, yslice, xslice]
            result = check_box_for_extension(subsection, saved_labels, saved_label_id)
            if result is None:
                continue
            for new_mask in result:
                saved_labels[new_mask] = saved_label_id
                self.main_window.annotation.write_annotations(saved_labels, xslice, yslice, zslice)
                total_pixels += new_mask.sum()
                total_masks += 1
                for extended_coord in get_new_box_centers(subsection, new_mask, new_coord, True):
                    zpos, ypos, xpos = extended_coord
                    if max([abs(xpos - it), abs(ypos - jt), abs(zpos - kt)]) > MAX_AUTOFILL_DISTANCE:
                        continue
                    position_queue.append(extended_coord)
            if len(position_queue) > 10000:
                print("Position queue too large; aborting autofill")
                break
        print(f"Finished autofill: processed {total_masks} active regions and autofilled {total_pixels} pixels")
        self.drawSlices()

    def drawSlices(self, update_labels=True):
        vv = self.volume_view
        vol = self.volume_view.volume
        it, jt, kt = vol.transposedIjkToIjk(vv.ijktf, vv.direction)
        islice = slice(
            max(0, it - self.radius[2]), 
            min(vol.data.shape[2], it + self.radius[2] + 1),
            None,
        )
        jslice = slice(
            max(0, jt - self.radius[1]), 
            min(vol.data.shape[1], jt + self.radius[1] + 1),
            None,
        )
        kslice = slice(
            max(0, kt - self.radius[0]), 
            min(vol.data.shape[0], kt + self.radius[0] + 1),
            None,
        )
        if self.update_annotations and update_labels:
            # Pull in saved annotations if available
            if self.main_window.annotation is not None:
                self.saved_labels = self.main_window.annotation.volume[kslice, jslice, islice]
                self.update_saved_label_table()
            else:
                self.saved_labels = None

            # Compute local annotations
            self.signal_section = vol.data[kslice, jslice, islice]
            self.new_labels = find_sheets(self.signal_section, self.saved_labels, threshold=self.threshold)
            self.update_new_label_table()

        # Actually draw the datawindows with appropriate shape offsets
        for i in range(self.slices):
            for datawindow, r in zip([self.xline[i], self.depth[i], self.inline[i]], self.radius):
                offsets = [0, 0, 0]
                axis = datawindow.axis
                offsets[axis] += (i - (self.slices // 2)) * ((r * 2) // (self.slices - 1))
                if self.update_annotations:
                    # Add an overlay of the annotations to each view.  First get the center
                    # point for this slice and pull out the labels for this region.
                    ijktf = list(vv.ijktf)
                    ijktf[axis] += offsets[axis]
                    # Coords in absolute space
                    it, jt, kt = vol.transposedIjkToIjk(vv.ijktf, vv.direction)
                    # Coords in local label volume space
                    iidx = list(range(vol.data.shape[2])[islice]).index(it)
                    jidx = list(range(vol.data.shape[1])[jslice]).index(jt)
                    kidx = list(range(vol.data.shape[0])[kslice]).index(kt)
                    slc = vv.getSlice(axis, ijktf)
                    if self.main_window.annotation is not None:
                        overlay = np.copy(self.main_window.annotation.getSlice(vv, axis, ijktf))
                        assert overlay.shape == slc.shape
                    else:
                        overlay = np.zeros_like(slc, dtype=int)
                    # TODO: pasting the overlay data into the center is a hack that 
                    # assumes we're not near the edge of a volume
                    if axis == 1:
                        tmp = (
                            self.new_labels[kidx + offsets[axis], :, :] +
                            self.saved_labels[kidx + offsets[axis], :, :]
                        )
                        x = (overlay.shape[0] - tmp.shape[0]) // 2
                        y = (overlay.shape[1] - tmp.shape[1]) // 2
                        overlay[x:x + tmp.shape[0], y:y + tmp.shape[1]] = tmp
                    elif axis == 2:
                        tmp = (
                            self.new_labels[:, jidx + offsets[axis], :] +
                            self.saved_labels[:, jidx + offsets[axis], :]
                        )
                        x = (overlay.shape[0] - tmp.shape[0]) // 2
                        y = (overlay.shape[1] - tmp.shape[1]) // 2
                        overlay[x:x + tmp.shape[0], y:y + tmp.shape[1]] = tmp
                    elif axis == 0:
                        tmp = (
                            self.new_labels[:, :, iidx + offsets[axis]].T +
                            self.saved_labels[:, :, iidx + offsets[axis]].T
                        )
                        x = (overlay.shape[0] - tmp.shape[0]) // 2
                        y = (overlay.shape[1] - tmp.shape[1]) // 2
                        overlay[x:x + tmp.shape[0], y:y + tmp.shape[1]] = tmp
                else:
                    overlay = None

                datawindow.drawSlice(offsets, crosshairs=False, fragments=False, overlay=overlay)

    def closeEvent(self, event):
        """We need to reset the main window's link to this when 
        the user closes this window.
        """
        print("Closing window")
        self.main_window.annotation_window = None
        event.accept()