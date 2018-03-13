from __future__ import print_function
from read_dftc_info import read_dftc_info
from read_dftc_info import get_particles_type

# math
import numpy as np
import math

# file processing
import os.path
import sys
import string
import csv
import re
from lxml import etree
import StringIO



"""   ===================================    SORTING DIRECTORIES    ===================================  """
"""
      These function work for directory name of kind: ./AspectRatio_6.2_512x128x128/data_x_1.0e-03__rV_0.0__wx0.010__dt0.001__N100000__2016-05-01_20:45:24,
      so check if the directories names are similar.
"""

def sort_dirs_by_aspects(dirs,split=False):
    """
    Sorts names of directories by 1-th float in this name (it should be aspect ratio of a trap by default!).
    
    @param dirs  - list of directories with data (necessarly full with aspect ratio and mesh size).
    @param split - if true returns list of lists where every list contains directories' names with the same interaction parameter
    @return      - list of diretories sorted by interaction parameter strength, 1D if split False, 2D if split True.
    """
    
    match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
    aspects = [[float(aspect) for aspect in re.findall(match_number, dirname)][0] for dirname in dirs]
    
    if split is True:
        dirslist = []
        for aspect in sorted(set(aspects)):
            newdirs = []
            pattern = 'AspectRatio_{:.1f}'.format(aspect)
            for dirname in dirs:
                if pattern in dirname:
                    newdirs.append(dirname)
            dirslist.append(newdirs)
        #print(len(set(aspects)),len(dirslist))
        return dirslist
    else:
        return [dirname for (aspect,dirname) in sorted(zip(aspects,dirs))]


def sort_dirs_by_x(dirs,split=False):
    """
    Sorts names of directories by 5-th float in this name (it should be interaction parameter x by default!).
    
    @param dirs  - list of directories with data (necessarly full with aspect ratio and mesh size).
    @param split - if true returns list of lists where every list contains directories' names with the same interaction parameter
    @return      - list of diretories sorted by interaction parameter strength, 1D if split False, 2D if split True.
    """
    
    match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
    xs = [[float(x) for x in re.findall(match_number, dirname)][4] for dirname in dirs]
    
    if split is True:
        dirslist = []
        for x in sorted(set(xs)):
            newdirs = []
            pattern = 'x_{:.1e}'.format(x)
            for dirname in dirs:
                if pattern in dirname:
                    newdirs.append(dirname)
            dirslist.append(newdirs)
        #print(len(set(xs)),len(dirslist))
        return dirslist
    else:
        return [dirname for (x,dirname) in sorted(zip(xs,dirs))]


def sort_dirs_by_rV(dirs,split=False,fltpts=1):
    """
    Sorts names of directories by 6-th float in this name (it should be initial position of vortex by default!).
    
    @param dirs  - list of directories with data (necessarly full with aspect ratio and mesh size).
    @param split - if true returns list of lists where every list contains directories' names with the same interaction parameter
    @return      - list of diretories sorted by interaction parameter strength, 1D if split False, 2D if split True.
    """
    
    #match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
    match_number = re.compile('[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?')
    #for dirname in dirs:
    #    print([float(rV[0]) for rV in re.findall(match_number, dirname)][5])
    
    rVs = [[float(rV[0]) for rV in re.findall(match_number, dirname)][5] for dirname in dirs]
    #print(rVs)
    
    if split is True:
        dirslist = []
        for rV in sorted(set(rVs)):
            newdirs = []
            if   fltpts == 1:
                pattern = 'rV_{:.1f}'.format(rV)
            elif fltpts == 2:
                pattern = 'rV_{:.2f}'.format(rV)
            for dirname in dirs:
                if pattern in dirname:
                    newdirs.append(dirname)
            dirslist.append(newdirs)
        #print(len(set(xs)),len(dirslist))
        return dirslist
    else:
        return [dirname for (rV,dirname) in sorted(zip(rVs,dirs))]


def sort_dirs_by_omega(dirs,split=False):
    """
    Sorts names of directories by 7-th float in this name (it should be parameter of potential by default!).
    
    @param dirs  - list of directories with data (necessarly full with aspect ratio and mesh size).
    @param split - if true returns list of lists where every list contains directories' names with the same interaction parameter
    @return      - list of diretories sorted by interaction parameter strength, 1D if split False, 2D if split True.
    """
    
    match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
    omegas = [[float(omega) for omega in re.findall(match_number, dirname)][6] for dirname in dirs]
    
    if split is True:
        dirslist = []
        for omega in sorted(set(omegas)):
            newdirs = []
            pattern = 'wx{:.3f}'.format(omega)
            for dirname in dirs:
                if pattern in dirname:
                    newdirs.append(dirname)
            dirslist.append(newdirs)
        #print(len(set(xs)),len(dirslist))
        return dirslist
    else:
        return [dirname for (omega,dirname) in sorted(zip(omegas,dirs))]

def sort_dirs_by_dts(dirs,split=False):
    """
    Sorts names of directories by 8-th float in this name (it should be simulation timestep by default!).
    
    @param dirs  - list of directories with data (necessarly full with aspect ratio and mesh size).
    @param split - if true returns list of lists where every list contains directories' names with the same simulation timestep
    @return      - list of diretories sorted by simulation timestep, 1D if split False, 2D if split True.
    """
    
    match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
    dts = [[float(dt) for dt in re.findall(match_number, dirname)][7] for dirname in dirs]
    
    if split is True:
        dirslist = []
        for x in sorted(set(dts)):
            newdirs = []
            pattern = 'dt{:.3f}'.format(x)
            for dirname in dirs:
                if pattern in dirname:
                    newdirs.append(dirname)
            dirslist.append(newdirs)
        #print(len(set(xs)),len(dirslist))
        return dirslist
    else:
        return [dirname for (dt,dirname) in sorted(zip(dts,dirs))]


def sort_dirs_by_Npart(dirs,split=False):
    """
    Sorts names of directories by 9-th float in this name (it should be number of particles by default!).
    
    @param dirs  - list of directories with data (necessarly full with aspect ratio and mesh size).
    @param split - if true returns list of lists where every list contains directories' names with the same interaction parameter
    @return      - list of diretories sorted by interaction parameter strength, 1D if split False, 2D if split True.
    """
    
    match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
    nparts = [[float(npart) for npart in re.findall(match_number, dirname)][8] for dirname in dirs]
    
    if split is True:
        dirslist = []
        for npart in sorted(set(nparts)):
            newdirs = []
            pattern = 'N{:.0f}'.format(npart)
            for dirname in dirs:
                if pattern in dirname:
                    newdirs.append(dirname)
            dirslist.append(newdirs)
        #print(len(set(xs)),len(dirslist))
        return dirslist
    else:
        return [dirname for (npart,dirname) in sorted(zip(nparts,dirs))]



"""   ====================================   XML FILES PROCESSING   ===================================  """

""" 
"""

class gpeXmlRaport:
    """
    This class provides methods for operating on xml file built on GPEXML.xsd scheme.
    Such a file could be used as a nicely formatted raport from simulation.
    """
    
    # CONSTUCTOR OF CLASS
    def __init__(self, filepath, create=True):
        # create or load xml file
        if create is True:
            f = StringIO.StringIO('''<?xml version="1.0" encoding="utf-8"?>
                <?xml-stylesheet type="text/xsl" href="/home/konrad/CProjects/libcugpe/xml_schemes/GPEXML.xsl"?>
                <SIMULATION xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="/home/konrad/CProjects/libcugpe/xml_schemes/GPEXML.xsd">
                    <PARAMETERS>
                    </PARAMETERS>
                    <ENERGIES>
                    </ENERGIES>
                    <CLOUD_PARAMS>
                    </CLOUD_PARAMS>
                    <VORTEX_PARAMS>
                    </VORTEX_PARAMS>
                    <IMAGES>
                    </IMAGES>
                </SIMULATION>
                ''')
            self.doc = etree.parse(f)
            print('# creating xml file',filepath,'\t\t',self.doc)
            print()
        else:
            self.doc = etree.parse(filepath)
            print('# loading xml from file',filepath,'\t\t',self.doc)
            print()
        
        # get root of xml file
        self.root = self.doc.getroot()
        
        self.filepath = filepath
        
        self.parameters    = self.root[0]
        self.energies      = self.root[1]
        self.cloud_params  = self.root[2]
        self.vortex_params = self.root[3]
        self.images        = self.root[4]
    
    
    # ###   GETTING DATA FROM XML RAPORT   #####################################################################
    
    def get_parameter(self, name):
        """
        Get the value of parameter with matching name (assuming the parameters probably will have unique names).
        
        @param name   - name of parameter
        @retutn       - list of values of parameters with given name
        """
        return self.root.xpath("//PARAMETER[@name='{:s}']/text()".format(name))[0]
    
    def get_energy(self, name):
        """
        Get the value of parameter with matching name (assuming the parameters probably will have unique names).
        
        @param name   - name of parameter
        @retutn       - list of values of parameters with given name
        """
        return self.root.xpath("//ENERGY[@name='{:s}']/text()".format(name))[0]
    
    def get_cloud_param(self, name):
        """
        Get the value of parameter with matching name (assuming the parameters probably will have unique names).
        
        @param name   - name of parameter
        @retutn       - list of values of parameters with given name
        """
        return self.root.xpath("//CLOUD_PARAM[@name='{:s}']/text()".format(name))[0]
    
    def get_vortex_param(self, name):
        """
        Get the value of parameter with matching name (assuming the parameters probably will have unique names).
        
        @param name   - name of parameter
        @retutn       - list of values of parameters with given name
        """
        print(name,self.root.xpath("//VORTEX_PARAM[@name='{:s}']/text()".format(name)))
        return self.root.xpath("//VORTEX_PARAM[@name='{:s}']/text()".format(name))[0]
    
    def get_images(self):
        """
        Get the value of parameter with matching name (assuming the parameters probably will have unique names).
        
        @param name   - name of parameter
        @retutn       - list of values of parameters with given name
        """
        return self.root.xpath("//IMAGE/text()")
    
    
    # ###   SAVING DATA FROM XML RAPORT   ######################################################################
    
    def add_parameter(self, name, value):
        parameter = etree.SubElement(self.parameters, "PARAMETER", name=name)
        parameter.text = value
    
    def add_energy(self, name, value):
        energy = etree.SubElement(self.energies, "ENERGY", name=name)
        energy.text = value
    
    def add_cloud_param(self, name, value):
        parameter = etree.SubElement(self.cloud_params, "CLOUD_PARAM", name=name)
        parameter.text = value
    
    def add_vortex_param(self, name, value):
        parameter = etree.SubElement(self.vortex_params, "VORTEX_PARAM", name=name)
        parameter.text = value
    
    def add_image(self, filepath):
        image = etree.SubElement(self.images, "IMAGE")
        image.text = filepath
    
    def save(self):
        with open(self.filepath, 'w') as outFile:
            self.doc.write(outFile, xml_declaration=True, encoding='utf-8')
    
    def print_xml(self):
        print(etree.tostring(self.doc, xml_declaration=True, encoding="utf-8"))




""" ============================================================================================================================================== """

# some other useful functions


def frexp10(x):
    exp = int(math.log10(x))
    return x / 10**exp, exp 
