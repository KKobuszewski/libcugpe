# -*- coding: utf-8 -*-
from __future__ import print_function

from lxml import etree
import sys

# xml raport path
sys.path.append('/home/konrad/CProjects/libcugpe/python')
from gpe_datautils import gpeXmlRaport

"""
for parameters in root:
	print(parameters)
	print(parameters.attrib)
	for parameter in parameters:
		print(parameter.attrib)
	print()


new_feed = etree.Element('{http://www.w3.org/2001/XMLSchema-instance}feed',
                 attrib={'{http://xml.comp-phys.org/2002/4/QMCXML.xsd}lang': 'en'})
print(etree.tostring(new_feed)) 
"""


# main
if __name__ == '__main__':
    xml_raport = gpeXmlRaport('./output.xml', create=False)
    
    xml_raport.print_xml()
    print(xml_raport.get_parameter('lattice'))
    print(xml_raport.get_energy('E0'))
    print(xml_raport.get_cloud_param('Rx'))
    print(xml_raport.get_vortex_param('Tv/Tx'))
    print(xml_raport.get_images())
    pass
