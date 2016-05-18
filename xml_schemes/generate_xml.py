from __future__ import print_function
from lxml import etree
import StringIO

import sys.path
sys.path.append('/home/konrad/CProjects/libcugpe/python')
from gpe_datautils import gpeXmlRaport

# main
if __name__ == '__main__':
    xml_raport = gpeXmlRaport('./output.xml')
    
    # Add data to XML
    xml_raport.add_parameter('lattice','512x128x128')
    xml_raport.add_parameter('a_scat','1.0')
    xml_raport.add_parameter('n0','350')
    
    xml_raport.add_energy('E0','0.0')
    
    xml_raport.add_cloud_param('Rx','10.0')
    xml_raport.add_cloud_param('Ry','2.0')
    
    xml_raport.add_vortex_param('Tv/Tx','5.0')
    
    xml_raport.add_image('./image.pdf')
    
    xml_raport.print_xml()
    
    # Save to XML file
    xml_raport.save()
