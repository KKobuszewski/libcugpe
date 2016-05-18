import lxml.etree as etree

tree = etree.parse('./job.xml')  
root = tree.getroot()                    

print(root)

for parameters in root:
	print(parameters)
	print(parameters.attrib)
	for parameter in parameters:
		print(parameter.attrib)
	print()


new_feed = etree.Element('{http://www.w3.org/2001/XMLSchema-instance}feed',
                 attrib={'{http://xml.comp-phys.org/2002/4/QMCXML.xsd}lang': 'en'})
print(etree.tostring(new_feed)) 
