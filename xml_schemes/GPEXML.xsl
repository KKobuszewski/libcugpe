<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
	<xsl:template match="/">
		<html>
			<head/>
			<body>
				<xsl:for-each select="SIMULATION">
						<br/>
						<H1>GPE Simulation</H1>
            <H2>Parameters</H2>
						<table border="1">
							<thead><tr><td>
											<B>Parameter</B>
										</td>
										<td>
											<B>Value</B>
										</td>
									</tr>
								</thead>
					  <tbody>
						<xsl:for-each select="PARAMETERS">
						<xsl:for-each select="PARAMETER">
							<tr>
								<td>
									<xsl:value-of select="@name"/>
								</td>
								<td>
									<xsl:apply-templates/>
								</td>
							</tr>
						</xsl:for-each>
						</xsl:for-each>
						</tbody></table>
            <H2>Energies</H2>
						<table border="1">
							<thead><tr><td>
											<B>Energy</B>
										</td>
										<td>
											<B>Value</B>
										</td>
									</tr>
								</thead>
					  <tbody>
						<xsl:for-each select="ENERGIES">
						<xsl:for-each select="ENERGY">
							<tr>
								<td>
									<xsl:value-of select="@name"/>
								</td>
								<td>
									<xsl:apply-templates/>
								</td>
							</tr>
						</xsl:for-each>
						</xsl:for-each>
						</tbody></table>
            <H2>Cloud parameters</H2>
						<table border="1">
							<thead><tr><td>
											<B>Parameter</B>
										</td>
										<td>
											<B>Value</B>
										</td>
									</tr>
								</thead>
					  <tbody>
						<xsl:for-each select="CLOUD_PARAMS">
						<xsl:for-each select="CLOUD_PARAM">
							<tr>
								<td>
									<xsl:value-of select="@name"/>
								</td>
								<td>
									<xsl:apply-templates/>
								</td>
							</tr>
						</xsl:for-each>
						</xsl:for-each>
						</tbody></table>
            <H2>Vortex parameters</H2>
						<table border="1">
							<thead><tr><td>
											<B>Parameter</B>
										</td>
										<td>
											<B>Value</B>
										</td>
									</tr>
								</thead>
					  <tbody>
						<xsl:for-each select="VORTEX_PARAMS">
						<xsl:for-each select="VORTEX_PARAM">
							<tr>
								<td>
									<xsl:value-of select="@name"/>
								</td>
								<td>
									<xsl:apply-templates/>
								</td>
							</tr>
						</xsl:for-each>
						</xsl:for-each>
						</tbody></table>
            <H2>Plots</H2>
                                         <xsl:for-each select="IMAGES">
                                         <xsl:for-each select="IMAGE">
                                         <object type="application/pdf" width="700px" height="550px">
                                         <xsl:attribute name="data">
                                                <!--xsl:text>./</xsl:text-->
                                                <xsl:apply-templates/>
                                         </xsl:attribute>
                                         <p>It appears you don't have a PDF plugin for this browser. Please contact the help desk.</p>
                                         </object>
                                         </xsl:for-each>
                                         </xsl:for-each>
                                </xsl:for-each>
                        </body>
		</html>
	</xsl:template>
</xsl:stylesheet>
