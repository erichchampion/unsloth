<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:dita-ot="http://dita-ot.sourceforge.net/ns/201007/dita-ot"
                xmlns:ditaarch="http://dita.oasis-open.org/architecture/2005/"
                xmlns:e="pdf-theme"
                xmlns:fo="http://www.w3.org/1999/XSL/Format"
                xmlns:fox="http://xmlgraphics.apache.org/fop/extensions"
                xmlns:opentopic-func="http://www.idiominc.com/opentopic/exsl/function"
                xmlns:opentopic="http://www.idiominc.com/opentopic"
                xmlns:xs="http://www.w3.org/2001/XMLSchema"
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                version="2.0"
                exclude-result-prefixes="xs e dita-ot ditaarch opentopic opentopic-func">

  <!-- Programming domain elements -->

  <!-- Code block - use monospace font -->
  <xsl:attribute-set name="codeblock">
    <xsl:attribute name="font-family" select="$monospace-font"/>
  </xsl:attribute-set>

  <!-- Inline code - use monospace font -->
  <xsl:attribute-set name="codeph">
    <xsl:attribute name="font-family" select="$monospace-font"/>
  </xsl:attribute-set>

  <!-- API name -->
  <xsl:attribute-set name="apiname">
    <xsl:attribute name="font-family" select="$monospace-font"/>
  </xsl:attribute-set>

  <!-- Other programming domain elements -->
  <xsl:attribute-set name="option">
    <xsl:attribute name="font-family" select="$monospace-font"/>
  </xsl:attribute-set>

  <xsl:attribute-set name="parmname">
    <xsl:attribute name="font-family" select="$monospace-font"/>
  </xsl:attribute-set>

  <xsl:attribute-set name="parml"/>
  <xsl:attribute-set name="plentry"/>
  <xsl:attribute-set name="pt"/>
  <xsl:attribute-set name="pd"/>
  <xsl:attribute-set name="synph"/>
  <xsl:attribute-set name="syntaxdiagram"/>
  <xsl:attribute-set name="groupseq"/>
  <xsl:attribute-set name="groupchoice"/>
  <xsl:attribute-set name="groupcomp"/>
  <xsl:attribute-set name="fragment"/>
  <xsl:attribute-set name="fragref"/>
  <xsl:attribute-set name="synblk"/>
  <xsl:attribute-set name="synnote"/>
  <xsl:attribute-set name="synnoteref"/>

  <xsl:attribute-set name="kwd">
    <xsl:attribute name="font-family" select="$monospace-font"/>
  </xsl:attribute-set>

  <xsl:attribute-set name="var">
    <xsl:attribute name="font-family" select="$monospace-font"/>
  </xsl:attribute-set>

  <xsl:attribute-set name="oper"/>
  <xsl:attribute-set name="delim"/>
  <xsl:attribute-set name="sep"/>
  <xsl:attribute-set name="repsep"/>

</xsl:stylesheet>
