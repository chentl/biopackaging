SELECT toFloat32(LAP) / 10.24              AS LAP,
       toFloat32(MMT) / 10.24              AS MMT,
       toFloat32(CMC) / 10.24              AS CMC,
       toFloat32(CNF) / 10.24              AS CNF,
       toFloat32(SLK) / 10.24              AS SLK,
       toFloat32(AGR) / 10.24              AS AGR,
       toFloat32(ALG) / 10.24              AS ALG,
       toFloat32(CAR) / 10.24              AS CAR,
       toFloat32(CHS) / 10.24              AS CHS,
       toFloat32(PEC) / 10.24              AS PEC,
       toFloat32(PUL) / 10.24              AS PUL,
       toFloat32(STA) / 10.24              AS STA,
       toFloat32(GEL) / 10.24              AS GEL,
       toFloat32(GLU) / 10.24              AS GLU,
       toFloat32(ZIN) / 10.24              AS ZIN,
       toFloat32(GLY) / 10.24              AS GLY,
       toFloat32(FFA) / 10.24              AS FFA,
       toFloat32(LAC) / 10.24              AS LAC,
       toFloat32(LEV) / 10.24              AS LEV,
       toFloat32(PHA) / 10.24              AS PHA,
       toFloat32(SRB) / 10.24              AS SRB,
       toFloat32(SUA) / 10.24              AS SUA,
       toFloat32(XYL) / 10.24              AS XYL,
       toFloat32(TensileStrength) / 32     AS TensileStrength,
       toFloat32(TensileSED) / 32          AS TensileSED,
       toFloat32(Uncertainty) / 10.24      AS Uncertainty
FROM samples
WHERE TensileSED >= 10
AND TensileStrength >= 50
AND Uncertainty <= 30
ORDER BY CompHash
LIMIT 10;