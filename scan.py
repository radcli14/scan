import os
import numpy as np
import gspread
from google.colab import auth
from oauth2client.client import GoogleCredentials
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

class Scan:
  def __init__(self, fileName="", cellBlock=0.25):
    self.fileName = fileName
    self.cellBlock = cellBlock

  def __repr__(self):
    return f'Scan(fileName="{self.fileName}")'

  @property
  def carName(self):
    _, carName = os.path.split(self.carDirectory)
    return carName

  @property
  def CdFromName(self):
    return self.propFromName("Cd")

  @property
  def ClFromName(self):
    return self.propFromName("Cl")

  def propFromName(self, prop):
    splitName = self.carName.split("_")
    for s in splitName:
      if prop in s:
        return float(s.replace(prop, ""))
    return None
  
  @property
  def scanDirectory(self):
    scanDirectory, _ = os.path.split(self.carDirectory)
    return scanDirectory

  @property
  def carDirectory(self):
    carDirectory, _ = os.path.split(self.fileName)
    return carDirectory

  @property
  def scanFile(self):
    _, scanFile = os.path.split(self.fileName)
    return scanFile

  @property
  def data(self):
    return gc.open(self.scanFile.replace(".gsheet", ""))

  @property
  def sheets(self):
    return self.data.worksheets()

  @property
  def front(self):
    return SingleDirectionScan(self.data.worksheet("Front"))

  @property
  def back(self):
    return SingleDirectionScan(self.data.worksheet("Back"))

  @property
  def top(self):
    return SingleDirectionScan(self.data.worksheet("Top"))

  @property
  def bottom(self):
    return SingleDirectionScan(self.data.worksheet("Bottom"))

  @property
  def left(self):
    return SingleDirectionScan(self.data.worksheet("Left"))

  @property
  def right(self):
    return SingleDirectionScan(self.data.worksheet("Right"))

  @property
  def aeroOldMethod(self, 
                    dragweights=[1.334023e+00,	3.789517e-01,	5.690723e-20, 0.0,	2.949861e+00,	2.949861e+00],
                    liftWeights=[2.283034e-18, 3.149287e-22, 2.154557e+00, 0.0, 2.500000e+00, 2.500000e+00]):
    projectedArea = self.front.projectedArea
                     """
    CdAfront, ClAfront = self.front.aeroOldMethod
    CdAback, ClAback = self.back.aeroOldMethod
    CdAtop, ClAtop = self.top.aeroOldMethod
    CdAbottom, ClAbottom = self.bottom.aeroOldMethod               
    CdAleft, ClAleft = self.left.aeroOldMethod
    CdAright, ClAright = self.right.aeroOldMethod
    CdAs = [CdAfront, CdAback, CdAtop, CdAbottom, CdAleft, CdAright]
    ClAs = [ClAfront, ClAback, ClAtop, ClAbottom, ClAleft, ClAright]  
    """
    aeros = [scan.oldAeroMethod for scan in (self.front, self.back, self.top, self.bottom, self.left, self.right)]
    CdA = sum([aero[0]*w for aero, w in zip(aeros, dragweights)]) # dragweights[0]*CdAfront + dragweights[1]*CdAback + dragweights[2]*CdAtop + dragweights[3]*CdAbottom + dragweights[4]*CdAleft + dragweights[5]*CdAright
    ClA = sum([aero[1]*w for aero, w in zip(aeros, liftweights)]) # liftWeights[0]*ClAfront + liftWeights[1]*ClAback + liftWeights[2]*ClAtop + liftweights[3]*ClAbottom liftWeights[4]*ClAleft + liftWeights[5]*ClAright
    Cd = CdA / projectedArea
    Cl = ClA / projectedArea
    return Cd, Cl, projectedArea, CdA, ClA


class SingleDirectionScan:
  def __init__(self, worksheet, cellBlock=0.25):
    self.worksheet = worksheet
    self.cellBlock = cellBlock
    self._values = None
    self._data = None
  
  @property
  def values(self):
    if self._values is None:
      self._values = self.worksheet.get_all_values()
      self._values = np.where(self._values=='', '0', self._values)
    return self._values

  @property
  def name(self):
    return self.values[0][0]

  @property
  def carClass(self):
    return self.values[1][0]

  @property
  def direction(self):
    return self.values[2][0]

  @property
  def forward(self):
    return np.array(self.values[3][1:4]).astype("float")

  @property
  def up(self):
    return np.array([0.0, 1.0, 0.0])

  @property
  def right(self):
    return np.cross(self.forward, self.up)

  @property
  def ray(self):
    return np.array(self.values[4][1:4]).astype("float")

  @property
  def header(self):
    return self.values[5]

  @property
  def data(self):
    if self._data is None:
      self._data = np.array(self.values[6:]).astype("float")
    return self._data

  @property
  def count(self):
    return self.data.shape[0]
  
  @property
  def hits(self):
    return self.data[:, :3]

  @property
  def normals(self):
    return self.data[:, 3:]

  @property
  def normalForward(self):
    return self.normals @ self.forward.transpose()

  @property
  def normalUp(self):
    return self.normals @ self.up.transpose()

  @property
  def normalRight(self):
    return self.normals @ self.right.transpose()

  @property
  def projectedAreaPerPixel(self):
    return self.cellBlock * self.cellBlock

  @property
  def projectedArea(self):
    return self.count * self.projectedAreaPerPixel
  
  @property
  def aeroOldMethod(self):
    r_dot_n = -self.normalForward  # Negative because this is the wind vector
    Cp = 1 - (1 - np.abs(r_dot_n))**2
    inclinedSurfaceArea = self.projectedAreaPerPixel * np.abs(r_dot_n)
    CdA = Cp * inclinedSurfaceArea * (-r_dot_n)
    ClA = Cp * inclinedSurfaceArea * (-self.normalUp)
    return CdA.sum(), ClA.sum()
