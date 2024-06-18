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

  _data = None

  @property
  def data(self):
    if self._data is None:
      self._data = gc.open(self.scanFile.replace(".gsheet", ""))
    return self._data

  _sheets = None

  @property
  def sheets(self):
    if self._sheets is None:
      self._sheets = self.data.worksheets()
    return self._sheets

  _front = None

  @property
  def front(self):
    if self._front is None:
      self._front = SingleDirectionScan(self.data.worksheet("Front"))
    return self._front

  _back = None

  @property
  def back(self):
    if self._back is None:
      self._back = SingleDirectionScan(self.data.worksheet("Back"))
    return self._back

  _top = None

  @property
  def top(self):
    if self._top is None:
      self._top = SingleDirectionScan(self.data.worksheet("Top"))
    return self._top

  _bottom = None

  @property
  def bottom(self):
    if self._bottom is None:
      self._bottom = SingleDirectionScan(self.data.worksheet("Bottom"))
    return self._bottom

  _left = None

  @property
  def left(self):
    if self._left is None:
      self._left = SingleDirectionScan(self.data.worksheet("Left"))
    return self._left

  _right = None

  @property
  def right(self):
    if self._right is None:
      self._right = SingleDirectionScan(self.data.worksheet("Right"))
    return self._right

  def aeroOldMethod(self,
      dragWeights=(1.334023e+00, 3.789517e-01, 5.690723e-20, 0.0, 2.949861e+00, 2.949861e+00),
      liftWeights=(2.283034e-18, 3.149287e-22, 2.154557e+00, 0.0, 2.500000e+00, 2.500000e+00)
    ):
    projectedArea = self.front.projectedArea
    aeros = [scan.aeroOldMethod for scan in (self.front, self.back, self.top, self.bottom, self.left, self.right)]
    CdA = sum([aero[0]*w for aero, w in zip(aeros, dragWeights)]) # dragweights[0]*CdAfront + dragweights[1]*CdAback + dragweights[2]*CdAtop + dragweights[3]*CdAbottom + dragweights[4]*CdAleft + dragweights[5]*CdAright
    ClA = sum([aero[1]*w for aero, w in zip(aeros, liftWeights)]) # liftWeights[0]*ClAfront + liftWeights[1]*ClAback + liftWeights[2]*ClAtop + liftweights[3]*ClAbottom liftWeights[4]*ClAleft + liftWeights[5]*ClAright
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
      data = np.array(self.values[6:])
      data = np.where(data == '', 0, data)
      data = np.where(data == '-', 0, data)
      self._data = data.astype("float")
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
  def normalRay(self):
    return self.normals @ self.ray.transpose()

  @property
  def projectedAreaPerPixel(self):
    return self.cellBlock * self.cellBlock

  @property
  def projectedArea(self):
    return self.count * self.projectedAreaPerPixel

  @property
  def inclinedSurfaceArea(self):
    inclinedSurfaceArea = self.projectedAreaPerPixel * np.abs(self.normalRay) ** (-1)
    return np.where(inclinedSurfaceArea == np.nan, 0, inclinedSurfaceArea)
  
  @property
  def aeroOldMethod(self):
    r_dot_n = -self.normalForward  # Negative because this is the wind vector
    Cp = 1 - (1 - np.abs(r_dot_n))**2
    inclinedSurfaceArea = self.projectedAreaPerPixel * np.abs(r_dot_n)
    CdA = Cp * inclinedSurfaceArea * (-r_dot_n)
    ClA = Cp * inclinedSurfaceArea * (-self.normalUp)
    return CdA.sum(), ClA.sum()

  def fcnCpSixTerms(self, a):
    u = self.normalForward
    v = self.normalUp
    w = self.normalRight
    x = [np.ones(self.count), u, u**2, v, v**2, w**2]
    return sum([xk*ak for xk, ak in zip(x, a)])

  def aeroNewMethod(self, a):
    CpA = self.fcnCpSixTerms(a) * self.inclinedSurfaceArea
    CdA = CpA * self.normalForward
    ClA = CpA * (-self.normalUp)
    return CdA.sum(), ClA.sum(), self.projectedArea