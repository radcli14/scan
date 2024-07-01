import os
import numpy as np
from sympy import symbols, Matrix, lambdify
import gspread
from google.colab import auth
from oauth2client.client import GoogleCredentials
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

# Function to replace nans and infs with zeros
def fix_bad(n):
  for b in (np.nan, np.inf, -np.inf):
    n = np.where(n == b, 0, n)
  return n

# Get forms of the aero polynomials given varying numbers of coefficients
"""
- $u = \mathbf{n} \cdot \mathbf{f}$ where $\mathbf{n}$ is the normal vector, and $\mathbf{f}$ is the car's forward vector
- $v = \mathbf{n} \cdot \mathbf{v}$ where $\mathbf{v}$ is the car's upward vector
- $w = \mathbf{n} \cdot \mathbf{s}$ where $\mathbf{s}$ is the car's right vector

- $b_0$: The amplitude of $C_p$ at a forward-facing normal $u$, where $u = 1$
- $b_1$: The amplitude of $C_p$ at an aft-facing normal, where $u = -1$
- $b_2$: The slope $\frac{\partial C_p}{\partial u}$ at a forward-facing normal
- $b_3$: The amplitude of $C_p$ at a upward-facing normal, where $v = 1$
- $b_4$: The amplitude of $C_p$ at an downward-facing normal, where $v = -1$
- $b_5$: The amplitude of $C_p$ at a side-facing normal, where $w = \pm 1$
"""
b0, b1, b2, b3, b4, b5 = symbols("b0, b1, b2, b3, b4, b5 ")
B = Matrix([b0, b1, b2, b3, b4, b5])
C = Matrix([
    [1, 1, 1, 0, 0, 0],
    [1, -1, 1, 0, 0, 0],
    [0, 1, 2, 0, 0, 0],
    [1, 0, 0, 1, 1, 0],
    [1, 0, 0, -1, 1, 0],
    [1, 0, 0, 0, 0, 1]
    ])
aFcn3Pars = lambdify(list(B[:3, 0]), list(C[:3, :3].inv() * B[:3, 0]))
aFcn5Pars = lambdify(list(B[:5, 0]), list(C[:5, :5].inv() * B[:5, 0]))
aFcn6Pars = lambdify(list(B), list(C.inv() * B))
# ---

class Scan:
  def __init__(self, fileName="", cellBlock=0.25):
    self.fileName = fileName
    self.cellBlock = cellBlock

  @classmethod
  def fromRoblox(cls, scanData):
    # TODO: create this function so the data can be loaded directly from the raw scan data sent from a Roblox API call
    return cls()

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

  _boundingBox = None

  @property
  def boundingBox(self):
    if self._boundingBox is None:
      boxes = [scan.boundingBox for scan in (self.front, self.back, self.top, self.bottom, self.left, self.right)]
      self._boundingBox = [[np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf]]
      for box in boxes:
        for i in range(3):
          self._boundingBox[i][0] = min(self._boundingBox[i][0], box[i][0])
          self._boundingBox[i][1] = max(self._boundingBox[i][1], box[i][1])

    return self._boundingBox

  @property
  def boundingBoxCenter(self):
    return [0.5 * (m[0] + m[1]) for m in self.boundingBox]

  def aeroOldMethod(self,
      dragWeights=(1.334023e+00, 3.789517e-01, 5.690723e-20, 0.0, 2.949861e+00, 2.949861e+00),
      liftWeights=(2.283034e-18, 3.149287e-22, 2.154557e+00, 0.0, 2.500000e+00, 2.500000e+00)
    ):
    projectedArea = self.front.projectedArea
    aeros = [scan.aeroOldMethod for scan in (self.front, self.back, self.top, self.bottom, self.left, self.right)]
    CdA = sum([aero.CdA*w for aero, w in zip(aeros, dragWeights)])
    ClA = sum([aero.ClA*w for aero, w in zip(aeros, liftWeights)])
    return AeroResult(A=projectedArea, CdA=CdA, ClA=ClA)

  def frontBackAero(self, CpFwd=0, CpAft=0, CpSlopeFwd=0, CpUpward=0, CpDownward=0, CpSide=0):
    """
    Obtains the `AeroResult` given the scan data from the front and back, given the surface normal components
    - $u = \mathbf{n} \cdot \mathbf{f}$ where $\mathbf{n}$ is the normal vector, and $\mathbf{f}$ is the car's forward vector
    - $v = \mathbf{n} \cdot \mathbf{v}$ where $\mathbf{v}$ is the car's upward vector
    - $w = \mathbf{n} \cdot \mathbf{s}$ where $\mathbf{s}$ is the car's right vector

    :param CpFwd: $b_0$: The amplitude of $C_p$ at a forward-facing normal $u$, where $u = 1$
    :param CpAft: $b_1$: The amplitude of $C_p$ at an aft-facing normal, where $u = -1$
    :param CpSlopeFwd: $b_2$: The slope $\frac{\partial C_p}{\partial u}$ at a forward-facing normal
    :param CpUpward: $b_3$: The amplitude of $C_p$ at a upward-facing normal, where $v = 1$
    :param CpDownward: $b_4$: The amplitude of $C_p$ at an downward-facing normal, where $v = -1$
    :param CpSide: $b_5$: The amplitude of $C_p$ at a side-facing normal, where $w = \pm 1$
    :return: an `AeroResult` object
    """
    a = aFcn6Pars(CpFwd, CpAft, CpSlopeFwd, CpUpward, CpDownward, CpSide)
    return self.front.aeroNewMethod(a) + self.back.aeroNewMethod(a)

  def topBottomAero(self, CpFwd=0, CpAft=0, CpSlopeFwd=0, CpUpward=0, CpDownward=0, CpSide=0):
    """
    Obtains the `AeroResult` given the scan data from the front and back, given the surface normal components
    """
    a = aFcn6Pars(CpFwd, CpAft, CpSlopeFwd, CpUpward, CpDownward, CpSide)
    top = self.top.aeroNewMethod(a)
    top.updateArea(self.front.projectedArea)
    return top + self.bottom.aeroNewMethod(a)

  def leftRightAero(self, CpFwd=0, CpAft=0, CpSlopeFwd=0, CpUpward=0, CpDownward=0, CpSide=0):
    """
    Obtains the `AeroResult` given the scan data from the left and right, given the surface normal components
    """
    a = aFcn6Pars(CpFwd, CpAft, CpSlopeFwd, CpUpward, CpDownward, CpSide)
    left = self.left.aeroNewMethod(a)
    left.updateArea(self.front.projectedArea)
    return left + self.right.aeroNewMethod(a)

  def allDirectionAero(self, CpFwd=0, CpAft=0, CpSlopeFwd=0, CpUpward=0, CpDownward=0, CpSide=0,
                       wDragFront=1, wDragBack=1, wDragTop=1, wDragBottom=1, wDragLeft=1, wDragRight=1,
                       wLiftFront=1, wLiftBack=1, wLiftTop=1, wLiftBottom=1, wLiftLeft=1, wLiftRight=1):
    """
    Obtains the `AeroResult` given the scan data from the left and right, given the surface normal components
    """
    a = aFcn6Pars(CpFwd, CpAft, CpSlopeFwd, CpUpward, CpDownward, CpSide)
    front = self.front.aeroNewMethod(a)
    back = self.back.aeroNewMethod(a)
    top = self.top.aeroNewMethod(a)
    bottom = self.bottom.aeroNewMethod(a)
    left = self.left.aeroNewMethod(a)
    right = self.right.aeroNewMethod(a)

    # Form into lists to simplify generating the sums
    aeros = [front, back, top, bottom, left, right]
    dragWeights = [wDragFront, wDragBack, wDragTop, wDragBottom, wDragLeft, wDragRight]
    liftWeights = [wLiftFront, wLiftBack, wLiftTop, wLiftBottom, wLiftLeft, wLiftRight]

    # Sum up the drag and lift coefficients times areas
    CdA = sum([w * aero.CdA for w, aero in zip(dragWeights, aeros)])
    ClA = sum([w * aero.ClA for w, aero in zip(liftWeights, aeros)])

    # Sum up the aerodynamic center, and calculate aero balance (percentage back from forward-most point to back-most)
    aeroCenterFwd = sum([w * aero.ClA * aero.aeroCenter[0] for w, aero in zip(liftWeights, aeros)]) / ClA
    aeroCenterRight = sum([w * aero.CdA * aero.aeroCenter[1] for w, aero in zip(dragWeights, aeros)]) / CdA
    aeroCenterUp = sum([w * aero.CdA * aero.aeroCenter[2] for w, aero in zip(dragWeights, aeros)]) / CdA

    # Set the aerodynamic center to be relative to a point at the front of the car, and the road surface
    aeroCenter = [aeroCenterFwd - self.boundingBox[0][1], aeroCenterRight-self.boundingBoxCenter[1], aeroCenterUp-self.boundingBox[2][0]]
    carLength = self.boundingBox[0][0] - self.boundingBox[0][1]
    aeroBalance = aeroCenter[0] / carLength

    return AeroResult(A = front.A, CdA = CdA, ClA = ClA, aeroCenter=aeroCenter, aeroBalance=aeroBalance)

class AeroResult:
  A = None
  Cd = None
  Cl = None
  CdA = None
  ClA = None
  aeroCenter = None

  def __init__(self, A=None, Cd=None, Cl=None, CdA=None, ClA=None, aeroCenter=None, aeroBalance=None):
    self.A = A
    self.Cd = Cd if Cd is not None else CdA / A
    self.Cl = Cl if Cl is not None else ClA / A
    self.CdA = CdA if CdA is not None else Cd * A
    self.ClA = ClA if ClA is not None else Cl * A
    self.aeroCenter = aeroCenter
    self.aeroBalance = aeroBalance

  def __repr__(self):
    return f'AeroResult(A={self.A}, Cd={self.Cd}, Cl={self.Cl}, aeroCenter={self.aeroCenter}, aeroBalance={self.aeroBalance})'

  def __add__(self, other):
    ClA = self.ClA + other.ClA
    if self.aeroCenter is not None and other.aeroCenter is not None:
      aeroCenter = (self.ClA * self.aeroCenter + other.ClA * other.aeroCenter) / ClA
    else:
      aeroCenter = None
    return AeroResult(A=self.A, CdA=self.CdA + other.CdA, ClA=ClA, aeroCenter=aeroCenter)

  def __mul__(self, other):
    return AeroResult(A=self.A, CdA=self.CdA * other, ClA=self.ClA * other)

  def __rmul__(self, other):
    return AeroResult(A=self.A, CdA=self.CdA * other, ClA=self.ClA * other)

  def updateArea(self, newArea):
    self.A = newArea
    self.Cd = self.CdA / self.A
    self.Cl = self.ClA / self.A


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
      self._data = fix_bad(self._data)
    return self._data

  @property
  def count(self):
    return self.data.shape[0]
  
  @property
  def hits(self):
    return self.data[:, :3] if self._hitReferencePoint is None else self.data[:, :3] - self._hitReferencePoint

  _hitReferencePoint = None

  @property
  def hitReferencePoint(self):
    return self._hitReferencePoint if self._hitReferencePoint is not None else np.array([0, 0, 0])

  @hitReferencePoint.setter
  def hitReferencePoint(self, value):
    self._hitReferencePoint = value

  @property
  def hitForward(self):
    x = self.hits @ self.forward.transpose()
    return fix_bad(x)

  @property
  def hitUp(self):
    y = self.hits @ self.up.transpose()
    return fix_bad(y)

  @property
  def hitRight(self):
    z = self.hits @ self.right.transpose()
    return fix_bad(z)

  @property
  def normals(self):
    return self.data[:, 3:]

  @property
  def normalForward(self):
    u = self.normals @ self.forward.transpose()
    return fix_bad(u)

  @property
  def normalUp(self):
    v = self.normals @ self.up.transpose()
    return fix_bad(v)

  @property
  def normalRight(self):
    w = self.normals @ self.right.transpose()
    return fix_bad(w)

  @property
  def normalRay(self):
    ray = self.normals @ self.ray.transpose()
    return fix_bad(ray)

  @property
  def projectedAreaPerPixel(self):
    return self.cellBlock * self.cellBlock

  @property
  def projectedArea(self):
    return self.count * self.projectedAreaPerPixel

  @property
  def inclinedSurfaceArea(self):
    inclinedSurfaceArea = self.projectedAreaPerPixel * np.abs(self.normalRay) ** (-1)
    return fix_bad(inclinedSurfaceArea)

  @property
  def boundingBox(self):
    return [(vec.min(), vec.max()) for vec in (self.hitForward, self.hitRight, self.hitUp)]

  @property
  def boundingBoxCenter(self):
    return [0.5 * (m[0] + m[1]) for m in self.boundingBox]

  @property
  def aeroOldMethod(self):
    r_dot_n = -self.normalForward  # Negative because this is the wind vector
    Cp = 1 - (1 - np.abs(r_dot_n))**2
    inclinedSurfaceArea = self.projectedAreaPerPixel * np.abs(r_dot_n)
    CdA = Cp * inclinedSurfaceArea * (-r_dot_n)
    ClA = Cp * inclinedSurfaceArea * (-self.normalUp)
    return AeroResult(A=self.projectedArea, CdA=CdA.sum(), ClA=ClA.sum())

  def fcnCpSixTerms(self, a):
    u = self.normalForward
    v = self.normalUp
    w = self.normalRight
    x = [np.ones(self.count), u, u**2, v, v**2, w**2]
    return sum([xk*ak for xk, ak in zip(x, a)])

  def aeroNewMethod(self, a):
    CpA = self.fcnCpSixTerms(a) * self.inclinedSurfaceArea
    CdA = CpA * (self.normalForward)
    ClA = CpA * (-self.normalUp)
    CdAsum = np.nansum(CdA)
    ClAsum = np.nansum(ClA)
    aeroCenterFwd = np.nansum(ClA * self.hitForward) / ClAsum if abs(ClAsum) > 0 else self.boundingBoxCenter[0]
    aeroCenterRight = np.nansum(CdA * self.hitRight) / CdAsum if abs(CdAsum) > 0 else self.boundingBoxCenter[1]
    aeroCenterUp = np.nansum(CdA * self.hitUp) / CdAsum if abs(CdAsum) > 0 else self.boundingBoxCenter[2]
    return AeroResult(A=self.projectedArea, CdA=CdAsum, ClA=ClAsum, aeroCenter=np.array([aeroCenterFwd, aeroCenterRight, aeroCenterUp]))