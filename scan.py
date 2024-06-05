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
  def __init__(self, fileName=""):
    self.fileName = fileName

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
    return self.data.worksheet("Back")

  @property
  def top(self):
    return self.data.worksheet("Top")

  @property
  def bottom(self):
    return self.data.worksheet("Bottom")

  @property
  def left(self):
    return self.data.worksheet("Left")

  @property
  def right(self):
    return self.data.worksheet("Right")


class SingleDirectionScan:
  def __init__(self, worksheet):
    self.worksheet = worksheet
    self._values = None
    self._data = None
  
  @property
  def values(self):
    if not self._values:
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
    return np.array(self.values[3][1:4])

  @property
  def ray(self):
    return np.array(self.values[4][1:4])

  @property
  def header(self):
    return self.values[5]

  @property
  def data(self):
    if not self._data:
      self._data = np.array(self.values[6:])
    return self._data

  @property
  def hits(self):
    return self.data[:, :3]

  @property
  def normals(self):
    return self.data[:, 3:]
