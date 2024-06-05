import os
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
    return self.data.worksheet("Front")

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
