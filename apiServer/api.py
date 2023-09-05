import requests
import time


def getRainFromApi():
  now = time
  nowstr = now.strftime('%Y%m%d%H00')
  URL = 'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php?obs=RN&tm={}&stn=0&help=1&authKey=Pi8YfpSBTPivGH6Ugaz4Kg'.format(nowstr)
  response = requests.get(URL)
  result = response.text
  splited = result.splitlines()
  return float(splited[80].split()[15])



def getRainFromDummy(i):
  dummyRainyDay = [0.0, 0.0, 13.3, 20.1, 8.4, 30.10, 60.0, 0,   103.1, 109.1,401.1,511.2,533.8, 409.4, 308.1, 123.1,   30.2,   0,   0,   0  ]
  return dummyRainyDay[(i+2)%20]