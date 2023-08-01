# 필요한 패키지 호출
import requests, xmltodict, json
import pandas as pd

# 신청한 데이터의 키
key='****'
# 주소
url = 'https://apis.data.go.kr/1741000/DisasterMsg3/getDisasterMsg1List?serviceKey={}&pageNo=100&numOfRows=1000&type=xml'.format(key)

# xml형태의 api 데이터 호출하기
content = requests.get(url).content

# xml 형태의 데이터를 dictionary 형태로 바꾸어주기 : 현재 content는 xml 형태이므로 dict라는 이름의 dictionary형태로 변환
dict = xmltodict.parse(content)

dict.keys()
dict.values()

# 파이썬에서 다루기 용이하도록 json형태(키-값 쌍으로 이루어진 데이터 형식)로 바꾸어주기
jsonString = json.dumps(dict['DisasterMsg'], ensure_ascii=False)
jsonObj = json.loads(jsonString)
jsonObj.keys()
jsonObj.values()
jsonObj['head']
jsonObj['row']
# jsonObj의 행 불러와보기
for row in jsonObj['row']:
    print(row)

# 불러온 json형태의 데이터를 일반적인 데이터프레임 형태로 변환하기
df = pd.DataFrame(jsonObj['row'])

# 데이터 프레임을 csv로 추출하기
df.to_csv("C:/Users/pheon/Desktop/긴급재난문자 데이터 크롤링/test.csv", encoding='cp949')
