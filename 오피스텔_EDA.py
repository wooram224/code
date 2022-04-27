import pandas as pd

officetel = pd.read_excel("C:/Users/user/Desktop/업무/data/오피스텔_전유_주민망_가구원_수.xlsx")

# 개별 호 코드 생성
officetel['도로명'] = officetel['NA_ROAD_CD'].astype('str')
officetel['본번'] = officetel['NA_MAIN_BUN'].astype('str')
officetel['부번'] = officetel['NA_SUB_BUN'].astype('str')
officetel['호수'] = officetel['HONM'].astype('str')

officetel['도로명_호'] = officetel['도로명']+ officetel['본번'] + officetel['부번'] +officetel['호수']
officetel.nunique()

# 면적 200제곱미터 이하 사용
officetel_2 = officetel[officetel["AREA"]<=200]
# 표본수 269941 맞는지 확인

# 나이
officetel_2['now'] = 2022
officetel_2['age'] = officetel_2['now'] - officetel_2['BIRTH_Y'] + 1

del(officetel)

# 면적구분 열 생성 : 사이값 이중조건
officetel_2['면적구분']= 0
officetel_2['면적구분'][(officetel_2['AREA']<=14)] = "0~14"
officetel_2['면적구분'][(officetel_2['AREA']>14) & (officetel_2['AREA']<=20)] = "14~20"
officetel_2['면적구분'][(officetel_2['AREA']>20) & (officetel_2['AREA']<=30)] = "20~30"
officetel_2['면적구분'][(officetel_2['AREA']>30) & (officetel_2['AREA']<=40)] = "30~40"
officetel_2['면적구분'][(officetel_2['AREA']>40) & (officetel_2['AREA']<=60)] = "40~60"
officetel_2['면적구분'][(officetel_2['AREA']>60) & (officetel_2['AREA']<=85)] = "60~85"
officetel_2['면적구분'][(officetel_2['AREA']>85) & (officetel_2['AREA']<=135)] = "85~135"
officetel_2['면적구분'][(officetel_2['AREA']>135) & (officetel_2['AREA']<=200)] = "135~200"
officetel_2['면적구분'].value_counts()

# 연령대 열 생성하기
officetel_2['연령대'] = "10대미만"
officetel_2['연령대'][(officetel_2['age']>=10) & (officetel_2['age']<20)] = "10대"
officetel_2['연령대'][(officetel_2['age']>=20) & (officetel_2['age']<30)] = "20대"
officetel_2['연령대'][(officetel_2['age']>=30) & (officetel_2['age']<40)] = "30대"
officetel_2['연령대'][(officetel_2['age']>=40) & (officetel_2['age']<50)] = "40대"
officetel_2['연령대'][(officetel_2['age']>=50) & (officetel_2['age']<60)] = "50대"
officetel_2['연령대'][(officetel_2['age']>=60)] = "60대 이상"
officetel_2['연령대'].value_counts()

officetel_2.to_csv("C:/Users/pheon/Desktop/주택정책지원센터/220418 오피스텔/오피스텔_핸들링.csv", encoding='cp949')

del(officetel_2)
import pandas as pd
# 핸들링 데이터
officetel = pd.read_csv("C:/Users/user/Desktop/업무/data//오피스텔_핸들링.csv", encoding='cp949')

# 표1. 오피스텔 면적별 통계량
pop = officetel['도로명_호'].groupby(officetel['면적구분']).count()
pop = pd.DataFrame(pop)
# 열이름 변경하는 2가지 방법
pop.columns = ['거주인구']
# hosu.rename(columns={'도로명_호':'거주인구'})
# hosu = officetel['도로명_호'].drop_duplicates() 특정열에 대한 중복값을 제거한 후, 해당 열만 남음
hosu = officetel.drop_duplicates(['도로명_호'])
hosu2 = hosu['도로명_호'].groupby(hosu['면적구분']).count()
hosu2 = pd.DataFrame(hosu2)
hosu2.columns = ['호수']
age= officetel['age'].groupby(officetel['면적구분']).mean()

officetel['child'] = 0
officetel['child'][officetel['age'] <= 7] = 1

child = officetel['child'].groupby(officetel['면적구분']).sum()
child = pd.DataFrame(child)
child.columns = ['미취학아동수']
# 인덱스를 기준으로 데이터프레임 합치기
merge = pd.merge(child, hosu2, left_index=True, right_index=True)

# 그래프 겹쳐그리기
# anaconda3 환경에서 plot이 렉걸리는 문제에 대한 백엔드 설정
import matplotlib
matplotlib.use('Qt5Agg')

# 그래프 한글 깨짐 방지
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# 1. index별로 데이터 재구성
graph = pd.merge(pop, hosu2, left_index=True, right_index=True)
# 행순서 변경
graph = graph.reindex(index=['0~14', '14~20', '20~30', '30~40', '40~60', '60~85','85~135', '135~200'])

# graph.index.name='면적(㎡)'
graph['면적(㎡)'] = graph.index
graph.head()
x = graph['면적(㎡)']
y1 = graph['거주인구']
y2 = graph['호수']
# 보조축 사용안하기
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots()
ax1.plot(x, y1, '-s', color='green', markersize=4, linewidth=4, alpha=0.7, label='거주인구수')
ax1.bar(x, y2, color='orange', label='호수', alpha=0.7, width=0.7)
ax1.set_ylim(0, 100000)
ax1.set_xlabel('면적(㎡)')
ax1.legend(loc='upper right', bbox_to_anchor=(0.98,0.99))
plt.text(1,96000, "40㎡ 이하")
plt.hlines(95000, -0.5, 3.5, color='blue', linestyle='--', linewidth=1)
# ax1.annotate("", xy=(-0.5, 95000), xytext=(1, 96000), arrowprops=dict(arrowstyle="->")) 한쪽 화살표
plt.vlines(-0.5, 0, 97000, color='blue', linestyle='solid', linewidth=1)
plt.vlines(3.5, 0, 97000, color='blue', linestyle='solid', linewidth=1)
plt.show()

# 그래프2 : 면적별로 거주인구대비 미취학아동수 데이터
merge2 = pd.merge(graph, merge, left_index=True, right_index=True)
# 중복열 삭제
merge2 = merge2.drop(['호수_x'], axis=1)

merge2 = merge2.rename(columns={'호수_y':'호수'})

merge2['미취학아동_전체인구대비'] = merge2['미취학아동수']/merge2['거주인구']

x = merge2['면적(㎡)']
y1 = merge2['호수']
y2 = merge2['미취학아동_전체인구대비']

# 그래프 겹쳐그리기 ( 막대그래프와 꺾은선 그래프) : 보조축 사용
fig, ax1 = plt.subplots()

ax1.bar(x, y1, color='orange', label='호수', alpha=0.7, width=0.7)
ax1.set_ylim(0, 90000)
ax1.set_xlabel('면적(㎡)')
ax1.set_ylabel('호수')
ax1.tick_params(axis='y', direction='in')

ax2 = ax1.twinx()
ax2.plot(x, y2, '-s', color='green', markersize=6, linewidth=4, alpha=0.7, label='미취학아동수(전체인구대비)')
ax2.set_ylim(0.001, 0.06)
ax2.set_ylabel(r'명')
ax2.tick_params(axis='both', direction='in')

ax2.set_zorder(ax1.get_zorder() + 10)
ax2.patch.set_visible(False)

ax1.legend(loc='upper left', bbox_to_anchor=(0.01,0.94))
ax2.legend(loc='upper left', bbox_to_anchor=(0.01,1.008))
plt.text(5,0.0575, "30㎡ 초과")
plt.hlines(0.057, 2.6, 7.5, color='blue', linestyle='--', linewidth=1)
# ax1.annotate("", xy=(-0.5, 95000), xytext=(1, 96000), arrowprops=dict(arrowstyle="->")) 한쪽 화살표
plt.vlines(2.6, 0, 97000, color='blue', linestyle='solid', linewidth=1)
plt.vlines(7.5, 0, 97000, color='blue', linestyle='solid', linewidth=1)
plt.show()


# 면적별 거주인구 연령별분포
table = officetel.pivot_table(index=['면적구분'], columns=['연령대'], values=['도로명_호'], aggfunc='count')

table = table.reindex(index=['0~14', '14~20', '20~30', '30~40', '40~60', '60~85','85~135', '135~200'])

table.columns
# 피벗테이블 결과 멀티인덱스 삭제하기
table.columns = table.columns.droplevel(0)
table.columns

table = table[['','10대미만', '10대', '20대', '30대', '40대', '50대','60대 이상']]
table = table.rename(columns={'':'면적'})

table.to_csv('C:/Users/user/Desktop/업무/4월/220419/통계표수정.csv', encoding='cp949')



#면적별 미취학 아동수 뽑기
officetel_child = officetel[officetel['child']==1]

table2 = officetel_child['child'].groupby(officetel['면적구분']).sum()
table2 = pd.DataFrame(table2)

# 테이블결합
table.set_index('면적', inplace=True)

table3 = pd.merge(table, table2, left_index=True, right_index=True)

table3 = table3.rename(columns={'child':'미취학아동'})
table3.columns
table3 = table3[['면적(㎡)','미취학아동','10대미만', '10대', '20대', '30대', '40대', '50대', '60대 이상']]

# 누적 그래프 그리기
table3['면적(㎡)'] = table3.index
table3.plot.bar(x='면적(㎡)', alpha=0.5, stacked=True, figsize=(15,7))
plt.xlabel('면적(㎡)', fontsize=13)
plt.ylabel('거주인구', fontsize=13)
plt.xticks(fontsize=12, rotation=0)
plt.show

# 2030데이터 분리
office_2030 = officetel[(officetel['연령대']=='20대') | (officetel['연령대']=='30대')]
graph_2030 = office_2030.pivot_table(index=['면적구분'], columns=['연령대'], values=['도로명_호'], aggfunc='count')
# 멀티인덱스 해제
graph_2030.columns = graph_2030.columns.droplevel(0)
# 행 순서 변경
graph_2030 = graph_2030.reindex(index=['0~14', '14~20', '20~30', '30~40', '40~60', '60~85','85~135', '135~200'])
graph_2030['면적(㎡)'] = graph_2030.index

x = graph_2030['면적(㎡)']
y1 = graph_2030['20대']
y2 = graph_2030['30대']

# 막대그래프 2개 붙여 그리기
# 그림 사이즈, 바 굵기 조정
fig, ax = plt.subplots(figsize=(12,6))
bar_width = 0.25
# 면적구간이 8개이므로 위치를 기준으로 삼음
index = np.arange(8)
# 각 연도별로 3개 샵의 bar를 순서대로 나타내는 과정, 각 그래프는 0.25의 간격을 두고 그려짐
b1 = plt.bar(index, graph_2030['20대'], bar_width, alpha=0.4, color='red', label='20대')
b2 = plt.bar(index + bar_width, graph_2030['30대'], bar_width, alpha=0.4, color='blue', label='30대')
# x축 위치를 정 가운데로 조정하고 x축의 텍스트를 면적 정보와 매칭
plt.xticks(np.arange(bar_width, 8 + bar_width, 1), graph_2030['면적(㎡)'])
# x축, y축 이름 및 범례 설정
plt.xlabel('면적(㎡)', size = 13)
plt.ylabel('명', size = 13)
plt.legend()
plt.show()

# 자치구별 거주인구
officetel = officetel.rename(columns={"SIGUNGU_NM":"자치구"})
pop_gu = officetel['도로명_호'].groupby(officetel['자치구']).count()
pop_gu = pd.DataFrame(pop_gu)
pop_gu.columns = ['거주인구']
# 자치구별 호수
officetel_ho = officetel.drop_duplicates(['도로명_호'])
hosu_gu = officetel_ho['도로명_호'].groupby(officetel_ho['자치구']).count()
hosu_gu = pd.DataFrame(hosu_gu)
hosu_gu.columns = ['호수']

pop_gu=pop_gu.sort_values(by=['거주인구'], axis=0, ascending=False)

office_gu = pd.merge(pop_gu, hosu_gu, left_index=True, right_index=True)

office_gu['자치구'] = office_gu.index

x = office_gu['자치구']
y1 = office_gu['호수']
y2 = office_gu['거주인구']

# 그래프 겹쳐그리기 ( 막대그래프와 꺾은선 그래프) : 보조축 사용
fig, ax1 = plt.subplots()
plt.xticks(rotation = 45)
ax1.bar(x, y1, color='orange', label='호수', alpha=0.7, width=0.7)
ax1.set_ylim(0, 30000)
ax1.set_xlabel('자치구')
ax1.set_ylabel('호수')
ax1.tick_params(axis='y', direction='in')

ax2 = ax1.twinx()
ax2.plot(x, y2, '-s', color='green', markersize=6, linewidth=4, alpha=0.7, label='거주인구수')
ax2.set_ylim(0, 40000)
ax2.set_ylabel(r'명')
ax2.tick_params(axis='both', direction='in')

ax2.set_zorder(ax1.get_zorder() + 10)
ax2.patch.set_visible(False)

ax1.legend(loc='upper right', bbox_to_anchor=(1,0.94))
ax2.legend(loc='upper right', bbox_to_anchor=(1,1.008))
plt.show()


# 표5 자치구별 면적별 호수
table5 = officetel_ho.pivot_table(index=['자치구'], columns=['면적구분'], values=['도로명_호'], aggfunc='count')
table5 = table5.fillna(0)
table5 = table5.astype('int')

table5.columns  = table5.columns.droplevel(0)
table5.columns
table5 = table5[['0~14', '14~20', '20~30', '30~40', '40~60', '60~85','85~135', '135~200']]


# 100% 누적막대그래프
import numpy as np
# table5['자치구'] = table5.index
# 쌓아올리는식으로 그리기
# c_bottom = np.add(table5['0~14'], table5['14~20'])
# d_bottom = np.add(c_bottom, table5['20~30'])
# f_bottom = np.add(d_bottom, table5['30~40'])
# g_bottom = np.add(d_bottom, table5['40~60'])
# h_bottom = np.add(d_bottom, table5['60~85'])
# i_bottom = np.add(d_bottom, table5['85~135'])
# j_bottom = np.add(d_bottom, table5['135~200'])
# x = range(len(table5['자치구']))
# plt.bar(x, table5['0~14'])
# plt.bar(x, table5['14~20'], bottom= table5['0~14'])
# plt.bar(x, table5['20~30'], bottom=c_bottom)
# plt.bar(x, table5['30~40'], bottom=d_bottom)
# plt.bar(x, table5['40~60'], bottom=f_bottom)
# plt.bar(x, table5['60~85'], bottom=f_bottom)
# plt.bar(x, table5['85~135'], bottom=f_bottom)
# plt.bar(x, table5['135~200'], bottom=f_bottom)
# ax = plt.subplot()
# ax.set_xticks(x)
# ax.set_xticklabels(table5['자치구'])
# plt.xlabel('자치구')
# plt.ylabel('호수')
# plt.show()



# 누적막대를 그리기위해 데이터프레임을 비율로 변경
table5 = table5.drop(['자치구'], axis=1)
table5['합계'] = table5.sum(axis=1)
table5['합계'] = table5['합계'].astype('int')
table5.columns

table5_percent = pd.DataFrame(index=table5.index, columns={'0~14', '14~20', '20~30', '30~40', '40~60', '60~85', '85~135','135~200',})
table5_percent["0~14"] = table5['0~14']/table5['합계']*100
table5_percent["14~20"] = table5["14~20"]/table5['합계']*100
table5_percent["20~30"] = table5["20~30"]/table5['합계']*100
table5_percent["30~40"] = table5["30~40"]/table5['합계']*100
table5_percent["40~60"] = table5["40~60"]/table5['합계']*100
table5_percent["60~85"] = table5["60~85"]/table5['합계']*100
table5_percent["85~135"] = table5["85~135"]/table5['합계']*100
table5_percent["135~200"] = table5["135~200"]/table5['합계']*100

table5_percent = table5_percent[['0~14', '14~20', '20~30', '30~40', '40~60', '60~85', '85~135', '135~200']]

# 퍼센트 합이 1이되는지 확인 : 행의 합
table5_percent['합계'] = table5_percent.sum(axis=1)
table5_percent = table5_percent.drop(['합계'], axis=1)

table5_percent['자치구'] = table5_percent.index

# 100% 누적 막대그래프 그리기
table5_percent = table5_percent.drop(['면적(㎡)'], axis=1)
table5_percent.plot.bar(x='자치구', alpha=0.5, stacked=True, figsize=(15,7))
plt.xlabel('자치구', fontsize=13)
plt.ylabel('호수', fontsize=13)
plt.xticks(fontsize=12, rotation=45)
plt.legend(loc='upper right',  bbox_to_anchor=(1.1,0.97))
plt.show

# 자치구별 연령대분포
table6 = officetel.pivot_table(index=['자치구'], columns=['연령대'], values=['도로명_호'], aggfunc='count')
table6.columns = table6.columns.droplevel(0)
table6.columns
table6 = table6[['10대미만', '10대', '20대', '30대', '40대', '50대', '60대 이상']]
table6['합계'] = table6.sum(axis=1)


table6_percent = pd.DataFrame(index=table6.index, columns={'10대미만', '10대', '20대', '30대', '40대', '50대', '60대 이상'})
table6_percent['10대미만'] = table6['10대미만']/table6['합계']*100
table6_percent['10대'] = table6['10대']/table6['합계']*100
table6_percent['20대'] = table6['20대']/table6['합계']*100
table6_percent['30대'] = table6['30대']/table6['합계']*100
table6_percent['40대'] = table6['40대']/table6['합계']*100
table6_percent['50대'] = table6['50대']/table6['합계']*100
table6_percent['60대 이상'] = table6['60대 이상']/table6['합계']*100
table6_percent.columns
table6_percent = table6_percent[['10대미만','10대','20대','30대','40대','50대','60대 이상']]

table6_percent['합계'] = table6_percent.sum(axis=1)
table6_percent = table6_percent.drop(['합계'],axis=1)


# 100% 누적 막대그래프 그리기
table6_percent['자치구'] = table6_percent.index
table6_percent.plot.bar(x='자치구', alpha=0.5, stacked=True, figsize=(15,7))
plt.xlabel('자치구', fontsize=13)
plt.ylabel('명', fontsize=13)
plt.xticks(fontsize=12, rotation=45)
plt.legend(loc='upper right',  bbox_to_anchor=(1.1,0.97))
plt.show
