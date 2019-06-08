---
layout: post
title: "Marketing Channel Attribution with Markov Chains in Python (Korean)"
image: img/kaylee123/DigitalMarketing.png
author: kaylee123
date: "2019-06-08T19:46:37.121Z"
tags: 
  - DigitalMarketing
  - MakovChain
  - Python
---

 마케팅 캠페인들을 적극적으로 운영하는 비즈니스에서는 마케팅 채널들이 실제 전환을 유도하는지를 파악하는데 관심을 가져야 합니다. 마케팅 노력들에 대한 ROI(투자수익률)이 중요한 KPI인 것은 모두가 아는 사실입니다.

 이 아티클에서는 다음 내용을 다룰 예정입니다.
 1. 채널별 기여(Attribution)가 중요한 이유가 무엇인지
 2. 3가지 기여 모델들
 3. 고급 기여 모델 : 마코프체인
 4. 파이썬에서 4가지 모델을 만드는 방법

## 왜 기여가 중요할까요?

 기업들이 고객들에게 마케팅할 수 있는 다수의 플래폼들이 증가하고, 대부분의 고객들은 다양한 채널들을 통해 당신의 컨텐츠에 관여하고 있기 때문에 어떠한 채널들이 전환에 기여하는지를 결정하는 것은 무엇보다도 중요합니다. 2017년의 연구에 따르면 92% 소비자들은 처음 방문한 소매업체의 웹사이트를 구매하지 않는다고 합니다([링크](https://www.episerver.com/about/news/press-room/pressreleases/92-percent-of-consumers-visiting-a-retailers-website-for-the-first-time-arent-there-to-buy/)).

![](https://cdn-images-1.medium.com/max/1200/0*fQlmBmmNldbS5RFG.jpg)

 기여의 중요성을 설명하기 위해 전환으로 이어지는 사용자의 여정의 간단한 예를 살펴보겠습니다. 이 예에서 사용자의 이름은 John 입니다.

 첫째날 : 
 귀하의 제품에 대한 John의 인식은 YouTube 광고에 의해 시작되어 귀하의 웹사이트를 방문하여 귀하의 제품 카탈로그를 검색하게 됩니다.

 약간의 브라우징 후에 귀하의 제품에 대한 John의 인식이 생겼지만 구매까지 완료할 의사는 발생하지 않았습니다.

 둘째날 :

 다음날 John이 Facebook 피드 스크롤링 하면서 귀하의 제품에 대한 또다른 광고를 수신하게 됩니다. 이 광고는 귀하의 웹사이트로 돌아가도록 권유하고 John은 이번에는 구매 프로세스를 완료합니다.

 이러한 경우에 마케팅 채널로부터의 ROI를 계산하려고 할 대 어떠한 마케팅 채널을 통해 John이 생성한 전환 금액이 어떻게 기여되었는지를 나타낼 수 있습니까?

 전통적으로 채널 기여는 퍼스트 터치, 라스트 터치 및 선형과 같은 간단하면서도 강력한 접근 방식이 사용되었습니다.


## 표준 기여 모델들

![](https://cdn-images-1.medium.com/max/2400/1*Jp6PSoHnCiP9hjPNM601qA.png)



### 라스트 터치 어트리뷰션(Last Touch Attribution)

이름에서 알 수 있듯이 라스트 터치는 발생한 수익이 사용자가 마지막에 접근한 마케팅 채널에서 기여되었다고 보는 기여 방법입니다. 

이 방식은 단순성 측면에서는 장점이 있지만 마지막 터치가 반드시 구매를 창출하는 마케팅 활동일 필요가 없으므로 기여를 지나치게 단순화할 위험이 있습니다.

위의 John의 예시에서 마지막 터치 채널(페이스북)은 구매 의도의 100%를 만들지 않았을 가능성이 큽니다. 이러한 인식은 Youtube 광고를 보았을 때 처음 발현된 것입니다.


### 퍼스트 터치 어트리뷰션(First Touch Attribution)

구매로 인해 발생한 수익은 구매를 향한 여행에서 사용자가 참여한 첫번째 마케팅 채널로부터 발생한 것입니다.

라스터 터치 방식과 마찬가지로 퍼스트 터치 어트리뷰션은 단순성 측면에서는 장점이 있지만 마찬가지로 기여 방법을 지나치게 단순화하는 위험이 있습니다.

### 선형 어트리뷰션(Linear Attribution)

이 방식에서는 구매로 이어지는 여정 동안에 사용자가 터치한 모든 마케팅 채널에 균등하게 기여가 분할 됩니다.

이 방법은 소비자 행동에서 볼 수 있는 다중 채널 터치 동작의 추세를 파악하는데 더 적합합니다. 그러나 서로 다른  채널을 구별하지 않으며 마케팅 노력에 대해모든 소비자의 


## 고급 기여 모델 : 마코프 체인(Makov Chain)

 ![](https://cdn-images-1.medium.com/max/1200/0*YSXOetB1r7L9D9oI)

## 파이썬으로 네가지 기여모델을 만드는 방법

 ![](https://cdn-images-1.medium.com/max/2400/1*6hruZFvBVnIPx7xVB361Zg.png)


```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess
```

```
# Load in our data
df = pd.read_csv('Channel_attribution.csv')

# Grab list of columns to iterate through
cols = df.columns

# Iterate through columns to change all ints to str and remove any trailing '.0'
for col in cols:
    df[col] = df[col].astype(str)
    df[col] = df[col].map(lambda x: str(x)[:-2] if '.' in x else str(x))
```


```
# Create a total path variable
df['Path'] = ''
for i in df.index:
    #df.at[i, 'Path'] = 'Start'
    for x in cols:
        df.at[i, 'Path'] = df.at[i, 'Path'] + df.at[i, x] + ' > '
```


```
# Select relevant columns
df = df[['Path', 'Conversion']]

# Sum conversions by Path
df = df.groupby('Path').sum().reset_index()

# Write DF to CSV to be executed in R
df.to_csv('Paths.csv', index=False)
The last line in the above piece of code will
```

```
# Read in the necessary libraries
if(!require(ChannelAttribution)){
  install.packages("ChannelAttribution")
  library(ChannelAttribution)
}
# Set Working Directory
setwd <- setwd('C:/Users/Morten/PycharmProjects/Markov Chain Attribution Modeling')
# Read in our CSV file outputted by the python script
df <- read.csv('Paths.csv')
# Select only the necessary columns
df <- df[c(1,2)]
# Run the Markov Model function
M <- markov_model(df, 'Path', var_value = 'Conversion', var_conv = 'Conversion', sep = '>', order=1, out_more = TRUE)
# Output the model output as a csv file, to be read back into Python
write.csv(M$result, file = "Markov - Output - Conversion values.csv", row.names=FALSE)
# Output the transition matrix as well, for visualization purposes
write.csv(M$transition_matrix, file = "Markov - Output - Transition matrix.csv", row.names=FALSE)
```

```
# Define the path to the R script that will run the Markov Model
path2script = 'C:/Users/Morten/PycharmProjects/Markov Chain Attribution Modeling/Markov.r'

# Call the R script
subprocess.call(['Rscript', '--vanilla', path2script], shell=True)

# Load in the CSV file with the model output from R
markov = pd.read_csv('Markov - Output.csv')

# Select only the necessary columns and rename them
markov = markov[['channel_name', 'total_conversion']]
markov.columns = ['Channel', 'Conversion']
```
![](https://cdn-images-1.medium.com/max/2400/1*enim8_qpl5bQUJd8opyBWQ.png)

```
# First Touch Attribution
df['First Touch'] = df['Path'].map(lambda x: x.split(' > ')[0])
df_ft = pd.DataFrame()
df_ft['Channel'] = df['First Touch']
df_ft['Attribution'] = 'First Touch'
df_ft['Conversion'] = 1
df_ft = df_ft.groupby(['Channel', 'Attribution']).sum().reset_index()

# Last Touch Attribution
df['Last Touch'] = df['Path'].map(lambda x: x.split(' > ')[-1])
df_lt = pd.DataFrame()
df_lt['Channel'] = df['Last Touch']
df_lt['Attribution'] = 'Last Touch'
df_lt['Conversion'] = 1
df_lt = df_lt.groupby(['Channel', 'Attribution']).sum().reset_index()

# Linear Attribution
channel = []
conversion = []
for i in df.index:
    for j in df.at[i, 'Path'].split(' > '):
        channel.append(j)
        conversion.append(1/len(df.at[i, 'Path'].split(' > ')))
lin_att_df = pd.DataFrame()
lin_att_df['Channel'] = channel
lin_att_df['Attribution'] = 'Linear'
lin_att_df['Conversion'] = conversion
lin_att_df = lin_att_df.groupby(['Channel', 'Attribution']).sum().reset_index()
```


```
# Concatenate the four data frames to a single data frame
df_total_attr = pd.concat([df_ft, df_lt, lin_att_df, markov])
df_total_attr['Channel'] = df_total_attr['Channel'].astype(int)
df_total_attr.sort_values(by='Channel', ascending=True, inplace=True)


# Visualize the attributions
sns.set_style("whitegrid")
plt.rc('legend', fontsize=15)
fig, ax = plt.subplots(figsize=(16, 10))
sns.barplot(x='Channel', y='Conversion', hue='Attribution', data=df_total_attr)
plt.show()
```

![](https://cdn-images-1.medium.com/max/2400/1*6aaPXxiD3augK0DdVLvw7A.png)

![](https://cdn-images-1.medium.com/max/2400/1*iTMDwKHRjSLzx5xPlOsCPw.png)

```
# Read in transition matrix CSV
trans_prob = pd.read_csv('Markov - Output - Transition matrix.csv')

# Convert data to floats
trans_prob ['transition_probability'] = trans_prob ['transition_probability'].astype(float)

# Convert start and conversion event to numeric values so we can sort and iterate through
trans_prob .replace('(start)', '0', inplace=True)
trans_prob .replace('(conversion)', '21', inplace=True)

# Get unique origin channels
channel_from_unique = trans_prob ['channel_from'].unique().tolist()
channel_from_unique.sort(key=float)

# Get unique destination channels
channel_to_unique = trans_prob ['channel_to'].unique().tolist()
channel_to_unique.sort(key=float)

# Create new matrix with origin and destination channels as columns and index
trans_matrix = pd.DataFrame(columns=channel_to_unique, index=channel_from_unique)

# Assign the probabilities to the corresponding cells in our transition matrix
for f in channel_from_unique:
    for t in channel_to_unique:
        x = trans_prob [(trans_prob ['channel_from'] == f) & (trans_prob ['channel_to'] == t)]
        prob = x['transition_probability'].values
        if prob.size > 0:
            trans_matrix[t][f] = prob[0]
        else:
            trans_matrix[t][f] = 0

# Convert all probabilities to floats
trans_matrix = trans_matrix.apply(pd.to_numeric)

# Rename our start and conversion events
trans_matrix.rename(index={'0': 'Start'}, inplace=True)
trans_matrix.rename(columns={'21': 'Conversion'}, inplace=True)

# Visualize this transition matrix in a heatmap
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(22, 12))
sns.heatmap(trans_matrix, cmap="RdBu_r")
plt.show()
```

## 결론



## 참고자료 

[Marketing Channel Attribution with Markov Chains in Python](https://medium.com/@mortenhegewald/marketing-channel-attribution-using-markov-chains-101-in-python-78fb181ebf1e) by Morten Hegewald

[구글애널리틱스 기여 모델](http://analyticsmarketing.co.kr/digital-analytics/google-analytics/1680/)

 


