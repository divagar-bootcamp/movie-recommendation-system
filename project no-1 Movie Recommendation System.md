```python
import numpy as np
import pandas as pd
import warnings
```


```python
warnings.filterwarnings('ignore')
```


```python
columns_name=['user_id','item_id','rating','timestamp']
df=pd.read_csv('u.data',sep="\t",names=columns_name)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>item_id</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>242</td>
      <td>3</td>
      <td>881250949</td>
    </tr>
    <tr>
      <th>1</th>
      <td>186</td>
      <td>302</td>
      <td>3</td>
      <td>891717742</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>377</td>
      <td>1</td>
      <td>878887116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>244</td>
      <td>51</td>
      <td>2</td>
      <td>880606923</td>
    </tr>
    <tr>
      <th>4</th>
      <td>166</td>
      <td>346</td>
      <td>1</td>
      <td>886397596</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (100000, 4)




```python
df['user_id']
```




    0        196
    1        186
    2         22
    3        244
    4        166
            ... 
    99995    880
    99996    716
    99997    276
    99998     13
    99999     12
    Name: user_id, Length: 100000, dtype: int64




```python
df['user_id'].nunique()
```




    943




```python
df['item_id'].nunique()
```




    1682




```python
movies_title=pd.read_csv('u.item',sep="\|",header=None,encoding="ISO-8859-1")
```




```python
movies_title.shape
```




    (1682, 24)




```python
movies_titles=movies_title[[0,1]]
movies_titles.columns=["item_id","title"]
movies_titles.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_id</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>GoldenEye (1995)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Four Rooms (1995)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Get Shorty (1995)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Copycat (1995)</td>
    </tr>
  </tbody>
</table>
</div>




```python
df=pd.merge(df,movies_titles,on="item_id")
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>item_id</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>242</td>
      <td>3</td>
      <td>881250949</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63</td>
      <td>242</td>
      <td>3</td>
      <td>875747190</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>226</td>
      <td>242</td>
      <td>5</td>
      <td>883888671</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>154</td>
      <td>242</td>
      <td>3</td>
      <td>879138235</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>306</td>
      <td>242</td>
      <td>5</td>
      <td>876503793</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>840</td>
      <td>1674</td>
      <td>4</td>
      <td>891211682</td>
      <td>Mamma Roma (1962)</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>655</td>
      <td>1640</td>
      <td>3</td>
      <td>888474646</td>
      <td>Eighth Day, The (1996)</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>655</td>
      <td>1637</td>
      <td>3</td>
      <td>888984255</td>
      <td>Girls Town (1996)</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>655</td>
      <td>1630</td>
      <td>3</td>
      <td>887428735</td>
      <td>Silence of the Palace, The (Saimt el Qusur) (1...</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>655</td>
      <td>1641</td>
      <td>3</td>
      <td>887427810</td>
      <td>Dadetown (1995)</td>
    </tr>
  </tbody>
</table>
<p>100000 rows × 5 columns</p>
</div>




```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>item_id</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>99995</th>
      <td>840</td>
      <td>1674</td>
      <td>4</td>
      <td>891211682</td>
      <td>Mamma Roma (1962)</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>655</td>
      <td>1640</td>
      <td>3</td>
      <td>888474646</td>
      <td>Eighth Day, The (1996)</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>655</td>
      <td>1637</td>
      <td>3</td>
      <td>888984255</td>
      <td>Girls Town (1996)</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>655</td>
      <td>1630</td>
      <td>3</td>
      <td>887428735</td>
      <td>Silence of the Palace, The (Saimt el Qusur) (1...</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>655</td>
      <td>1641</td>
      <td>3</td>
      <td>887427810</td>
      <td>Dadetown (1995)</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings=pd.DataFrame(df.groupby('title').mean()['rating'])
```


```python
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Til There Was You (1997)</th>
      <td>2.333333</td>
    </tr>
    <tr>
      <th>1-900 (1994)</th>
      <td>2.600000</td>
    </tr>
    <tr>
      <th>101 Dalmatians (1996)</th>
      <td>2.908257</td>
    </tr>
    <tr>
      <th>12 Angry Men (1957)</th>
      <td>4.344000</td>
    </tr>
    <tr>
      <th>187 (1997)</th>
      <td>3.024390</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings['num of ratings']=pd.DataFrame(df.groupby('title').count()['rating'])
```

#Create the recommendar system


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>item_id</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>242</td>
      <td>3</td>
      <td>881250949</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63</td>
      <td>242</td>
      <td>3</td>
      <td>875747190</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>226</td>
      <td>242</td>
      <td>5</td>
      <td>883888671</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>154</td>
      <td>242</td>
      <td>3</td>
      <td>879138235</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>306</td>
      <td>242</td>
      <td>5</td>
      <td>876503793</td>
      <td>Kolya (1996)</td>
    </tr>
  </tbody>
</table>
</div>




```python
moviemat=df.pivot_table(index="user_id",columns="title",values="rating")
```


```python
moviemat.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>title</th>
      <th>'Til There Was You (1997)</th>
      <th>1-900 (1994)</th>
      <th>101 Dalmatians (1996)</th>
      <th>12 Angry Men (1957)</th>
      <th>187 (1997)</th>
      <th>2 Days in the Valley (1996)</th>
      <th>20,000 Leagues Under the Sea (1954)</th>
      <th>2001: A Space Odyssey (1968)</th>
      <th>3 Ninjas: High Noon At Mega Mountain (1998)</th>
      <th>39 Steps, The (1935)</th>
      <th>...</th>
      <th>Yankee Zulu (1994)</th>
      <th>Year of the Horse (1997)</th>
      <th>You So Crazy (1994)</th>
      <th>Young Frankenstein (1974)</th>
      <th>Young Guns (1988)</th>
      <th>Young Guns II (1990)</th>
      <th>Young Poisoner's Handbook, The (1995)</th>
      <th>Zeus and Roxanne (1997)</th>
      <th>unknown</th>
      <th>Á köldum klaka (Cold Fever) (1994)</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1664 columns</p>
</div>




```python
starwars_user_ratings=moviemat['Star Wars (1977)']
```


```python
starwars_user_ratings.head(20)
```




    user_id
    1     5.0
    2     5.0
    3     NaN
    4     5.0
    5     4.0
    6     4.0
    7     5.0
    8     5.0
    9     5.0
    10    5.0
    11    NaN
    12    4.0
    13    5.0
    14    5.0
    15    5.0
    16    NaN
    17    NaN
    18    4.0
    19    NaN
    20    3.0
    Name: Star Wars (1977), dtype: float64




```python
similar_to_starwars=moviemat.corrwith(starwars_user_ratings)
```


```python
similar_to_starwars
```




    title
    'Til There Was You (1997)                0.872872
    1-900 (1994)                            -0.645497
    101 Dalmatians (1996)                    0.211132
    12 Angry Men (1957)                      0.184289
    187 (1997)                               0.027398
                                               ...   
    Young Guns II (1990)                     0.228615
    Young Poisoner's Handbook, The (1995)   -0.007374
    Zeus and Roxanne (1997)                  0.818182
    unknown                                  0.723123
    Á köldum klaka (Cold Fever) (1994)            NaN
    Length: 1664, dtype: float64




```python
corr_starwars=pd.DataFrame(similar_to_starwars,columns=['correlation'])
```


```python
corr_starwars.dropna(inplace=True)
```


```python
corr_starwars
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>correlation</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Til There Was You (1997)</th>
      <td>0.872872</td>
    </tr>
    <tr>
      <th>1-900 (1994)</th>
      <td>-0.645497</td>
    </tr>
    <tr>
      <th>101 Dalmatians (1996)</th>
      <td>0.211132</td>
    </tr>
    <tr>
      <th>12 Angry Men (1957)</th>
      <td>0.184289</td>
    </tr>
    <tr>
      <th>187 (1997)</th>
      <td>0.027398</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>Young Guns (1988)</th>
      <td>0.186377</td>
    </tr>
    <tr>
      <th>Young Guns II (1990)</th>
      <td>0.228615</td>
    </tr>
    <tr>
      <th>Young Poisoner's Handbook, The (1995)</th>
      <td>-0.007374</td>
    </tr>
    <tr>
      <th>Zeus and Roxanne (1997)</th>
      <td>0.818182</td>
    </tr>
    <tr>
      <th>unknown</th>
      <td>0.723123</td>
    </tr>
  </tbody>
</table>
<p>1410 rows × 1 columns</p>
</div>




```python
corr_starwars.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>correlation</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Til There Was You (1997)</th>
      <td>0.872872</td>
    </tr>
    <tr>
      <th>1-900 (1994)</th>
      <td>-0.645497</td>
    </tr>
    <tr>
      <th>101 Dalmatians (1996)</th>
      <td>0.211132</td>
    </tr>
    <tr>
      <th>12 Angry Men (1957)</th>
      <td>0.184289</td>
    </tr>
    <tr>
      <th>187 (1997)</th>
      <td>0.027398</td>
    </tr>
  </tbody>
</table>
</div>




```python
corr_starwars.sort_values('correlation',ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>correlation</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Man of the Year (1995)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Hollow Reed (1996)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Stripes (1981)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Beans of Egypt, Maine, The (1994)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Old Lady Who Walked in the Sea, The (Vieille qui marchait dans la mer, La) (1991)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Outlaw, The (1943)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Line King: Al Hirschfeld, The (1996)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Hurricane Streets (1998)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Good Man in Africa, A (1994)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Safe Passage (1994)</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>num of ratings</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Til There Was You (1997)</th>
      <td>2.333333</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1-900 (1994)</th>
      <td>2.600000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>101 Dalmatians (1996)</th>
      <td>2.908257</td>
      <td>109</td>
    </tr>
    <tr>
      <th>12 Angry Men (1957)</th>
      <td>4.344000</td>
      <td>125</td>
    </tr>
    <tr>
      <th>187 (1997)</th>
      <td>3.024390</td>
      <td>41</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Young Guns II (1990)</th>
      <td>2.772727</td>
      <td>44</td>
    </tr>
    <tr>
      <th>Young Poisoner's Handbook, The (1995)</th>
      <td>3.341463</td>
      <td>41</td>
    </tr>
    <tr>
      <th>Zeus and Roxanne (1997)</th>
      <td>2.166667</td>
      <td>6</td>
    </tr>
    <tr>
      <th>unknown</th>
      <td>3.444444</td>
      <td>9</td>
    </tr>
    <tr>
      <th>Á köldum klaka (Cold Fever) (1994)</th>
      <td>3.000000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1664 rows × 2 columns</p>
</div>




```python
corr_starwars=corr_starwars.join(ratings['num of ratings'])
```


```python
corr_starwars[corr_starwars['num of ratings']>100].sort_values("correlation",ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>correlation</th>
      <th>num of ratings</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Star Wars (1977)</th>
      <td>1.000000</td>
      <td>583</td>
    </tr>
    <tr>
      <th>Empire Strikes Back, The (1980)</th>
      <td>0.747981</td>
      <td>367</td>
    </tr>
    <tr>
      <th>Return of the Jedi (1983)</th>
      <td>0.672556</td>
      <td>507</td>
    </tr>
    <tr>
      <th>Raiders of the Lost Ark (1981)</th>
      <td>0.536117</td>
      <td>420</td>
    </tr>
    <tr>
      <th>Austin Powers: International Man of Mystery (1997)</th>
      <td>0.377433</td>
      <td>130</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Edge, The (1997)</th>
      <td>-0.127167</td>
      <td>113</td>
    </tr>
    <tr>
      <th>As Good As It Gets (1997)</th>
      <td>-0.130466</td>
      <td>112</td>
    </tr>
    <tr>
      <th>Crash (1996)</th>
      <td>-0.148507</td>
      <td>128</td>
    </tr>
    <tr>
      <th>G.I. Jane (1997)</th>
      <td>-0.176734</td>
      <td>175</td>
    </tr>
    <tr>
      <th>First Wives Club, The (1996)</th>
      <td>-0.194496</td>
      <td>160</td>
    </tr>
  </tbody>
</table>
<p>334 rows × 2 columns</p>
</div>




```python
def predict_movies(movie_name):
    movie_user_ratings=moviemat[movie_name]
    similar_to_movie=moviemat.corrwith(movie_user_ratings)
    corr_movie=pd.DataFrame(similar_to_movie,columns=["correlation"])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(ratings['num of ratings'])
    
    predictions=corr_movie[corr_movie['num of ratings']>100].sort_values('correlation',ascending=False)
    
    return predictions
```


```python
predict_my_movie=predict_movies("Titanic (1997)")
```


```python
predict_my_movie.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>correlation</th>
      <th>num of ratings</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Titanic (1997)</th>
      <td>1.000000</td>
      <td>350</td>
    </tr>
    <tr>
      <th>River Wild, The (1994)</th>
      <td>0.497600</td>
      <td>146</td>
    </tr>
    <tr>
      <th>Abyss, The (1989)</th>
      <td>0.472103</td>
      <td>151</td>
    </tr>
    <tr>
      <th>Bram Stoker's Dracula (1992)</th>
      <td>0.443560</td>
      <td>120</td>
    </tr>
    <tr>
      <th>True Lies (1994)</th>
      <td>0.435104</td>
      <td>208</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```
