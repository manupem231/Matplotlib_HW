# Matplotlib_HW
Matplotlib Homework



```python
# Importing Dependencies

import pandas as pd
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Defining csv file path
file_path_mouse_drug = os.path.join('mouse_drug_data.csv')
file_path_clinicaltrial_data = os.path.join('clinicaltrial_data.csv')
```


```python
mouse_drug_df = pd.read_csv(file_path_mouse_drug) # Reading CSV file data
mouse_drug_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mouse ID</th>
      <th>Drug</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>f234</td>
      <td>Stelasyn</td>
    </tr>
    <tr>
      <th>1</th>
      <td>x402</td>
      <td>Stelasyn</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a492</td>
      <td>Stelasyn</td>
    </tr>
    <tr>
      <th>3</th>
      <td>w540</td>
      <td>Stelasyn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>v764</td>
      <td>Stelasyn</td>
    </tr>
  </tbody>
</table>
</div>




```python
clinicaltrial_data_df = pd.read_csv(file_path_clinicaltrial_data) # Reading CSV file data
clinicaltrial_data_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mouse ID</th>
      <th>Timepoint</th>
      <th>Tumor Volume (mm3)</th>
      <th>Metastatic Sites</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b128</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>f932</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>g107</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a457</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c819</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merging two data frames
Pymaceuticals_df = pd.merge(mouse_drug_df, clinicaltrial_data_df, on= 'Mouse ID') 

# Dropping duplicate values
Pymaceuticals_df.drop_duplicates(keep="first")

# Reading DataFrame with specific 'Drug' values (Capomulin, Infubinol, Ketapril, Placebo)
Pymaceuticals_required_df = Pymaceuticals_df[(Pymaceuticals_df['Drug'] == "Capomulin") |
                                    (Pymaceuticals_df['Drug'] == "Infubinol") |
                                    (Pymaceuticals_df['Drug'] == "Ketapril") |
                                    (Pymaceuticals_df['Drug'] == "Placebo")]
Pymaceuticals_required_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mouse ID</th>
      <th>Drug</th>
      <th>Timepoint</th>
      <th>Tumor Volume (mm3)</th>
      <th>Metastatic Sites</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>393</th>
      <td>q119</td>
      <td>Ketapril</td>
      <td>0</td>
      <td>45.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>394</th>
      <td>q119</td>
      <td>Ketapril</td>
      <td>5</td>
      <td>47.864440</td>
      <td>0</td>
    </tr>
    <tr>
      <th>395</th>
      <td>q119</td>
      <td>Ketapril</td>
      <td>10</td>
      <td>51.236606</td>
      <td>0</td>
    </tr>
    <tr>
      <th>396</th>
      <td>n923</td>
      <td>Ketapril</td>
      <td>0</td>
      <td>45.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>397</th>
      <td>n923</td>
      <td>Ketapril</td>
      <td>5</td>
      <td>45.824881</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# *** Tumor Response to Treatment ***
```


```python
# Defing pd.pivot_table to get the below DataFrame format
Tumor_Volume_Avg = pd.pivot_table(Pymaceuticals_required_df, values='Tumor Volume (mm3)', index=['Timepoint'], columns=['Drug'], aggfunc=np.mean)
```


```python
Tumor_Volume_Avg
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Timepoint</th>
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>44.266086</td>
      <td>47.062001</td>
      <td>47.389175</td>
      <td>47.125589</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>43.084291</td>
      <td>49.403909</td>
      <td>49.582269</td>
      <td>49.423329</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>42.064317</td>
      <td>51.296397</td>
      <td>52.399974</td>
      <td>51.359742</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>40.716325</td>
      <td>53.197691</td>
      <td>54.920935</td>
      <td>54.364417</td>
    </tr>
    <tr>
      <th>5</th>
      <td>25</td>
      <td>39.939528</td>
      <td>55.715252</td>
      <td>57.678982</td>
      <td>57.482574</td>
    </tr>
    <tr>
      <th>6</th>
      <td>30</td>
      <td>38.769339</td>
      <td>58.299397</td>
      <td>60.994507</td>
      <td>59.809063</td>
    </tr>
    <tr>
      <th>7</th>
      <td>35</td>
      <td>37.816839</td>
      <td>60.742461</td>
      <td>63.371686</td>
      <td>62.420615</td>
    </tr>
    <tr>
      <th>8</th>
      <td>40</td>
      <td>36.958001</td>
      <td>63.162824</td>
      <td>66.068580</td>
      <td>65.052675</td>
    </tr>
    <tr>
      <th>9</th>
      <td>45</td>
      <td>36.236114</td>
      <td>65.755562</td>
      <td>70.662958</td>
      <td>68.084082</td>
    </tr>
  </tbody>
</table>
</div>




```python
Tumor_Volume_Avg.reset_index(inplace=True) # Reseting Index to read respective column names
```


```python
Tumor_Volume_Avg.columns
```




    Index(['Timepoint', 'Capomulin', 'Infubinol', 'Ketapril', 'Placebo'], dtype='object', name='Drug')




```python
# Calulating Standard Error [SEM] of 'Tumor Volume' based on 'Drug' & 'Timepoint' indexes.
Tumor_Volume_Error = pd.DataFrame(Pymaceuticals_required_df.groupby(['Drug', 'Timepoint']).sem()['Tumor Volume (mm3)'])
```


```python
Tumor_Volume_Unstack = Tumor_Volume_Error.unstack(level=0) # Unstacking the Dataframe based on 'Drug'
Tumor_Volume_Unstack.columns = Tumor_Volume_Unstack.columns.get_level_values(1) # Unstacking and assigning each column values to specific 'Drug'
Tumor_Volume_Unstack.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.448593</td>
      <td>0.235102</td>
      <td>0.264819</td>
      <td>0.218091</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.702684</td>
      <td>0.282346</td>
      <td>0.357421</td>
      <td>0.402064</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.838617</td>
      <td>0.357705</td>
      <td>0.580268</td>
      <td>0.614461</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.909731</td>
      <td>0.476210</td>
      <td>0.726484</td>
      <td>0.839609</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_axis = np.arange(0, Tumor_Volume_Avg['Timepoint'].max()+5, 5) # Defining x_axis
plt.figure(figsize = (12,8)) # Defining figure dimensions

plt.grid() # Defining grid

plt.ylim(20, 80) # Defining y-axis limits
plt.xlim(5, Tumor_Volume_Avg['Timepoint'].max()) # Defining x-axis limits

plt.title("Tumor Response to Treatment", fontsize='14') # Defining title
plt.xlabel("Time (Days)", fontsize='14') # Defining x-label
plt.ylabel("Tumor Volume (mm3)", fontsize='14') # Defining y-label

markers = ["o", "s", "p", "P"] # List of markers for various drug names
labels = ['Capomulin', 'Infubinol', 'Ketapril', 'Placebo'] # # List of labels for various drug names
count = 0 # Defining initial count to 'Zero'

# Plotting Errorbar by considering both 'Mean(Avg)' and 'Standard Error(SEM)'
for drug in Tumor_Volume_Avg[['Capomulin', 'Infubinol', 'Ketapril', 'Placebo']]:
    plt.errorbar(x_axis, Tumor_Volume_Avg[drug], Required_Sem_Unstack[drug], marker=markers[count], alpha=0.75, linestyle = '--', label=labels[count])
    count += 1 # Incrementing count

plt.legend(loc="upper left", fancybox=True, fontsize = '12') # Defining 'legend' position

plt.savefig("Tumor_Response_to_Treatment.png") # Saving figure
plt.show() # Showing figure
```


![png](Pymaceuticals_HW_Matplotlib_files/Pymaceuticals_HW_Matplotlib_11_0.png)



```python
# *** Metastatic Response to Treatment ***
```


```python
# Defing pd.pivot_table to get the below DataFrame format
Metastatic_Avg = pd.pivot_table(Pymaceuticals_required_df, values='Metastatic Sites', index=['Timepoint'], columns=['Drug'], aggfunc=np.mean)
```


```python
Metastatic_Avg.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.160000</td>
      <td>0.280000</td>
      <td>0.304348</td>
      <td>0.375000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.320000</td>
      <td>0.666667</td>
      <td>0.590909</td>
      <td>0.833333</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.375000</td>
      <td>0.904762</td>
      <td>0.842105</td>
      <td>1.250000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.652174</td>
      <td>1.050000</td>
      <td>1.210526</td>
      <td>1.526316</td>
    </tr>
  </tbody>
</table>
</div>




```python
Metastatic_Avg.reset_index(inplace=True) # Reseting Index to read respective column names
Metastatic_Avg.dropna()
#Metastatic_Avg_df = Metastatic_Avg[['Timepoint', 'Capomulin', 'Infubinol', 'Ketapril', 'Placebo']]
```


```python
# Calulating Standard Error [SEM] of 'Metastatic Sites' based on 'Drug' & 'Timepoint' indexes.
Metastatic_Error = pd.DataFrame(Pymaceuticals_required_df.groupby(['Drug', 'Timepoint']).sem()['Metastatic Sites'])
Metastatic_Error.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Metastatic Sites</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th>Timepoint</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Capomulin</th>
      <th>0</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.074833</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.125433</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.132048</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.161621</td>
    </tr>
  </tbody>
</table>
</div>




```python
Metastatic_Unstack = Metastatic_Error.unstack(level=0) # Unstacking the Dataframe based on 'Drug'
Metastatic_Unstack.columns = Metastatic_Unstack.columns.get_level_values(1) # Unstacking and assigning each column values to specific 'Drug'
Metastatic_Unstack.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.074833</td>
      <td>0.091652</td>
      <td>0.098100</td>
      <td>0.100947</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.125433</td>
      <td>0.159364</td>
      <td>0.142018</td>
      <td>0.115261</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.132048</td>
      <td>0.194015</td>
      <td>0.191381</td>
      <td>0.190221</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.161621</td>
      <td>0.234801</td>
      <td>0.236680</td>
      <td>0.234064</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_axis = np.arange(0, Metastatic_Avg_df['Timepoint'].max()+5, 5) # Defining x-axis
plt.figure(figsize=(12,8)) # Defining figure dimensions

plt.grid() # Defining grid

plt.ylim(0,4) # Defining y-axis limit
plt.xlim(0, Metastatic_Avg_df['Timepoint'].max()) # Defining x-axis limit

plt.title("Metastatic Spread During Treatment", fontsize='14') # Defining title
plt.xlabel("Treatment Duration (Days)", fontsize='14') # Defining x-label
plt.ylabel("Metastatic Sites", fontsize='14') # Defining y-label

count = 0 # Initializing count value as 'Zero'

# Plotting Errorbar by considering both 'Mean(Avg)' and 'Standard Error(SEM)'
for drug in Metastatic_Avg_df[['Capomulin', 'Infubinol', 'Ketapril', 'Placebo']]:
        plt.errorbar(x_axis, Metastatic_Avg_df[drug], Metastatic_Unstack[drug], marker=markers[count], alpha=0.75, linestyle = '--', label=labels[count])
        count += 1

plt.legend(loc="upper left", fancybox=True, fontsize='12') #Defining legend location

plt.savefig("Metastatic_Spread_During_Treatment.png") # Saving figure
plt.show() # Showing figure
```


![png](Pymaceuticals_HW_Matplotlib_files/Pymaceuticals_HW_Matplotlib_18_0.png)



```python
# *** Survival Rates ***
```


```python
# Defing pd.pivot_table to get the below DataFrame format
Survival_count = pd.pivot_table(Pymaceuticals_df, values='Metastatic Sites', index=['Timepoint'], columns=['Drug'], aggfunc=len)
Survival_count
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
    <tr>
      <th>Timepoint</th>
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
      <th>0</th>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>26</td>
      <td>25</td>
      <td>26</td>
      <td>25</td>
    </tr>
    <tr>
      <th>5</th>
      <td>25</td>
      <td>21</td>
      <td>25</td>
      <td>23</td>
      <td>23</td>
      <td>24</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>24</td>
    </tr>
    <tr>
      <th>10</th>
      <td>25</td>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>21</td>
      <td>24</td>
      <td>23</td>
      <td>24</td>
      <td>23</td>
      <td>22</td>
    </tr>
    <tr>
      <th>15</th>
      <td>24</td>
      <td>19</td>
      <td>21</td>
      <td>19</td>
      <td>21</td>
      <td>20</td>
      <td>17</td>
      <td>24</td>
      <td>23</td>
      <td>21</td>
    </tr>
    <tr>
      <th>20</th>
      <td>23</td>
      <td>18</td>
      <td>20</td>
      <td>19</td>
      <td>20</td>
      <td>19</td>
      <td>17</td>
      <td>23</td>
      <td>21</td>
      <td>17</td>
    </tr>
    <tr>
      <th>25</th>
      <td>22</td>
      <td>18</td>
      <td>18</td>
      <td>19</td>
      <td>18</td>
      <td>17</td>
      <td>14</td>
      <td>23</td>
      <td>19</td>
      <td>16</td>
    </tr>
    <tr>
      <th>30</th>
      <td>22</td>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>15</td>
      <td>15</td>
      <td>13</td>
      <td>23</td>
      <td>18</td>
      <td>15</td>
    </tr>
    <tr>
      <th>35</th>
      <td>22</td>
      <td>14</td>
      <td>12</td>
      <td>17</td>
      <td>15</td>
      <td>14</td>
      <td>10</td>
      <td>21</td>
      <td>16</td>
      <td>14</td>
    </tr>
    <tr>
      <th>40</th>
      <td>21</td>
      <td>14</td>
      <td>10</td>
      <td>15</td>
      <td>15</td>
      <td>12</td>
      <td>9</td>
      <td>20</td>
      <td>12</td>
      <td>14</td>
    </tr>
    <tr>
      <th>45</th>
      <td>21</td>
      <td>13</td>
      <td>9</td>
      <td>11</td>
      <td>13</td>
      <td>11</td>
      <td>7</td>
      <td>20</td>
      <td>11</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>




```python
Survival_count.reset_index(inplace=True) # Resetting Index
Survival_count.dropna()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Timepoint</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>26</td>
      <td>25</td>
      <td>26</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>25</td>
      <td>21</td>
      <td>25</td>
      <td>23</td>
      <td>23</td>
      <td>24</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>25</td>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>21</td>
      <td>24</td>
      <td>23</td>
      <td>24</td>
      <td>23</td>
      <td>22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>24</td>
      <td>19</td>
      <td>21</td>
      <td>19</td>
      <td>21</td>
      <td>20</td>
      <td>17</td>
      <td>24</td>
      <td>23</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>23</td>
      <td>18</td>
      <td>20</td>
      <td>19</td>
      <td>20</td>
      <td>19</td>
      <td>17</td>
      <td>23</td>
      <td>21</td>
      <td>17</td>
    </tr>
    <tr>
      <th>5</th>
      <td>25</td>
      <td>22</td>
      <td>18</td>
      <td>18</td>
      <td>19</td>
      <td>18</td>
      <td>17</td>
      <td>14</td>
      <td>23</td>
      <td>19</td>
      <td>16</td>
    </tr>
    <tr>
      <th>6</th>
      <td>30</td>
      <td>22</td>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>15</td>
      <td>15</td>
      <td>13</td>
      <td>23</td>
      <td>18</td>
      <td>15</td>
    </tr>
    <tr>
      <th>7</th>
      <td>35</td>
      <td>22</td>
      <td>14</td>
      <td>12</td>
      <td>17</td>
      <td>15</td>
      <td>14</td>
      <td>10</td>
      <td>21</td>
      <td>16</td>
      <td>14</td>
    </tr>
    <tr>
      <th>8</th>
      <td>40</td>
      <td>21</td>
      <td>14</td>
      <td>10</td>
      <td>15</td>
      <td>15</td>
      <td>12</td>
      <td>9</td>
      <td>20</td>
      <td>12</td>
      <td>14</td>
    </tr>
    <tr>
      <th>9</th>
      <td>45</td>
      <td>21</td>
      <td>13</td>
      <td>9</td>
      <td>11</td>
      <td>13</td>
      <td>11</td>
      <td>7</td>
      <td>20</td>
      <td>11</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>




```python
New_Survival_df = Survival_count[['Timepoint', 'Capomulin', 'Infubinol', 'Ketapril', 'Placebo']] # Retrieving specific columns
```


```python
x_axis = np.arange(0, New_Survival_df['Timepoint'].max()+5, 5) # Defining x-axis
plt.figure(figsize=(12,8)) # Defining figure size

plt.ylim(20, 120) # Defining y-axis limits
plt.xlim(0, New_Survival_df['Timepoint'].max()) # Defining x-axis limits

plt.title("Survival During Treatment", fontsize = '14') # Defining title
plt.xlabel("Time (Days)", fontsize = '14') # Defining x-axis label
plt.ylabel("Survival Rate (%)", fontsize = '14') # Defining x-axis label

count = 0 # Re-initialising count as 'Zero'

for drug in New_Survival_df[['Capomulin', 'Infubinol', 'Ketapril', 'Placebo']]:
    yvals = New_Survival_df[drug]/New_Survival_df.loc[0,drug]*100 
    plt.errorbar(x_axis, yvals, marker=markers[count], linestyle = '--', alpha=0.75, label=labels[count])
    count += 1
    
plt.legend(loc="upper left", fancybox=True, fontsize = '12') # Defining legend
plt.savefig("Survival_During_Treatment.png") # Saving figure
plt.show() # Showing the plot
```


![png](Pymaceuticals_HW_Matplotlib_files/Pymaceuticals_HW_Matplotlib_23_0.png)



```python
# *** Summary Bar Graph ***
```


```python
Tumor_volume = pd.DataFrame(Pymaceuticals_required_df.groupby(['Drug', 'Timepoint']).mean()['Tumor Volume (mm3)'])
Tumor_volume.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Tumor Volume (mm3)</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th>Timepoint</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Capomulin</th>
      <th>0</th>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>44.266086</td>
    </tr>
    <tr>
      <th>10</th>
      <td>43.084291</td>
    </tr>
    <tr>
      <th>15</th>
      <td>42.064317</td>
    </tr>
    <tr>
      <th>20</th>
      <td>40.716325</td>
    </tr>
  </tbody>
</table>
</div>




```python
Tumor_volume = Tumor_volume.unstack(level = 0)
Tumor_volume.columns = Tumor_volume.columns.get_level_values(1)
Tumor_volume
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>44.266086</td>
      <td>47.062001</td>
      <td>47.389175</td>
      <td>47.125589</td>
    </tr>
    <tr>
      <th>10</th>
      <td>43.084291</td>
      <td>49.403909</td>
      <td>49.582269</td>
      <td>49.423329</td>
    </tr>
    <tr>
      <th>15</th>
      <td>42.064317</td>
      <td>51.296397</td>
      <td>52.399974</td>
      <td>51.359742</td>
    </tr>
    <tr>
      <th>20</th>
      <td>40.716325</td>
      <td>53.197691</td>
      <td>54.920935</td>
      <td>54.364417</td>
    </tr>
    <tr>
      <th>25</th>
      <td>39.939528</td>
      <td>55.715252</td>
      <td>57.678982</td>
      <td>57.482574</td>
    </tr>
    <tr>
      <th>30</th>
      <td>38.769339</td>
      <td>58.299397</td>
      <td>60.994507</td>
      <td>59.809063</td>
    </tr>
    <tr>
      <th>35</th>
      <td>37.816839</td>
      <td>60.742461</td>
      <td>63.371686</td>
      <td>62.420615</td>
    </tr>
    <tr>
      <th>40</th>
      <td>36.958001</td>
      <td>63.162824</td>
      <td>66.068580</td>
      <td>65.052675</td>
    </tr>
    <tr>
      <th>45</th>
      <td>36.236114</td>
      <td>65.755562</td>
      <td>70.662958</td>
      <td>68.084082</td>
    </tr>
  </tbody>
</table>
</div>




```python
Tumor_Perc_Change = (Tumor_volume.loc[45,:] - Tumor_volume.loc[0,:])/Tumor_volume.loc[0,:]*100
Tumor_Perc_Change_df = pd.DataFrame(Tumor_Perc_Change)
Tumor_Perc_Change_df 
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Capomulin</th>
      <td>-19.475303</td>
    </tr>
    <tr>
      <th>Infubinol</th>
      <td>46.123472</td>
    </tr>
    <tr>
      <th>Ketapril</th>
      <td>57.028795</td>
    </tr>
    <tr>
      <th>Placebo</th>
      <td>51.297960</td>
    </tr>
  </tbody>
</table>
</div>




```python
Tumor_Perc_Change.index
```




    Index(['Capomulin', 'Infubinol', 'Ketapril', 'Placebo'], dtype='object', name='Drug')




```python
#x_axis = np.arange(0, New_Survival_df['Timepoint'].max()+5, 5)
plt.figure(figsize=(12,8))

xlabels = Tumor_Perc_Change.index 
plt.axhline(y=0, color = 'black') #adds a horizontal line at zero
plt.xticks(np.arange(len(xlabels)), xlabels) # Defining xticks
plt.xlabel("Drug", fontsize = '14') # Defining x-label name

plt.title("Tumor Volume Change over 45 Day Treatment", fontsize = '14') # Defining Title name
plt.ylabel("% Tumor volume change", fontsize = '14') # Defining y-axis label



plt.bar(np.arange(4), 
        Tumor_Perc_Change, 
        # colors each bar appropriately based on + or - values by using a conditional
        color = ['red' if Tumor_Perc_Change[r] > 0 else 'green' for r in np.arange(len(xlabels))]
       )

#count for x coordinate location of value label
count = 0

#loops through tumor change data and places the value in the correct position based on + or - values
for r in Tumor_Perc_Change:
    if r < 0:
        y_coor = -3.5
    else:
        y_coor = 2
    plt.text(count, y_coor, str(round(r, 1)) + '%', ha = 'center', color = 'white')
    count += 1

plt.savefig("Tumor_Volume_Change_over_45_Day_Treatment.png") # Saving figure
plt.show()
```


    <matplotlib.figure.Figure at 0x10eceaeb8>



![png](Pymaceuticals_HW_Matplotlib_files/Pymaceuticals_HW_Matplotlib_29_1.png)
