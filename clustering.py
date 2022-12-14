import pandas as pd
from collections import Counter
import sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import json
import wget
import argparse
from datetime import datetime
import os.path

pd.options.mode.chained_assignment = None

def save_output(file, name):
    file.to_csv(f'{name}.csv', sep=',', header=True, index=False, encoding='utf-8-sig')

def load_file(url):
    # Checking for geojson file availability
    geojson_name = url.split('/')[-1]
    print('\n', 'Start clustering')
    
    if os.path.exists(geojson_name):
        return geojson_name
    
    else:
        URL = url
        response = wget.download(URL, URL.split('/')[-1])
        geojson_name = URL.split('/')[-1]
        return geojson_name

def set_datetime(dataset, year_to_filter, all_time_period, year_period, quarter_year_period, month_year_period):
    
    dataset = dataset.sort_values(by='datetime')
    year = year_to_filter

    if year_period == 'True':
        dataset = dataset[(dataset.datetime >= f'{year}-01-01') & (dataset.datetime < f'{year+1}-01-01')].reset_index(drop=True)
        period = 'year_period'
        n_clusters_time = 8/10
        
    elif quarter_year_period != 0:

        if quarter_year_period == 1:
            dataset = dataset[(dataset.datetime >= f'{year}-01-01') & (dataset.datetime < f'{year}-04-01')].reset_index(drop=True)
            period = f'quarter_period_{quarter_year_period}'
            
        elif quarter_year_period == 2:
            dataset = dataset[(dataset.datetime >= f'{year}-04-01') & (dataset.datetime < f'{year}-07-01')].reset_index(drop=True)
            period = f'quarter_period_{quarter_year_period}'
            
        elif quarter_year_period == 3:
            dataset = dataset[(dataset.datetime >= f'{year}-07-01') & (dataset.datetime < f'{year}-10-01')].reset_index(drop=True)
            period = f'quarter_period_{quarter_year_period}'
            
        elif quarter_year_period == 4:
            dataset = dataset[(dataset.datetime >= f'{year}-10-01') & (dataset.datetime <= f'{year}-12-31')].reset_index(drop=True)
            period = f'quarter_period_{quarter_year_period}'
            
        n_clusters_time = 13/15
          
    elif month_year_period != 0:
        if month_year_period <= 9:
            dataset = dataset[(dataset.datetime >= f'{year}-0{month_year_period}-01') & (dataset.datetime <= f'{year}-0{month_year_period}-31')].reset_index(drop=True)
            period = f'month_period_0{month_year_period}'
        else:
            dataset = dataset[(dataset.datetime >= f'{year}-{month_year_period}-01') & (dataset.datetime <= f'{year}-{month_year_period}-31')].reset_index(drop=True)
            period = f'month_period_{month_year_period}'
            
        n_clusters_time = 14/15
           
    elif all_time_period == 'True':
        period = 'all_time_period'
        n_clusters_time = 6/10
        return dataset, period, n_clusters_time
    
    return dataset, period, n_clusters_time

def get_correcting_dataset(geojson_name, year_to_filter_, all_time_period_, year_period_, quarter_year_period_, month_year_period_):
    
    with open(geojson_name, encoding='utf-8') as f:
        file = json.load(f)
        
        dd = []
        for i in file['features']:
            dd.append([i['properties']['datetime'], i['geometry'], i['properties']['dead_count'], 
                       i['properties']['injured_count'], i['properties']['point']])
        
        f.close()
        
    data = pd.DataFrame(dd, columns=['datetime', 'geometry', 'dead_count', 'injured_count', 'point'])
    data, pperiod, n_clusters_time = set_datetime(dataset=data, year_to_filter=year_to_filter_, all_time_period=all_time_period_, 
                        year_period=year_period_, quarter_year_period=quarter_year_period_, month_year_period=month_year_period_)
    
    data = data[['datetime', 'geometry', 'dead_count', 'injured_count', 'point']]
    data['lat'] = [i['lat'] for i in data.point]
    data['lng'] = [i['long'] for i in data.point]

    first_date = pd.to_datetime(data['datetime'].iloc[0])
    last_date = pd.to_datetime(data['datetime'].iloc[-1])
    
    frame = []
    for i in file['features']:
        if (first_date <= pd.to_datetime(i['properties']['datetime'])) and (last_date >= pd.to_datetime(i['properties']['datetime'])):
            frame.append([i['properties']['weather'][0], i['properties']['road_conditions'][0], i['properties']['light'], i['properties']['datetime']])
            
    frame = pd.DataFrame(frame, columns=['weather', 'road_conditions', 'light', 'datetime'])
    frame.sort_values(by='datetime', inplace=True)
    frame.reset_index(drop=True, inplace=True)
    
    frame['datetime'] = pd.to_datetime(frame['datetime'])
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    data.drop_duplicates('datetime', inplace=True)
    frame.drop_duplicates('datetime', inplace=True)

    data = pd.merge(data, frame, on='datetime', how='left', validate='one_to_one')
    return data, pperiod, n_clusters_time

def clustering(url, _year_to_filter, _all_time_period, _year_period, _quarter_year_period, _month_year_period):
    
    start_time = datetime.now()
    geojson_name = load_file(url)
    data, pperiod, n_clusters_time = get_correcting_dataset(geojson_name, year_to_filter_=_year_to_filter, all_time_period_=_all_time_period, 
                                  year_period_=_year_period, quarter_year_period_=_quarter_year_period, month_year_period_=_month_year_period)
    
    all_data = data[data['dead_count']>=0].copy()
    all_data = all_data[['datetime', 'dead_count', 'lat', 'lng']]
    all_data.dropna(axis=0, inplace=True)
    all_data.reset_index(drop=True, inplace=True)
  
    dead_data = data[data['dead_count']>0].copy()
    dead_data = dead_data[['datetime', 'dead_count', 'lat', 'lng']]
    dead_data.dropna(axis=0, inplace=True)
    dead_data.reset_index(drop=True, inplace=True)    
 
    injured_data = data[ (data['injured_count']>1) | (data['dead_count']>0) ].copy()
    injured_data = injured_data[['datetime', 'injured_count', 'lat', 'lng']]
    injured_data.dropna(axis=0, inplace=True)
    injured_data.reset_index(drop=True, inplace=True)
    
    w_data = data[['datetime', 'lat', 'lng', 'weather']].copy()
    w_data.replace('Метель', 'Снегопад', inplace=True)
    w_data.replace('Туман', 'Дождь', inplace=True)
    w_data.dropna(axis=0, inplace=True)
    
    rc_data = data[['datetime', 'lat', 'lng', 'road_conditions']]
    rc_data.dropna(axis=0, inplace=True)
    
    lht_data = data[['datetime', 'lat', 'lng', 'light']]
    lht_data.dropna(axis=0, inplace=True)
    lht_data.replace('Не установлено', 'В темное время суток, освещение отсутствует', inplace=True)
    lht_data.replace('Сумерки', 'В темное время суток, освещение отсутствует', inplace=True)
    lht_data.replace('В темное время суток, освещение не включено', 'В темное время суток, освещение отсутствует', inplace=True)
    
    rc_list = ['Сухое', 'Мокрое', 'Обработанное противогололедными материалами', 
               'Отсутствие, плохая различимость горизонтальной разметки проезжей части', 'Иные недостатки']
    w_list = ['Ясно', 'Пасмурно', 'Снегопад', 'Дождь']
    lht_list = ['Светлое время суток', 'В темное время суток, освещение включено', 'В темное время суток, освещение отсутствует']
    
    for i in dict(Counter(rc_data['road_conditions']).most_common()).keys():
        if i not in rc_list:
            rc_data.replace(i, 'Иные недостатки', inplace=True)
    
    
    
    to_save_rc = pd.DataFrame(columns=['type', 'labels', 'number_of_crashes', 'center_lat', 'center_lng'])
    for types in rc_list:
        under_rc_data = rc_data[rc_data['road_conditions'] == types].copy()
        under_rc_data.dropna(axis=0, inplace=True)
        under_rc_data.reset_index(drop=True, inplace=True)
        rc = get_clusters(data_for_cluster=under_rc_data, n_clusters_=n_clusters_time, coll=types, cur_year=_year_to_filter, time=pperiod, type_of_indent='str', col_name='road_conditions') 
        to_save_rc = pd.concat([to_save_rc, rc])
        to_save_rc.reset_index(drop=True, inplace=True)
        print('\n', types, '\n')
        
    to_save_w = pd.DataFrame(columns=['type', 'labels', 'number_of_crashes', 'center_lat', 'center_lng'])    
    for types in w_list:
        under_w_data = w_data[w_data['weather'] == types].copy()
        under_w_data.dropna(axis=0, inplace=True)
        under_w_data.reset_index(drop=True, inplace=True)
        w = get_clusters(data_for_cluster=under_w_data, n_clusters_=n_clusters_time, coll=types, cur_year=_year_to_filter, time=pperiod, type_of_indent='str', col_name='weather')
        to_save_w = pd.concat([to_save_w, w])
        to_save_w.reset_index(drop=True, inplace=True)
        print('\n', types, '\n')
        
    to_save_lht = pd.DataFrame(columns=['type', 'labels', 'number_of_crashes', 'center_lat', 'center_lng'])     
    for types in lht_list:
        under_lht_data = lht_data[lht_data['light'] == types].copy()
        under_lht_data.dropna(axis=0, inplace=True)
        under_lht_data.reset_index(drop=True, inplace=True)
        lht = get_clusters(data_for_cluster=under_lht_data, n_clusters_=n_clusters_time, coll=types, cur_year=_year_to_filter, time=pperiod, type_of_indent='str', col_name='light')
        to_save_lht = pd.concat([to_save_lht, lht])
        to_save_lht.reset_index(drop=True, inplace=True)
        print('\n', types, '\n')
    
    get_clusters(data_for_cluster=all_data, n_clusters_=n_clusters_time, coll='dead', cur_year=_year_to_filter, time=pperiod, type_of_indent='int', col_name='all_crashes')     
    get_clusters(data_for_cluster=dead_data, n_clusters_=n_clusters_time, coll='dead', cur_year=_year_to_filter, time=pperiod, type_of_indent='int', col_name='dead')
    get_clusters(data_for_cluster=injured_data, n_clusters_=n_clusters_time, coll='injured', cur_year=_year_to_filter, time=pperiod, type_of_indent='int', col_name='injured')
    save_output(file=to_save_w, name = f'cluster_weather_{_year_to_filter}_{pperiod}')    
    save_output(file=to_save_rc, name = f'cluster_road_conditions_{_year_to_filter}_{pperiod}')
    save_output(file=to_save_lht, name = f'cluster_light_{_year_to_filter}_{pperiod}')    
    print('\n', 'Complete clustering', datetime.now() - start_time)
    
def get_clusters(data_for_cluster, n_clusters_, coll, cur_year, time, type_of_indent, col_name):
    
    try:
        x = data_for_cluster[['lat', 'lng']].values
        
        n_clust = int(len(data_for_cluster)*n_clusters_)
        
        if time=='all_time_period':
            kmeans = MiniBatchKMeans(n_clusters=n_clust, init='k-means++', max_iter=1000, batch_size=2048, compute_labels=True, verbose=1).fit(x)
        else:
            kmeans = KMeans(n_clusters=n_clust, init='k-means++', max_iter=1000, random_state=0, n_init=8).fit(x)
            
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        data_for_cluster['labels'] = labels
        
        armin, dicts = sklearn.metrics.pairwise_distances_argmin_min(centers, list(zip(data_for_cluster['lat'], data_for_cluster['lng'])))
        data_for_cluster['center_lat'] = data_for_cluster['center_lng'] = ['zero_label' for i in range(len(data_for_cluster))]

        for i, point in enumerate(armin):
            data_for_cluster['center_lat'][point] = centers[i][0]
            data_for_cluster['center_lng'][point] = centers[i][1]

        for i in range(len(data_for_cluster)):
            if data_for_cluster['center_lat'][i] == 'zero_label':

                tab = data_for_cluster[data_for_cluster['labels']==data_for_cluster['labels'][i]]

                for u in tab.index:
                    if tab['center_lat'][u] != 'zero_label':
                        cent_lat = tab['center_lat'][u]
                        cent_lng = tab['center_lng'][u]

                for q in tab.index:
                    if tab['center_lat'][q] == 'zero_label':
                        data_for_cluster['center_lat'][q] = cent_lat
                        data_for_cluster['center_lng'][q] = cent_lng
                        
        most_count = 1
            
        while list(dict(Counter(data_for_cluster['labels']).most_common(most_count)).values())[-1] != 1:
            most_count += 1
        
        most_count = most_count - 1
        active_labels = np.array(list(dict(Counter(data_for_cluster['labels']).most_common(most_count)).keys()))
        data_for_cluster = data_for_cluster[data_for_cluster['labels'].isin(active_labels)]
        data_for_cluster.reset_index(drop=True, inplace=True)
        
        table_for_result = data_for_cluster.copy()

        if type_of_indent=='int':
            result = pd.DataFrame(columns=['labels', f'number_of_{coll}', 'number_of_crashes', 'center_lat', 'center_lng'])

            for i in set(table_for_result['labels']):
                tab = table_for_result[table_for_result['labels']==i]
                tab.reset_index(drop=True, inplace=True)

                dead_sum = sum(tab[f'{coll}_count'])
                dtp_sum = len(tab)

                lst = []
                lst.append([tab['labels'][0], dead_sum, dtp_sum, tab['center_lat'][0], tab['center_lng'][0]])

                res = pd.DataFrame(lst, columns=['labels', f'number_of_{coll}', 'number_of_crashes', 'center_lat', 'center_lng'])
                result = pd.concat([result, res])
                result.reset_index(drop=True, inplace=True)
                
            if col_name=='all_crashes':
                result.rename(columns={ f'number_of_{coll}' : col_name }, inplace=True)   
                
            save_output(file=result, name = f'cluster_{col_name}_{cur_year}_{time}')
                
        elif type_of_indent=='str':
            result = pd.DataFrame(columns=['type', 'labels', 'number_of_crashes', 'center_lat', 'center_lng'])

            for i in set(table_for_result['labels']):
                tab = table_for_result[table_for_result['labels']==i]
                tab.reset_index(drop=True, inplace=True)

                dtp_sum = len(tab)
                cond = tab[col_name][0]
                lst = []
                lst.append([cond, tab['labels'][0], dtp_sum, tab['center_lat'][0], tab['center_lng'][0]])

                res = pd.DataFrame(lst, columns=['type', 'labels', 'number_of_crashes', 'center_lat', 'center_lng'])
                result = pd.concat([result, res])        
                result.reset_index(drop=True, inplace=True)
                
            return result
    except:
        print(f'No result in period, type: {col_name}, year: {cur_year}')
    
def main():
    parser = argparse.ArgumentParser(description='Clustering crashes: default_time_period: year, default_location: Samara')
    parser.add_argument('url', nargs='?', default='https://cms.dtp-stat.ru/media/opendata/samarskaia-oblast.geojson', type=str, help='URL to geojson file with car crashes')
    parser.add_argument('_year_to_filter', nargs='?', default=2021, type=int, help='Year to do clustering, default=2021')
    parser.add_argument('_all_time_period', nargs='?', default='False', type=str, help='Choose period: all time period, default=False')
    parser.add_argument('_year_period', nargs='?', default='True', type=str, help='Choose period: year period, default=True')
    parser.add_argument('_quarter_year_period', nargs='?', default=0, type=int, help='Choose period: quarter period of year, default=0')
    parser.add_argument('_month_year_period', nargs='?', default=0, type=int, help='Choose period: mounth of year, default=0')
    args = parser.parse_args()
    clustering(args.url, args._year_to_filter, args._all_time_period, args._year_period, args._quarter_year_period, args._month_year_period)

if __name__ == "__main__":
        main()