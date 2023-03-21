from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from bokeh.io import output_notebook, show
import os 
from bokeh.plotting import gmap 
from bokeh.models import ColumnDataSource, ColorBar, GMapOptions, HoverTool
from bokeh.transform import linear_cmap
from bokeh.palettes import Plasma256 as palette
from bokeh.embed import components
from geopy.geocoders import Nominatim
from math import cos, asin, sqrt
from textblob import TextBlob
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('/Users/hruix/Desktop/stat418/yelp_project/yelp_flask/templates'), auto_reload=True)
env.cache = {}

app = Flask(__name__, template_folder = 'templates')

output_notebook()
bokeh_width, bokeh_height = 650,550
app.config['GOOGLEMAPS_KEY'] = 'AIzaSyCLugVcuQ4zSKDkSGD2qSxy1PkWMlHciOw'

data = pd.read_csv('/Users/hruix/Desktop/stat418/yelp_project/yelp_pro.csv')

sid = SentimentIntensityAnalyzer()

@app.route('/')
def index():
     print('Request for index page received')
     return render_template('index.html')


@app.route('/map/', methods=['POST'])
def googlemap(): #lat=lat, lng=lng, zoom=13, map_type = 'terrain, color = 'stars_review', data = data
    location = request.form.get('location')

    if location:
        def getSubjectivity(text):
            return TextBlob(text).sentiment.subjectivity
  
        def getPolarity(text):
            return TextBlob(text).sentiment.polarity
  
        data['TextBlob_Subjectivity'] = data['text'].apply(getSubjectivity)
        data['TextBlob_Polarity'] = data['text'].apply(getPolarity)
        def getAnalysis(score):
            if score < 0:
                return 'Negative'
            elif score == 0:
                return 'Neutral'
            else:
                return 'Positive'
        data['TextBlob_Analysis'] = data['TextBlob_Polarity'].apply(getAnalysis)

        data['scores'] = data['text'].apply(lambda text: sid.polarity_scores(text))
        data['compound']  = data['scores'].apply(lambda score_dict: score_dict['compound'])
        data['overall_rating'] = data['stars_review'] + data['TextBlob_Polarity'] * data['TextBlob_Subjectivity'] * data['compound']

        def distance(lat1, lon1, lat2, lon2):
            p = 0.017453292519943295
            hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
            return 12742 * asin(sqrt(hav))

        def get_data(location):
            geolocator = Nominatim(user_agent="MyApp")
            loc = geolocator.geocode(location)
            lat = loc.latitude
            lon = loc.longitude
            v = {'lat': lat, 'lon': lon}
    
            mn = data[['latitude','longitude']]
            mn = mn.values.tolist()
            dist = []
            for i in range(30000):
                dist.append(distance(v['lat'],v['lon'],mn[i][0],mn[i][1]))
            data['distance'] = dist
            result = data.nsmallest(500, 'distance', keep='first')
            result = result.drop_duplicates(subset = 'name', keep = 'first')
            final = result.sort_values(by='overall_rating', ascending=False)
            final = final[0:10]
            return lat,lon,final

    
        output = get_data(location)
        lat = output[0]
        lon = output[1]
        final = output[2]

        gmap_options = GMapOptions(lat=lat, lng=lon, 
                                map_type='terrain', zoom=13)
        hover = HoverTool(
         tooltips = [
             ('name', '@name'),
             ('categoriy', '@categories')
         ])
        p = gmap(app.config['GOOGLEMAPS_KEY'], gmap_options, title='Restaurant distribution in Philadelphia', 
              width=bokeh_width, height=bokeh_height,
             tools=[hover,'reset', 'wheel_zoom', 'pan'])
    
        source = ColumnDataSource(final)
    
        mapper = linear_cmap('stars_review', palette, final['stars_review'].min(), final['stars_review'].max())
    
        center = p.circle('longitude', 'latitude', size=5, alpha=0.7, 
                      color=mapper, source=source)
        color_bar = ColorBar(color_mapper=mapper['transform'], 
                         location=(0,0), title = 'stars_review')
        center1 = p.circle([lon], [lat], size=13, alpha=1, color='red')
        p.add_layout(color_bar, 'right')
        script , div = components(p)

        print('Request for googlemap page received with location=%s' % location)
        return render_template('map.html', script = script, div = div)
    
    else:
        print('Request for googlemap page received with no location or blank location -- redirecting')
        return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(debug=True)