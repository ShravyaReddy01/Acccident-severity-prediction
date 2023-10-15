import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import pickle
# Define your ordinal_encoder function
def ordinal_encoder(value, options):
    if value in options:
        encoded_value = options.index(value)
    else:
        encoded_value = -1  # Or use another appropriate encoding
    return encoded_value

# Define your get_prediction function
def get_prediction(data, model):
    predictions = model.predict(data)
    return predictions
model = joblib.load('Baseline_RandomForest_final.joblib')
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
model=joblib.load('model.pkl')
st.set_page_config(page_title="Accident Severity Prediction App", page_icon="🚧", layout="wide")  
Driving_experience = ['1-2yr', 'Above 10yr', '5-10yr', '2-5yr', '0', 'No Licence', 'Below 1yr', 'unknown']
Day_of_week = ['Monday', 'Sunday', 'Friday', 'Wednesday', 'Saturday', 'Thursday', 'Tuesday']
Age_band_of_driver = ['18-30', '31-50', 'Under 18', 'Over 51', 'Unknown', '0']
Type_of_vehicle = ['Automobile', 'Public (> 45 seats)', 'Lorry (41-100Q)', '0', 'Public (13-45 seats)', 'Lorry (11-40Q)', 'Long lorry', 'Public (12 seats)', 'Taxi', 'Pick up upto 10Q', 'Stationwagen', 'Ridden horse', 'Other', 'Bajaj', 'Turbo', 'Motorcycle', 'Special vehicle', 'Bicycle']
Weather_conditions = ['Normal', 'Raining', 'Raining and Windy', 'Cloudy', 'Other', 'Windy', 'Snow', 'Unknown', 'Fog or mist']
Routes_taken = [
    'Ravindra Barathi', 'City college to Hassan nagar via PTO, Puranapool, Bahadurpura X Roads', 'Necklace road', 'Chikkadpally market to RTC X Roads', 'Lakdi-ka-pool', 'Tarnaka to Habsiguda X Roads', 'R.P road', 'St.mary road', 'Chadarghat to Ramkoti', 'Plaza to Begumpet flyover, S.P. Road', 'Ayodhya Junction to PTI building', 'JBS towards airtel office', 'S.A.bazar to M.J.market', 'Ballamrai Junction to Tadbund Junction', 'Ramser caf� Junction to AOC centre', 'Abid road', 'Mehdipatnam to Attapur bridge', 'V V statue khairatabad to gokul theater erragadda ( via nims, panjaguut, s.r. Nagar, esi NH-9)', 'St. Anns school bollaram to Bollaram checkpost', 'Himayath nagar road', 'Others', 'NMDC to Mehdipatnam', 'New Monda Market, Bowenpally towards Bapunji nagar', 'NTR marg', 'Amberpet to Ramanthapur', 'Trimulgherry  x RDS towards lothukunta (rajiv rahadari)', 'AOC rotary to R.K. Puram X Roads', 'BJR statue Chermas', 'Bowenpally check post towards Suchitra Junction.', 'Greenlands to Ameerpet', 'Malakunta to M.J..bridge', 'Bowenpally X Roads to Bowenpally check post', 'Rd no.2 Banjara Hills', 'Chaderghat rotary towards S.J. Rotary via Kalikabher.', 'Rethi Bowli to Shaikpet Nala', 'Tadbund to Bowenpally', 'RTC X Roadss to Golconda X Roads', 'Zoo park, Tadban X Roads', 'Foot of fly over at NMDC end and NMDC Junction', 'Jubilee hills check post to madhapur', 'Greenlands fly over to V.V. statue (via raj bhavan)', 'Old Saifabad PS Junction', 'Grrenlands to NFCL Junction', 'Brook bond X Roads towards new Monda market, Bowenpally', 'Bank street to DM and HS', 'Ranghamahal to Afzalgunj', 'Rd no.12 Banjara Hills', 'Nampally Junction to AP legislative Assembly', 'Lothukunta towards st. Anns school Bollaram (Rahadari)', 'Military Diary form road Junction towards holy family Junction', 'Tankbund Arch to Sailing club', 'Sailing club to CTO, M.G.road', 'Nanal nagar to Tipu khan bridge', 'VST to RTC X Roads', 'Bowenpally towards Balanagar', 'Hanuman temple in between RTA office', 'Near foot of the flyover near old gate sect', 'Chadharghat to M.J. Market', 'Liberty to Telugutalli', 'Nayapool bridge, Afzalgunj  to S.A.Bazar'
]

st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App =�</h1>", unsafe_allow_html=True)
def main():
    option = st.sidebar.radio('Select an option', ['Home', 'prediction'])
    if option == 'Home':
         st.markdown('## Home Page')
         st.write("Here is a list of traffic signs.")
         image_path = 'cautionarysigns.jpg'
         st.image(image_path, caption='cautionarysigns.jpg', use_column_width=True)
    elif option == 'prediction':
       with st.form('prediction_form'):
        st.subheader("Enter the input for following features:")
        hour = st.slider("Time: ", 0, 23, value=0, format="%d")

        day_of_week_ = st.selectbox("Select Day of the Week: ", options=Day_of_week)
        Driving_experience_ = st.selectbox("Driving experience: ", options=Driving_experience)
        age = st.selectbox("Age band: ", options=Age_band_of_driver)
        vehicle_type = st.selectbox("Vehicle Type: ", options=Type_of_vehicle)
        weather = st.selectbox("Weather condition: ", options=Weather_conditions)
        Routes = st.selectbox("Route: ", options=Routes_taken)
        
        submit = st.form_submit_button("Predict")
        if submit:
            day_of_week_ = ordinal_encoder(day_of_week_,Day_of_week)
            Driving_experience_ = ordinal_encoder(Driving_experience_,Driving_experience)
            age = ordinal_encoder(age,Age_band_of_driver )
            vehicle_type = ordinal_encoder( vehicle_type,Type_of_vehicle)
            weather =  ordinal_encoder(weather,Weather_conditions )
            Routes =  ordinal_encoder(Routes,Routes_taken )
   
    
        data = np.array([hour, day_of_week_, Driving_experience_, age, vehicle_type, weather, Routes])
        data = data.reshape(1, -1)
            # Make predictions using the loaded model
        pred = get_prediction(data=data,model=model)

            # Display the prediction result
        st.subheader("Prediction Result:")
        st.write(f"The predicted accident severity is: {pred[0]}")
            
if __name__ == '__main__':
    main()


