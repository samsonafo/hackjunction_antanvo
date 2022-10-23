#import libraries

import pandas as pd
import numpy as np
import streamlit as st

from PIL import Image
import pickle

import warnings
warnings.filterwarnings('ignore')



st.set_page_config(
    layout='wide',
    page_title = 'Customer In-Action',
    page_icon = 'img/icon.png',
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: shown;}
            footer {visibility: hidden;}
            footer:after {
                          content:'Created by Edith | Samuel | Samson'; 
                          visibility: visible;
                          display: block;
                          position: relative;
                          #background-color: white;
                          padding: 4px;
                          top: 2px;
                          }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
#st.sidebar.image("img/side_img.jpg", width=250)

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
#set_background('')
st.title('Customer In-Action')
image = Image.open('./img/header.png')
st.image(image, caption='Photo from Antavo Website')


#load data
new_df = pd.read_csv('./data/new_opt_out_df.csv')
out_prob = pd.read_csv('./data/out_prob.csv')



#load in pipeline
filename = './pipeline/model.pkl'
model = pickle.load(open(filename,'rb'))


def intro():
    import streamlit as st

    st.write("# Welcome to Team European Triangle 👋")
    st.sidebar.success("Select a Page above.")

    st.markdown(
        """
        Working on the Antanvo Challenge, we find that being able to suggest the best next action to the customer. This keeps the customer in the loyalty program longer, while also enjoying it. And brings more profit to the business of course. 
        
        To implement this, we built a "next-step" recommender system, that uses the number of points a customer has accumulated, number days since last step and last activity to recommend what is the next best step. This best step is recommended based on the smallest churn probability possible.

        **👈 Select a Page from the dropdown on the left** to navigate within the App!

        ### Want to try out the Recommender?

        - Check out the Recommender System Page.
        - Happy to discuss our ideas - Team European Triangle 😉😊.

        ### Next Steps (If we had more time)

        - Build more combinations of steps and calculate churn(opt_out) probability.
        - Include more possible options in the dataset.
        - Feedbacks Appreciated.
    """
    )


def main():
    
    # get data
    with st.form(key='my_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            last_step = st.selectbox('last_step',['Joined the Program'])
        
        with col2:
            next_step = st.selectbox('recomended_next_step',['level_up','profile'])
        
        submit_button = st.form_submit_button(label='Submit')
        
    
    #add proposed last step to points_at_opt_out
    new_df['proposed_last_step'] = np.round(out_prob[out_prob['action'] == next_step]['last_action'].values[0],5)

    #set X and y
    X = new_df[['days','points_at_opt_out','proposed_last_step']]
    y = new_df['opt_out_prob']
    
    
    # predictions
    new_df['preds'] = model.predict(X)
    
    #calculate probability
    disp = new_df.groupby('action').agg({'opt_out_prob':'mean',"preds": "mean"}).reset_index()
    disp = disp.rename(columns={'opt_out_prob':'previous_churn_probability',
                        'preds':'new_churn_probability'})
    
    #get weight
    weight = 1/2.33   #ratio of level up to profile
    
    if submit_button:
        st.success("Find below the churn probability of next steps")
        disp['recommended_next_step'] = next_step
        disp = disp[['action','recommended_next_step','previous_churn_probability','new_churn_probability']]
        if next_step == 'level_up':
            disp['new_churn_probability'] = disp['new_churn_probability']*weight
        st.table(disp[disp['action'] == last_step].reset_index(drop=True))
        


#main()


page_names_to_funcs = {
    "Introduction": intro,
    "Recommender System": main
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
