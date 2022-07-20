import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
from PIL import Image
import plotly.express as px
from zipfile import ZipFile
from sklearn.cluster import KMeans
import lime 
from lime import lime_tabular
plt.style.use('fivethirtyeight')
#sns.set_style('darkgrid')


def main() :

    @st.cache
    def load_data():

        data2 = pd.read_csv('risk100.csv', index_col='SK_ID_CURR', encoding ='utf-8')


        sample1 = pd.read_csv('X_sampleok.csv', index_col='SK_ID_CURR', encoding ='utf-8')
        sample=sample1.drop(columns=['Unnamed: 0'],axis=1)
        # description = pd.read_csv("graph4.xslx")

        target = data2["TARGET"]

        return data2, sample, target
        # , description


    def load_model():
        '''loading the trained model'''
        pickle_in = open('pickle1000.pkl', 'rb') 
        clf = pickle.load(pickle_in)
        return clf


    @st.cache
    def load_infos_gen(data):
        lst_infos = [data.shape[0],
                     round(data["AMT_INCOME_TOTAL"].mean(), 2),
                     round(data["AMT_CREDIT"].mean(), 2)]

        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]

        targets = data2.TARGET.value_counts()

        return nb_credits, rev_moy, credits_moy, targets


    def identite_client(data, id):
        data_client = data[data.index == int(id)]
        return data_client

    @st.cache
    def load_age_population(data):
        data_age = round((data["DAYS_BIRTH"]/365), 2)
        return data_age

    @st.cache
    def load_income_population(sample):
        df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
        df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
        return df_income

    @st.cache
    def load_prediction(sample, id, clf):
        X=sample.iloc[:, :-1]
        score = clf.predict_proba(X[X.index == int(id)])[:,1]
        return score
    
    @st.cache
    def load_ext1 (sample):
        df_income111 = pd.DataFrame(sample[["TARGET","EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]])
        return df_income111

    @st.cache
    def age2 (sample):
        df_income11 = pd.DataFrame(sample[["TARGET",'DAYS_BIRTH']])
        return df_income11

    @st.cache
    def emploi (sample):
        df_empl = pd.DataFrame(sample[["TARGET",'DAYS_EMPLOYED']])
        return df_empl

    @st.cache
    def goods (sample):
        df_goods = pd.DataFrame(sample[['SK_ID_CURR', 'TARGET', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED_PERCENT',
       'CREDIT_TERM', 'ANNUITY_INCOME_PERCENT', 'CREDIT_INCOME_PERCENT',
       'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1', 'AMT_GOODS_PRICE']])
        return df_goods

    #Loading data……
    data2, sample, target= load_data()
    id_client = sample.index.values

    clf = load_model()


    #######################################
    # SIDEBAR
    #######################################

    #Title display
    html_temp = """
    <div style="background-color: grey; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Aide pour l'octroi du crédit</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)


    padding = 0
    st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)
    # Display the logo in the sidebar
    path = "logo.png"
    image = Image.open(path)
    st.sidebar.image(image, width=180)


   
    st.sidebar.markdown('***')
    st.sidebar.header("**Informations du client sélectionné**")

    # Chargement de la selectbox
    chk_id = st.sidebar.selectbox("Selection Identifiant Client", id_client)


    st.sidebar.markdown('***')


    # Chargement des infos générales
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data2)


    #######################################
    # PAGE PRINCIPALE
    #######################################

    st.markdown('***')

    # Affichage solvabilité client
    prediction = load_prediction(sample, chk_id, clf)
    st.header("**Probabilité de défaut de paiement du client sélectionné : **             {:.0f} %".format(round(float(prediction)*100, 2)))

    st.markdown('***')

    st.subheader("*Interprétation de la prédiction :*")
    new_title = '<p style="font-family:sans-serif; color:Green; font-size: 20px;">Vert : caractéristiques favorables à l obtension du crédit</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    new_title2 = '<p style="font-family:sans-serif; color:red; font-size: 20px;">Rouge : caractéristiques défavorables à l obtension du crédit</p>'
    st.markdown(new_title2, unsafe_allow_html=True)

    st.markdown('***')

    client22= sample[sample.index == chk_id]
    client2= client22.iloc[0, :-1]
    client1 =client2.T

    X_train=sample.iloc[:, :-1]
    explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X_train.columns,
            class_names=['bad', 'good'],
            mode='classification'
            )
    exp = explainer.explain_instance(
            data_row=client1, 
            predict_fn=load_model().predict_proba
            )
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    exp.as_pyplot_figure()
    st.pyplot()

    # Affichage état civil
    plt.clf()

    if  st.sidebar.checkbox("Informations générales du client ?"):

        st.markdown('***')
         
        st.subheader("*Etat civil*")

        infos_client = identite_client(data2, chk_id)

        st.write("**Genre : **", infos_client["CODE_GENDER"].values[0])
        st.write("**Age : **{:.0f} ans".format(int(infos_client["DAYS_BIRTH"]/365)))
        st.write("**Status : **", infos_client["NAME_FAMILY_STATUS"].values[0])
        st.write("**Nombre d'enfants : **{:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))
        st.markdown('***')

       # Affichage distribution Age 
        st.subheader("*Age client par rapport à l échantillon*")
        data_age = load_age_population(data2)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor = 'k', color="lightblue", bins=20)
        ax.axvline(int(infos_client["DAYS_BIRTH"].values / 365), color="green", linestyle='--')
        ax.set(title='', xlabel='Age', ylabel='')
        st.pyplot(fig)
        
        st.markdown('***')
         
        st.subheader("*Revenus client par rapport à l échantillon (USD)*")
        st.write("**Total des revenus : **{:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
        st.write("**Montant du crédit : **{:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
        st.write("**Rentes de crédit : **{:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
        st.write("**Montant du bien à créditer : **{:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))
            
       # Affichage distribution revenus
        data_income = load_income_population(data2)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color="lightblue", bins=10)
        ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
        ax.set(title='', xlabel='Revenu (USD)', ylabel='')
        st.pyplot(fig)

    else:
        st.sidebar.markdown("", unsafe_allow_html=True)

        st.markdown('***')

    st.sidebar.markdown('***')
    st.sidebar.header("**Informations générales échantillon**")


    st.markdown('***')
    if st.sidebar.checkbox("Détails"):
        st.subheader("*Informations générales échantillon*")

        # Affichage des infos dans la sidebar
        # Nombre de crédits existants
        st.markdown("<u>Nombre de prêts dans l'échantillon :</u>", unsafe_allow_html=True)
        st.text(nb_credits)

        # Revenus moyens
        st.markdown("<u>Revenus moyens $(USD) :</u>", unsafe_allow_html=True)
        st.text(rev_moy)

        # Montant crédits moyen
        st.markdown("<u>Montant moyen du prêt $(USD) :</u>", unsafe_allow_html=True)
        st.text(credits_moy)
        
        
        # Graphique camembert
        st.markdown("<u>Proportion de défaut de paiement sur l'ensemble de la clientèle</u>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,5))
        plt.pie(targets, explode=[0, 0.1], labels=['Pas de risque', 'Risque Default Paiement'], autopct='%1.1f%%', startangle=90)
        st.pyplot(fig)
        
                
    else:
        st.sidebar.markdown("", unsafe_allow_html=True)



    st.markdown('***')


    if st.sidebar.checkbox("Importances des caractéristiques ?"):
        st.subheader("*Importances des caractéristiques sur l échantillon*")
        shap.initjs()
        client = sample.iloc[:, :-1]
        # client = client[client.index == chk_id]
        number = st.slider("Choix du nombre caractéristiques",5, 10)

        fig, ax = plt.subplots(figsize=(10, 10))
        explainer = shap.TreeExplainer(load_model())
        shap_values = explainer.shap_values(client)
        shap.summary_plot(shap_values[0], client, plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
        st.pyplot(fig)  



    else:
        st.sidebar.markdown("", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
