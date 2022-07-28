import os
import sqlite3
from datetime import datetime

import pandas as pd
import numpy as np
import geopandas as gpd

from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st


# Configuration de la page
st.set_page_config(
     page_title="Analyse Piézo",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
 )
# Réduire bande blanche en haut de la page
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;                              
                }
        </style>
        """, unsafe_allow_html=True)

# Affichage des titres
st.title("Affichage cartographique de données piézomètriques")
st.markdown("Cette application permet de visualiser la répartition des piézomètres \
            en fonction du type de terrain et du clustering")

@st.cache
def f_requete_sql (requete) :
    try:
        connexion = sqlite3.connect('./liste_piezos.db')
        curseur = connexion.cursor()
        curseur.execute(requete)
        connexion.commit()
        resultat = curseur.fetchall()
        curseur.close()
        connexion.close()
        return resultat
    except sqlite3.Error as error:
        print("Erreur lors du mis à jour dans la table", error)
        
@st.cache        
def cmyk_to_rgb(c, m, y, k, cmyk_scale, rgb_scale=1):
    r = rgb_scale * (1.0 - c / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    g = rgb_scale * (1.0 - m / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    b = rgb_scale * (1.0 - y / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    return r, g, b

@st.cache(allow_output_mutation=True)
def f_data_litho():  
    # Chargement de la carto lithographie
    URL = 'http://mapsref.brgm.fr/wxs/infoterre/catalogue?SERVICE=WFS&REQUEST=GetFeature&VERSION=2.0.0&TYPENAMES=ms:LITHO_1M_SIMPLIFIEE'

    data_litho = gpd.read_file(URL)
    data_litho[["C_FOND", "M_FOND", "J_FOND", "N_FOND"]] = data_litho[["C_FOND", "M_FOND", "J_FOND", "N_FOND"]].astype(int, copy=False)
    data_litho["RGB"] = [(tuple(color)) for color in np.stack(cmyk_to_rgb(data_litho.C_FOND, data_litho.M_FOND, data_litho.J_FOND, data_litho.N_FOND, cmyk_scale = 100, rgb_scale=255)).T]
    data_litho["RGB_sns"] = [(tuple(color)) for color in np.stack(cmyk_to_rgb(data_litho.C_FOND, data_litho.M_FOND, data_litho.J_FOND, data_litho.N_FOND, cmyk_scale = 100, rgb_scale=1)).T]
    data_litho["RGB_str"] = [str(tuple(color)) for color in np.stack(cmyk_to_rgb(data_litho.C_FOND, data_litho.M_FOND, data_litho.J_FOND, data_litho.N_FOND, cmyk_scale = 100, rgb_scale=255)).T]
    data_litho["RGB_px"] = ["rgb" + data_litho.loc[line,"RGB_str"] for line in data_litho.index]
    return data_litho

@st.cache(allow_output_mutation=True)
def f_data_piezo():
    # Requete de chargement des données auprès de la bdd sqlite
    requete = """
            SELECT code_bss, latitude, longitude, altitude_station, profondeur_investigation, codes_bdlisa, cluster_kmeans, FRANCE_lvl_1, 
                FRANCE_lvl_2, FRANCE_lvl_3, FRANCE_lvl_4, EtatEH, NatureEH, MilieuEH, ThemeEH, OrigineEH
            FROM data_piezo
            INNER JOIN TME
            ON codes_bdlisa = CodeEH
            WHERE cluster_kmeans IS NOT NULL
            """
    data_piezo = pd.DataFrame(f_requete_sql(requete))
    data_piezo.columns = ["code_bss", "latitude", "longitude", "altitude_station", "profondeur_investigation", "codes_bdlisa", "cluster_kmeans", "FRANCE_lvl_1", "FRANCE_lvl_2", "FRANCE_lvl_3", "FRANCE_lvl_4", "EtatEH", "NatureEH", "MilieuEH", "ThemeEH", "OrigineEH"]
    data_piezo.set_index("code_bss", inplace=True)

    dict_etat = {"1":"Nappe captive" , 
             "2":"Nappe libre", 
             "3": "Parties libres et captives", 
             "4":"Alternativement libre puis captive"}
    dict_nature = {"0":"inconnue",
                "3":"Système aquifère", 
                "5":"Unité aquifère", 
                "6":"Unité semi-perméable",
                "7":"Unité imperméable"}
    dict_milieu = {"1":"Poreux", 
                "2":"Sédimentaire", 
                "3": "Karstique", 
                "4":"Matricielle / fissures",
                "5": "Karstique / fissures",
                "6": "Fractures et/ou fissures",
                "8":"Matricielle / karstique",
                "9":"Matrice/fracture/karst"}
    dict_theme = {"0":"inconnue",
                "1":"Alluvial", 
                "2":"Sédimentaire", 
                "3":"Matricielle / fissures" , 
                "4":"Intensément plissés de montagne",
                "5":"Volcanisme"}
    dict_origine = {"1":"Carte géologique ou hydrogéologique", 
                    "2":"Complétude Totale", 
                    "3":"Complétude Partielle", 
                    "4":"Agrégation par héritage"}

    data_piezo.replace({"EtatEH": dict_etat}, inplace=True)
    data_piezo.replace({"NatureEH": dict_nature}, inplace=True) 
    data_piezo.replace({"MilieuEH": dict_milieu}, inplace=True) 
    data_piezo.replace({"ThemeEH": dict_theme}, inplace=True) 

    return data_piezo
  
@st.cache
def f_geo_data_piezo(data_litho, data_piezo):
    data_piezo = gpd.GeoDataFrame(data_piezo, geometry=gpd.points_from_xy(data_piezo.longitude, data_piezo.latitude, crs="CRS84"))
    data_piezo.to_crs(crs='EPSG:4326', inplace=True)
    data_piezo = gpd.sjoin(data_piezo, data_litho.loc[:,["DESCR", "TYPE", "geometry", "RGB_sns"]], how='left')
    data_piezo["dummy_column_for_size"] = 1
    # Suppresion des lignes NaN dans DESCR
    index_to_drop= data_piezo[ data_piezo['DESCR'].isna()].index
    data_piezo.drop(index_to_drop , inplace=True)
    return data_piezo



@st.cache(suppress_st_warning=True, hash_funcs={dict: lambda _: None})
def f_carto(data_litho, data_piezo, cluster_level) :
    
    dict_colors = dict(zip(data_litho.DESCR.unique(), data_litho.RGB_px.unique()))
        
    carto = px.choropleth_mapbox(data_frame = data_litho, 
                     geojson=data_litho["geometry"], 
                     locations = data_litho.index, 
                     color=data_litho["DESCR"],
                     mapbox_style='white-bg',
                     color_discrete_map=dict_colors,
                     opacity=0.5,
                     labels= {"DESCR" : "Terrain"},
                     center={"lat": 47, "lon": 2.2},
                     category_orders = {"DESCR": sorted(data_litho.DESCR.unique())},
                     zoom=5)

    
    
    data_scat = px.scatter_mapbox(data_piezo,
                         lat="latitude",
                         lon="longitude",
                         color=cluster_level,
                         color_discrete_sequence = px.colors.qualitative.Light24,
                         size = "dummy_column_for_size",
                         size_max = 5,
                         hover_name = data_piezo.index,
                         hover_data={cluster_level:True,
                                     'dummy_column_for_size':False,
                                     'latitude':False,
                                     'longitude':False,
                                     'DESCR': True,
                                     'TYPE': True,
                                     'EtatEH' : True,
                                     'NatureEH' : True,
                                     'MilieuEH' : True,
                                     'ThemeEH' : True},
                         labels={cluster_level : "Clusters"},
                         category_orders = {cluster_level: sorted(data_piezo[cluster_level].unique())},
                         ).data

    for item in range(len(data_scat)):
        carto.add_trace(data_scat[item])
        carto.update_traces(mode="markers", 
                          selector=dict(type='scattermapbox'))
    carto.update_layout(legend= {'itemsizing': 'constant'},
                    margin=dict(l=0, r=0, t=0, b=0),
                    autosize=False, width=600, height=600)

    return carto


def f_count_plot(data_piezo, data_litho, filtre, valeur, cluster_level):
    # Create an array with the colors you want to use 

    fig, ax = plt.subplots()
    
    if filtre == "Terrain" :
        
        donnees_cluster = pd.crosstab(data_piezo[cluster_level], data_piezo["DESCR"], normalize='index')
        ax = sns.barplot(x=valeur, y=donnees_cluster.index,
                palette = px.colors.qualitative.Light24,
                data=donnees_cluster)

        ax.set_xlim(right=1)
        ax.set_ylabel('')
        ax.set_xlabel('')

    if filtre == "Cluster" :
        
        donnees_terrain = pd.crosstab(data_piezo["DESCR"], data_piezo[cluster_level],  normalize='index')
        # Create an array with the colors you want to use
        colors = data_litho[["DESCR","RGB_sns"]].drop_duplicates().sort_values(by="DESCR")["RGB_sns"].values
        # Set your custom color palette
        customPalette = sns.set_palette(sns.color_palette(colors))

        ax = sns.barplot(x=valeur, y=donnees_terrain.index,
                        palette = customPalette,
                        data=donnees_terrain,
                           order = sorted(donnees_terrain.index.drop_duplicates().values))
        ax.set_xlim(right=1)
        ax.set_xlabel('')
        ax.set_ylabel('')

    st.pyplot(fig)
    

@st.cache(allow_output_mutation=True)
def f_chroniques(cluster_level) :
    # Liste des clusters
    requete = f"""
                SELECT DISTINCT {cluster_level}
                FROM cluster_list
                """
    clusters = [item[0] for item in f_requete_sql(requete)]

    # Liste des chroniques pour 1er cluster
    requete = f"""
            SELECT code_bss
            FROM cluster_list
            WHERE {cluster_level} = {clusters[0]}
            """
    list_chroniques = [item[0] for item in f_requete_sql(requete)]

    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    chroniques = pd.read_csv(f"./Clustering/data/FRANCE.csv", sep = ";", parse_dates=True, index_col="date_mesure", date_parser=custom_date_parser)

    df_clusters = pd.DataFrame()
    df_clusters[clusters[0]] = chroniques[list_chroniques].mean(axis=1)

    for index, cluster in enumerate(clusters) :
        if index > 0 :
            requete = f"""
            SELECT code_bss
            FROM cluster_list
            WHERE {cluster_level} = {cluster}
                """
            list_chroniques = [item[0] for item in f_requete_sql(requete)]

            custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
            chroniques = pd.read_csv(f"./Clustering/data/FRANCE.csv", sep = ";", parse_dates=True, index_col="date_mesure", date_parser=custom_date_parser)

            df_clusters[cluster] = chroniques[list_chroniques].mean(axis=1)

    figure = px.line(df_clusters, color_discrete_sequence = px.colors.qualitative.Light24)
    figure.update_layout(legend_title="Clusters")
    return figure
            

def f_cluster_terrain (df_data_piezo, cluster_level, filt_EtatEH, filt_MilieuEH, filt_DESCR) :
    cross_df = pd.crosstab(df_data_piezo[cluster_level], columns=[df_data_piezo["EtatEH"],
                                                                    df_data_piezo["MilieuEH"],
                                                                    df_data_piezo["DESCR"]],
                                                    normalize=True)
    # Filtrage
    result_df = pd.DataFrame(cross_df.idxmax(axis=0), columns=["cluster"]).reset_index()
    result_df = result_df.loc[result_df["EtatEH"].isin(filt_EtatEH) & result_df["MilieuEH"].isin(filt_MilieuEH) & result_df["DESCR"].isin(filt_DESCR)]
    return result_df


def f_correlation_cluster_terrain (cluster_level, data_piezo, filtre1, filtre2):
    crosstab = pd.crosstab(data_piezo[cluster_level],
                       columns=[data_piezo[filtre1], data_piezo[filtre2]],
                       normalize='columns')

    fig, ax = plt.subplots(figsize=(15,7))
    ax = sns.heatmap(crosstab, annot=pd.crosstab(data_piezo[cluster_level],
                       columns=[data_piezo[filtre1], data_piezo[filtre2]]),
                       fmt='g')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)


def main():    

    dict_cluster_levels = {1: "FRANCE_lvl_1",
                2: "FRANCE_lvl_2",
                3: "FRANCE_lvl_3",
                4: "FRANCE_lvl_4"}
    
    col1, col2 = st.columns((8,3))

    with col1 :
        st_cluster_level = st.selectbox(label = "Niveau de clustering :", options = (1, 2, 3, 4), index = 2)
        cluster_level = dict_cluster_levels[st_cluster_level]

        cluster_level = dict_cluster_levels[st_cluster_level]

        data_litho = f_data_litho()    
        data_piezo = f_data_piezo()
        data_piezo = f_geo_data_piezo(data_litho, data_piezo)
        data_piezo = data_piezo.loc[(data_piezo["EtatEH"] != 'X') & (data_piezo["MilieuEH"] != 'X')]

        list_DESCR = sorted(data_piezo["DESCR"].drop_duplicates().values)
        list_clusters = sorted(data_piezo[cluster_level].drop_duplicates().values)
        list_EtatEH = sorted(data_piezo["EtatEH"].drop_duplicates().values)
        list_MilieuEH = sorted(data_piezo["MilieuEH"].drop_duplicates().values)


        carte = f_carto(data_litho, data_piezo, cluster_level)
        st.plotly_chart(carte, use_container_width=True)



    with col2 :   
        valeur_cluster = st.selectbox('Choix du cluster :', tuple(list_clusters))
        f_count_plot(data_piezo, data_litho, "Cluster", valeur_cluster, cluster_level)

        valeur_terrain = st.selectbox('Choix du terrain :', tuple(list_DESCR))
        f_count_plot(data_piezo, data_litho, "Terrain", valeur_terrain, cluster_level)


    # Plot chroniques clusters
    chroniques = f_chroniques(cluster_level)
    st.subheader("Chroniques type des clusters")
    st.plotly_chart(chroniques, use_container_width=True)
    
    # Top cluster pour combinaison de terrains
    ## Sélection des filtres
    with st.form('Form1'): 
        st.subheader("Cluster type en fonction des caractéristiques terrain")

        colA, colB, colC, colD= st.columns((1,1,1,1))
        with colA :
            filt_EtatEH = st.multiselect("Choix de la variable Etat :", list_EtatEH, "Nappe libre")
        with colB :
            filt_MilieuEH = st.multiselect("Choix de la variable Milieu :", list_MilieuEH, "Sédimentaire")
        with colC :
            filt_DESCR = st.multiselect("Choix de la variable DESCR :", list_DESCR, list_DESCR[0])
        with colD :
            submitted1 = st.form_submit_button('Submit')
        
        colD, colE, colF = st.columns((1,3,1))
        with colD:
            st.empty()
        with colE:
            df_cluster_terrain = f_cluster_terrain(data_piezo, cluster_level, filt_EtatEH, filt_MilieuEH, filt_DESCR)
            st.dataframe(df_cluster_terrain)
        with colF:
            st.empty()
    
    # Filtres pour corrélations cluster terrain
    ## Création des selectbox
    with st.form('Form2'):
        st.subheader("Corrélation entre clusters et combinaison de cractéristiques terrain")

        colG, colH, colI, colJ= st.columns((1,1,1,1))
        with colG:
            st.empty()
        with colH:
            filtre1 = st.selectbox("Choix de la variable 1", ("EtatEH", "NatureEH", "MilieuEH", "ThemeEH", "DESCR", "TYPE"), index=0)
        with colI:
            filtre2 = st.selectbox("Choix de la variable 2", ("EtatEH", "NatureEH", "MilieuEH", "ThemeEH", "DESCR", "TYPE"), index=1)
        with colJ:
            submitted2 = st.form_submit_button('Submit')
    ## Affichage du graphqiue
        colK, colL, colM = st.columns((1,2,1))
        with colK:
            st.empty()
        with colL:
            f_correlation_cluster_terrain(cluster_level, data_piezo, filtre1, filtre2)
        with colM:
            st.empty()
    

if __name__ == "__main__":
    main()

    
