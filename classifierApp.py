# importing libraries and modules

import streamlit as st
import pandas as pd
#import numpy as np
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.decomposition import PCA

hide_menu = """
    <style>
        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: visible;
        }
        footer:After {
            content: "Developed by martinsanalytics.com";
            display: block;
            position: relative;
            color: tomato;
            padding:10px, 0px;
        }   
    </style>
"""
from streamlit_option_menu import option_menu

selected = option_menu(
        menu_title= None, #required
        options= ["Home", "How it Works", "Contact"], #required
        icons=["house","book","envelope"],
        default_index=0,
        orientation="horizontal"
    )

#st.set_page_config(page_title="ML Predictor")

st.cache(suppress_st_warning=True)

if selected == "Home":
    st.title("Machine Learning Classifier")

    st.write(
        """
        ### Explores different ML Classifier Algorithms
        """)
    st.markdown(hide_menu, unsafe_allow_html=True)
    # File upload section

    with st.sidebar:
        st.write("**Please read how it works first.**")

    uploaded_file = st.sidebar.file_uploader("Upload your dataset: ", type=["csv","xlsx"])


    if uploaded_file is not None:
        
        print(uploaded_file)
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            # print(e)
            df = pd.read_excel(uploaded_file)
        
        try:
            st.subheader("Sample dataset")
            st.write(df)
            
        except Exception as e:
            st.write("Please upload file to the Application")

        #Dataset Summary
        st.subheader('Summary Statistics:')
        st.write(df.describe())

        #Extracting the features selection.
        features = df.shape[1]

        X = df.iloc[:,0:features - 1]
        y = df.iloc[:,-1].values

        # Categorical boolean mask
        categorical_feature_mask = X.dtypes==object


        # filter categorical columns using mask and turn it into a list
        categorical_cols = X.columns[categorical_feature_mask].tolist()
        #class_col = y.[class_mask].tolist()

        # instantiate labelencoder object
        le = LabelEncoder()
        
        #st.write(X)
        # apply le on categorical feature columns
        X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))
        y = le.fit_transform(y)

        # X[categorical_cols].head(10)

        st.subheader("Selected Features for Predictions:")
        st.write(X)

        # using the train test split function
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=104)

        classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN","SVM","Random Forest",
                                                                    "AdaBoost","Logistic Regression"))

        # Setting parameters for each classifier

        def add_parameter_ui(clf_name):
            params = dict()
            if clf_name == "KNN":
                K = st.sidebar.slider("K", 1,15)
                params["K"] = K
            elif clf_name == "SVM":
                C = st.sidebar.slider("C", 0.01,10.0)
                params["C"] = C
            elif clf_name == "Random Forest":
                max_depth = st.sidebar.slider("max_depth",2,15)
                n_estimators = st.sidebar.slider("n_estimators",1,100)
                min_samples_leaf = st.sidebar.slider("min_samples_leaf", 5,15)
                params["max_depth"] = max_depth
                params["n_estimators"] = n_estimators
                params["min_samples_leaf"] = min_samples_leaf
            elif clf_name == "AdaBoost":
                n_estimators = st.sidebar.slider("n_estimators",50,100)
                params["n_estimators"] = n_estimators
            else:
                penalty = st.sidebar.selectbox("Select Penalty",("l2","l1","elasticnet"))
                C = st.sidebar.slider("C",0.05,1.0)
                solver = st.sidebar.selectbox("solver",("lbfgs","liblinear","newton-cg","newton-cholesky","sag","saga"))
                params["penalty"] = penalty
                params["C"] = C
                params["solver"] = solver
            return params

        params = add_parameter_ui(classifier_name)

        def get_classifier(clf_name, params):
            if clf_name == "KNN":
                clf = KNeighborsClassifier(n_neighbors=params["K"])
            elif clf_name == "SVM":
                clf = SVC(C = params["C"])
            elif clf_name == "Random Forest":
                clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                            max_depth=params["max_depth"],min_samples_leaf = params["min_samples_leaf"],
                                            random_state=1234)
            elif clf_name == "AdaBoost":
                clf = AdaBoostClassifier(n_estimators=params["n_estimators"])
            else:
                clf = LogisticRegression(penalty= params["penalty"], C=params["C"],solver=params["solver"]) 
            return clf

        clf = get_classifier(classifier_name, params)

        # classification
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)

        accuracy = round(accuracy_score(y_test,y_pred),2) * 100
        precision = round(precision_score(y_test,y_pred),2) * 100
        confusionMatrix = confusion_matrix(y_test, y_pred)

        st.write(f"Classifier = {classifier_name}")
        st.write(f"Model Accuracy Score = {accuracy} %")
        st.write(f"Model Precision Score = {precision} %")
        st.subheader("Confusion Matrix Table:")
        st.write(confusionMatrix)

        # Plotting the Principal Component Analysis
        st.subheader("Plotting the PCA Scatter Plot:")
        pca  = PCA(2)
        X_projected = pca.fit_transform(X)

        x1 = X_projected[:,0]
        x2 = X_projected[:,1]

        fig = plt.figure()
        plt.scatter(x1, x2, c=y, cmap="viridis", alpha=0.8)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar()

        st.pyplot(fig)
if selected == "How it Works":
    st.title("How it works")

    st.write(
        """
             The app takes either Excel or CSV files. After the file has been uploaded into App, 
             it auto-encodes the categorical variables. \n
             **Note: The target must be the last column in your dataset.** \n
             The Application deploys five Machine Learning Algorithmns:
             - KNN
             - SVM
             - Random Forest
             - AdaBoost
             - Logistic Regression.\n
             Read the Scikit Learn Documentation for each of the classifier to know how they work before applying them on your dataset.\n
             **Note: Support Vector Machine runs slow on large dataset. If your data is big, it may take longer time before it runs.** 
             """
             )
if selected == "Contact":
    st.header(":mailbox: Get In Touch With Me!")
    
    contact_form = """
    <form action="https://formsubmit.co/olawumifadero.m@gmail.com" method="POST">
    <input type="hidden" name="_subject" value="New submission!">
     <input type="text" name="name" placeholder="Your Name" required>
     <input type="email" name="email address" placeholder="Your Email Address" required>
     <textarea name="message" placeholder="Your message goes here"></textarea>
     <input type="hidden" name="_captcha" value="false">
     <button type="submit">Send</button>
    </form>
    """

    st.markdown(contact_form, unsafe_allow_html=True)

    # Use local CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
    
    local_css("style.css")