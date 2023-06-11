




# WHAT WE FIRST SAW ABOUT STREAMLIT

# st.title("My first app")
# st.info("what about this")
# st.write("hahaha")


#Libraries:
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import pickle


# importing iris
iris = datasets.load_iris()

# X - Y is already divided
X = iris.data
y = iris.target


#Modeling
X_train, X_test, y_train, y_test = train_test_split(X, y)

lin_reg = LinearRegression()
log_reg = LogisticRegression()
svc_ = SVC()

lin_reg_fit = lin_reg.fit(X_train, y_train)
log_reg_fit = log_reg.fit(X_train, y_train)
svc_fit = svc_.fit(X_train, y_train)

# Editing the front-end
# st.title("Modeling iris dataset by Nara")



# creating pickle files
with open("lin_reg.pkl", "wb") as li:  # wb: mode write
    pickle.dump(lin_reg_fit, li)

with open("log_reg.pkl", "wb") as lo:
    pickle.dump(log_reg_fit, lo)

with open("svc_.pkl", "wb") as sv:
    pickle.dump(svc_fit, sv)

# opening

with open("lin_reg.pkl", "rb") as li:  # rb: mode read
    linear_regression = pickle.load(li)

with open("log_reg.pkl", "rb") as lo:
    logistic_regression = pickle.load(lo)

with open("svc_.pkl", "rb") as sv:
    support_vector_classifier = pickle.load(sv)


#function to classify the plants

def classify(target_num):
    if target_num == 0: return "🌷Iris Setosa"
    elif target_num == 1: return "🌺Iris Versicolor"
    else: return "🌹Iris Virginica"

def main():

    #title
    st.title("Finding what type of Iris flower I have in my hands 🫱🏻💐")
    st.write("_**Modeling iris dataset by Nara Guerra**_")
    st.title("💫🌸🌷🌺")
    #image_url = "C:\Users\narag\OneDrive\Documentos\Ironhack\Week 15\prueba\flower.jpg"
#    st.image(image_url, caption="Iris flower")

    st.write("<p style='color:#FF1493; font-weight: bold;'>This is a result based in user input parameters.</p>", unsafe_allow_html=True)
    # st.write("**This is a result based in user input parameters**")
    # st.write("This is my first try using streamlit with iris dataset.")

    #sidebar title
    st.sidebar.header("User Input Parameters")

    #function for the User to put parameters in sidebar

    def user_input_parameters():
        sepal_length = st.sidebar.slider("Sepal Length", 4.3, 7.9, 6.0)  # label, min, max, default value
        sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.4, 3.0)  # label, min, max, default value
        petal_length = st.sidebar.slider("Petal Length", 1.0, 6.9, 4.0)  # label, min, max, default value
        petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.0)  # label, min, max, default value
        data = {"sepal_length":sepal_length,
                "sepal_width":sepal_width,
                "petal_length":petal_length,
                "petal_width":petal_width}
        features_df = pd.DataFrame(data, index=[0])
        return features_df

    df = user_input_parameters()

    #the user will also choose the model
    option = {"Linear Regression", "Logistic Regression", "SVM Classifier"}
    model = st.sidebar.selectbox("Which model do you want to use?", option)

    st.write("**The selected model by the user is:**")
    st.write(model)
    st.write("**The selected parameters by the user are:**")
    st.write(df)

    if st.button("RUN"):
        if model == "Linear Regression":
            st.success(classify(linear_regression.predict(df)))
        elif model == "Logistic Regression":
            st.success(classify(logistic_regression.predict(df)))
        else:
            st.success(classify(support_vector_classifier.predict(df)))

if __name__ == '__main__':
    main()