import streamlit as st
import pickle as pk
import pandas as pd


st.title("See :green[movies] here, It will also suggest similar 4 movies..")

movies_list = pk.load(open('movies_list.pkl', 'rb'))
find_similarity =  pk.load(open('find_similarity.pkl', 'rb'))




#----------------------------------------------
def create_tuple(index):
    find_distance = find_similarity[index]
    l = []
    for index, score in enumerate(find_distance):
        l.append((index, score))
    return l



def movie_recommandation(movie_name):
    find = final_dataset['original_title'] == movie_name
    find_index = final_dataset[find].index[0]

    movies = sorted(create_tuple(find_index), key=lambda x:x[1], reverse=True)

    top_4_similar_movies = movies[1:5]

    temp = []
    for movie in top_4_similar_movies:
        temp.append(final_dataset.iloc[movie[0]].original_title)
    return temp


final_dataset = pd.DataFrame(movies_list)
#----------------------------------------------------------





col1, col2 = st.columns(2)

with col1:
    selected_movie_name = st.selectbox(
    "Select Movie from here..",
    (movies_list['original_title']))

    st.header("Similar kind of movies..")
    recommanded_movies = movie_recommandation(selected_movie_name)
    for i in recommanded_movies:
        st.write(i)



with col2:
    movie = st.text_input("Search Movie here..")
    movie = movie.title()

    st.header("Similar kind of movies..")
    try:
        recommanded_movies = movie_recommandation(movie)
        for i in recommanded_movies:
            st.write(i)
    except:
        st.write("Write a movie name..")





