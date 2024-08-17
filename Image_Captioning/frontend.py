import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
import pickle as pk
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import keras
import numpy as np
from PIL import Image
from IPython.display import display
import re
from tqdm import tqdm



dir = "D:/internship_project/internship_project/Image_Captioning/dataset/captions.txt"
with open(dir, 'r') as cap:
    captions = cap.read()




#----------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = 'startseq ' + text + ' endseq'
    return text

#In the dataset for 1 images, there are some captions
def preprocess_captions(captions):
    
    find_captions = {}
    gather_all_captions = list()
    maximum_length = 0

    for cap in (captions.split('\n')):
        # print(cap)
        # print()
        try:
            temp = cap.split(',')
            id = temp[0].split('.')[0]
            if id not in find_captions:
                find_captions[id] = list()

            CleanText = clean_text(temp[1])
#             print(CleanText)
            find_captions[id].append(CleanText)
            gather_all_captions.append(CleanText)

            if len(CleanText.split()) > maximum_length:
                maximum_length = len(CleanText.split())
        except:
            continue

    return find_captions, gather_all_captions, maximum_length


find_captions, gather_all_captions, maximum_length = preprocess_captions(captions)
#---------------------------------------------------------------------------------------



tokenizer = Tokenizer()
tokenizer.fit_on_texts(gather_all_captions)


feature_gathered = pk.load(open('feature_gathered.pkl', 'rb'))
model = keras.saving.load_model('my_model.keras')


#---------------------------------------------------------------------
def avoid_multiple_words(caption):
    temp = ""
    temp2 = []
    for i in caption:
        if i != " ":
            temp += i
        elif i == " ":
            temp2.append(temp)
            temp=""

    sentence = []
    for i in temp2:
        if i not in sentence:
            sentence.append(i)
            
    final_text = ''
    for i in sentence:
        if i == 'startseq':
            continue
        final_text+=i
        final_text+=" "
            
    return(final_text)

#----------------------------------------------------------------------
def find_index_to_word(predicted_word, tokenizer):
    find_word = 'can not found'  
    for word, index in tokenizer.word_index.items():
        if index == predicted_word:
            return word
    
    if find_word == 'can not found':
        return 0
    else:
        return find_word

    

#---------------------------------------------------------------------
def generate_caption(img):
    image_id = img.split('.')
    image_id = image_id[0]

    img = feature_gathered[image_id]
    
    full_text = 'startseq'
    for i in range(maximum_length):

        seq = tokenizer.texts_to_sequences([full_text])[0]
        seq = pad_sequences([seq], maxlen=maximum_length)
        pred_word = np.argmax(model.predict([img, seq], verbose=0))
     
        find_word = find_index_to_word(pred_word, tokenizer)

        if find_word == 0:
            print("Sorry, Can not find word..!")
            break
        elif find_word == 'endseq':
            break

        full_text += " " + find_word
    temp = avoid_multiple_words(full_text)
    return temp
#------------------------------------------------------------------





if 'caption' not in st.session_state:
    st.session_state.caption = None
if 'image' not in st.session_state:
    st.session_state.image = None

st.title("Image Captioning", anchor=False)

try:
    image = st.file_uploader("Enter you image here...")
    st.session_state.image = image
except:
    st.info("Upload image first..")

def find_result():
    caption = generate_caption(st.session_state.image.name)
    st.session_state.caption = caption


    #Define columns
    col1, col2 = st.columns(2)
    with col1:
        st.header("Image that uploaded..")
        image = Image.open(st.session_state.image)
        st.write(image)

    with col2:
        st.header("The generated caption..")
        st.write(st.session_state.caption)
    

if st.button("Generate"):
    find_result()





    
