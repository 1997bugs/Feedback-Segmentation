import streamlit as st
from plugnplay import *


comment = st.text_input('Enter comment','This is good')
st.write('Comment is ',comment)

text_inp = comment
ps = cmt_pos2_edit(text_inp)
fl,psble = sim_bucketing(ps,model,buckets)

st.write("Your Input Text : ",text_inp)
#print("Word pairs captured:",ps)
st.write("##### Output #####")
st.write("Major themes:",fl)
st.write("Next closest pairs:",psble)
