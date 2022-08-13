# Polarization-in-SocialRec
In Data-Transformation-Code folder

1) Run gen_socialadj_listsv2.py
2) Run format_tweetersv3_compressed.py

-------------------------------------------
Choose to either:

A) Put all training data into history lists:

3)Run format_retweetersv3.py
4)Run format_negsamplesv3.py

-------------------------------------------

B) Split training data 1/4 goint into history lists, 3/4 held out:

3)Run hist_format_retweetersv3.py
4)Run hist_format_negsamplesv3.py

-------------------------------------------


5)Run format_noninteractersv2.py


6) Copy gendata folder over to GraphRec-WWW19 folder
7) Run GraphRec.py
