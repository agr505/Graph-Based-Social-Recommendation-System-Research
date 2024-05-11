# Polarization-in-SocialRec
In Data-Transformation-Code folder (the folder where the dataset is located)

0) Unzip the zip files
1) Run gen_socialadj_listsv2.py
2) Run format_tweetersv3_compressed.py

-------------------------------------------
Choose to either:

A) Put all training data into history lists: (This gives all interations as history to the model)

3)Run format_retweetersv3.py 

4)Run format_negsamplesv3.py (Create negative retweeter examples and format appropriately)

-------------------------------------------

B) Split training data 1/4 goint into history lists, 3/4 held out:  (This splits some interations for history other for future interactions)
Temporal makes sure that history is consistent with time stamps

3)Run hist_format_retweetersv3.py or format_retweeters_temporal.py(latest)

4)Run hist_format_negsamplesv3.py or format_negsamplesv3_temporal.py(latest) (Create negative retweeter examples and format appropriately)

-------------------------------------------
5) Run gen_temporal_history_timeupdate.py (This orders interactions by time, making sure the interaction events are in temporal order)

6)Run format_noninteractersv2.py (Just need to do this so index errors aren't thrown)

7) Copy gendata folder over to GraphRec-WWW19 folder

8) Run either Debug_GraphRec.py or Tune_GraphRec.py


Important Variables:
pos_to_neg_interaction_dict maps each positive interaction to its associated sampled negative interactions for use during training
