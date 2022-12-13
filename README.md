# INF573_Project - DETECTION OF PIANO KEYS PRESSED IN A VIDEO
 
This readme concerns the github repository at the adress https://github.com/LongVanTH/INF573_Project.git.

The main directory contains the most important .ipynb files:
1_piano_key_notes.ipynb             Determine and label the keys regions 
2_sheet_reader.ipynb                Read the sheet music
3_hands.ipynb                       Detect the keys that might be pressed
4_keys_pressed_detection.ipynb      Determine which keys are considered as pressed
5_combine.ipynb                     Combines everything to produce the final output

The "src" folder contain the source codes of basic functions and of the first three parts above.

The "pictures" folder contain some images used.

The "mp4" folder contain
-input videos: scarborough_fair.mp4, mainly used, but also autumn_leaves.mp4 and canon_in_D.mp4,
-output videos: in subfolders with names linked to the corresponding producing .ipynb file.
All of the output videos can be regenerated running the notebooks, eventually by decommenting the code for the videos with audio (and having ffmpeg).

In the main directory:
-the following .ipynb files correspond to versions containing directly the source code:
    2_sheet_reader_with_functions_definition.ipynb,
    3_hands_with_functions_definition.ipynb,
-the following .ipynb files correspond to old attempts:
    old_full_code.ipynb: first attempt for the keyboard and the notes pressed,
    old_sheet_reader.ipynb: first reading of sheet music,
    old_sheet_reader_problem.ipynb: problems detected in the firt sheet reader, and ideas of resolution,
-z_for_report_images.ipynb was used to generate images for the report, the slides and the teaser video.