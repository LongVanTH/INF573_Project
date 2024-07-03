# Detection of piano keys pressed in a video (INF573 Project)
 
This project aims to detect the keys pressed on a piano keyboard in a video, read the corresponding sheet music, and synchronize the two to compare them.<br>
More details can be found in the [report](https://github.com/LongVanTH/INF573_Project/blob/main/INF573___Detection_of_Piano_Keys_Pressed_in_a_Video___Report.pdf) and the [slides](https://github.com/LongVanTH/INF573_Project/blob/main/INF573___Detection_of_Piano_Keys_Pressed_in_a_Video___Slides.pdf).

Here is a teaser video of the project:

![Teaser video](teaser.gif)
![Detection](mp4/output_pipelines_hands/keyboard_confidence.gif)

This joint project was carried out by Aude Bouillé and Long Vân Tran Ha as part of the *INF573 - Image Analysis and Computer Vision* course offered by École Polytechnique from September 2023 to December 2023.

**Table of Contents**
<div id="user-content-toc">
  <ul>
    <li><a href='#installations'>Installations</a></li>
    <li><a href='#project-structure'>Project Structure</a></li>
	</ul>
</div>

## Installations

This project heavily relies on OpenCV and Jupyter Notebook.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/32/OpenCV_Logo_with_text_svg_version.svg" width="40%"/>
  &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/38/Jupyter_logo.svg" width="40%"/>
</p>

To install and set up the project, follow the steps below (after cloning the repository and `cd <repository-directory>`):

1. Create and activate a virtual environment:

* For Mac:
```
python3.10 -m venv .venv
source .venv/bin/activate
```

* For Windows:
```
python -m venv .venv
.venv\Scripts\activate
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Project Structure

The project is structured as follows:

```
1_piano_key_notes.ipynb             Locate and label the keys regions
2_sheet_reader.ipynb                Read the sheet music
3_hands.ipynb                       Detect the keys that might be pressed
4_keys_pressed_detection.ipynb      Determine which keys are considered as pressed
5_combine.ipynb                     Combines everything to produce the final output
```

The `src` folder contain the source codes of basic functions and of the first three notebooks above.

The `pictures` folder contains some images used.

The `mp4` folder contains
- input videos: scarborough_fair.mp4, mainly used, but also autumn_leaves.mp4 and canon_in_D.mp4,
- output videos: in subfolders with names linked to the corresponding producing .ipynb file.
All of the output videos can be regenerated running the notebooks, eventually by uncommenting the code for the videos with audio (and having `ffmpeg`).

In the main directory: <br/>
- the following .ipynb files correspond to versions containing directly the source code:<br/>
    `2_sheet_reader_with_functions_definition.ipynb`,<br/>
    `3_hands_with_functions_definition.ipynb`,<br/>
- the following .ipynb files correspond to old attempts:<br/>
    `old_full_code.ipynb`: first attempt for the keyboard and the notes pressed,<br/>
    `old_sheet_reader.ipynb`: first reading of sheet music,<br/>
    `old_sheet_reader_problem.ipynb`: problems detected in the first sheet reader, and ideas of resolution,<br/>
- `z_for_report_images.ipynb` was used to generate images for the report, the slides and the teaser video.
