# MLNLP

## Get Started

1. Unpack `training_data.zip` in `/data/processed`.
2. Install `requirements.txt` with `pip`.
3. Go to `/scripts/end_to_end_bert/` and run `train.py`.

## Data Structure

```
Project_Folder
  |-- data
      |-- raw           <-- folder for download scripts (not uploaded!)
          |-- train
          |-- test
      |-- processed     <-- folder for processed jsons (not uploaded!)
          |-- train
          |-- test
  |-- docker            <-- Dockerfiles, etc.
  |-- models            <-- repository for our models (not uploaded!)
      |-- {model_name_1}
      |-- {model_name_2}
          ...
  |-- notebooks         <-- jupyter notebooks, etc.
  |-- references        <-- papers, pdfs, references
  |-- scripts           <-- our code
      |-- {folder1}
          ...
  - .gitignore
  - README.MD
  - requirements.txt
```
