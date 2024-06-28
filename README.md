# street_name_recognation

## Conda Environment

For running this project, you have to create a Conda environment in native Windows. After installing Anaconda or MiniConda, you have to install packages to support the creation of CNN models. Read the following link for instruction.

https://medium.com/@ardisyam/create-conda-environments-in-native-windows-for-cnn-projects-4250e91ecc81

## Project Folder

|-- street_name_recognation
    |-- images
       |-- bbox
       |-- pred
       |-- test
       |-- train
   |-- models
   |-- sources
       |-- constan.py
       |-- snr.py
       |-- train_predict_pipeline.py
   |-- README.MD
   |-- LICENCE
   |-- .gitignore

## Functions

Functions are located inside the snr.py. Those functions are as follows:

- CreateListOfXmlfile(xml_folder_name)
- CreateListOfBBox(xml_filepath)
- GetImageFilePath(folder_name, filename)
- CreateListOfImages(xml_files)
- NormalizeImage(image_names, labels)
- PredictObjectLocation(test_path, prediction_path, model)
- PerformOcr(test_folder, prediction_folder, filename, model)
