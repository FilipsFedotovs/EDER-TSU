# EDER-TSU
Emulsion Data Event Reconstruction - Track Segment Union.
Release 1.

This README just serves as a very short user guide, the documentation will be written much later.

Installation steps
--
1) pip3 install --upgrade pip --user

   This step is highly recommended in order to avoid possible problems with installation of other packages.

3) pip3 install tensorflow==1.14.0 --user

4) pip3 install keras==2.3.1 --user

5) pip3 install tensorflow-gpu==1.14.0 --user
   
   Only required if you have intent to create/train CNN models

6) pip3 install pandas
   
   Pandas are used extensively in this package for routine data manipulation.

7) pip3 install psutil
   
   This utility helps to monitor the script memory usage

8) pip3 install 'h5py==2.10.0' --force-reinstall
   
   Downgrading h5py in order to make it work with the existing version of the Tensorflow.

9) go to your home directory on AFS where you would like to install the package

10) git clone https://github.com/FilipsFedotovs/EDER-TSU/
11) cd EDER-TSU/
12) python3 setup.py
13) The installation will require another directory, please enter the location on EOS where you would like to keep data and the models
   Has to provide up to 10-100 GB of storage depending on whether particular components of the framework is used. An example of the input is /eos/user/<username      first letter>/<user name> . In theory AFS work location also can be specified but it is not recommended.
14) The installer will copy and analyse existing data and the pre-trained model, it might take 5-10 minutes.
15) if the message 'EDER-TSU setup is successfully completed' is displayed, it means that the package is ready for work

Additional info
--
1) It is recommended to run those processes on lxplus in the tmux shell as some scripts can take up to several hours to execute.
2) The script name prefixes indicate what kind of operations this script perform: R is for actual reconstruction routines, E for evaluation and M for model creation and training.
3) In general the numbers in prefixes reflect the order at which scripts have to be executed e.g: R1, R2,R3...
4) --help argument provides all the available run arguments of a script and its purpose.
5) The output of each script has the same prefix as the script that generates it. If script generates a temporary output for another script it will have the double prefix e.g: R2_R3 etc.
6) The files that are the final output have names with capital letters only such as: R6_REC_AND_GLUED_TRACKS.csv
   Those files are not deleted after execution. If not all letters in the file are capitalised that means that the file is temporary and will be eventually deleted by the package once it is not needed anymore.
7) The screen output of the scripts is colour coded: 
   - White for routine operations
   - Blue for the file and folder locations
   - Green for successful operation completions
   - Yellow for warnings and non-critical errors.
   - Red for critical errors.
8) Once the program successfully executes it will leave a following message before exiting: 
   "###### End of the program #####"


Track Segment Gluing Process
--
1) Please make sure that you have a file with hits that there were reconstructed as Tracks.
   Following columns are required: 
   - Track ID Quadrant
   - FEDRA Track ID
   - x-coordinates of the track hits
   - y-coordinates of the track hits
   - z-coordinates of the track hits

2) Please open $AFS/EDER_TSU/Code/Utilities/Parameters.py and check that the lines between 6-13 
   Within the list of naming conventions correspond to headers in the file that you intend to use.

3) Check the 'Pre_CNN_Model_Name' variable - it has the name of the Model that is used for reconstruction (included in the package). If you wish to use your own,        please place it in the $EOS/EDER_TSU/Models and change the 'Pre_CNN_Model_Name' variable accordingly. You might need to change resolution and MaxX, MaxY, MaxZ      parameters if the model was trained with images that have had different size because the model might fail.

4) If happy, save and close the file.

5) cd ..

6) tmux 
   
   please note the number of lxplus machine at which tmux session is logged in

7) kinit your<username>@CERN.CH -l 24h00m

8) python3 R1_PrepareRecData.py --Xmin 50000 --Xmax 60000 --Ymin 50000 --Ymax 60000 --f (your file with reconstructed tracks) 
   
   Purpose: This script prepares the reconstruction data for EDER-TSU gluing routines by using the custom file with track reconstruction data
   
   FYI: min and max value arguments can be changed or completely removed if all ECC data to be reconstructed. The script can take 1-5 minutes depending on the size of the input file. Once it finish it will give the message "The          track    data has been created successfully and written to ....' and exit.

9) python3 R2_GenerateTracks.py --Mode R
   
   Purpose: This script selects and prepares 2-segments track seed candidates that could be used for gluing. The seeds are subject to distance cuts
   FYI: The script will send warning, type Y. The program will send HTCondor jobs and exit. The jobs take about an hour.

10) python3 R2_GenerateTracks.py --Mode C
    
    FYI: It will check whether the HTCondor jobs have been completed, if not it will give a warning. If the jobs are completed it will remove duplicates from the seeds and generate the following message: "Track seed generation is completed".

11) python3 R3_FilterTracks.py --Mode R
    
    Purpose: This script takes preselected 2-segment track candidates from previous step and refines them by applying additional cuts on the parameters such as DOCA and mutual angle.
   FYI: The script will send warning, type Y. The program will send HTCondor jobs and exit. The jobs can take few hours.

12) python3 R3_FilterTracks.py --Mode C 
    
    FYI: It will check whether the HTCondor jobs have been completed, if not it will give a warning.

13) python3 R4_FitTracks.py --Mode R
    
    Purpose: This script takes refined 2-segment track candidates from previous step and performs a vertex fit by using pre-trained CNN model(s).
    FYI: The script will send warning, type Y. The program will send HTCondor jobs and exit. The jobs can take few hours.
    This script can run with two CNN models simultaneously. This can be enabled by option --ReFit Y If the second model is present then it has to be located in the model folder and its name has to be specified in the Parameters.py under Post_CNN_Model_Name.
    The first model by default is specified in Parameters.py by parameter Pre_CNN_Model_Name

14) python3 R4_FitTracks.py --Mode C 
   
    FYI: It will check whether the HTCondor jobs have been completed, if not it will give a warning.
    If you have run previously with the option --ReFit Y you have to specify it here too.
   
15) python3 R5_GlueTrackSegments.py --Mode R
    
    Purpose: This script takes CNN-fitted 2-segment track candidates from previous step and merges them if tracks have a common segment.
    
    FYI: The script will send warning, type Y. The program will send HTCondor jobs and exit. The jobs can take few hours.

16) python3 R5_GlueTrackSegments.py --Mode C
    If all HTCondor jobs finished, it will ask whether you want to perform final merging. Press F and it will continue on the lxplus machine. The execution can take up to a day if the data size is big.
    The program will produce the R5_GLUED_TRACKS.pkl (Its just FYI, you don't really need it) and R5_GLUED_TRACKS.csv (This is the mapping file that you need in order to remap TrackID data).

17) R6_MapTracksToSegments.py -f {Your Original file}
    
    Purpose: This script remaps your original track reconstruction file with output from the previous step. All tracks that have been glued will have 'TSU' as Quarter.
    FYI: The execution can take up to several hours if the data size is big. The program will produce the R6_REC_AND_GLUED_TRACKS.csv. 
    You can use this file as an input for EDER-VIANN.
   
   
EDER-TSU Track de-segmentation evaluation
--
Can only be used if there is a data available with MC track truth information.
   
1) python3 E1_PrepareEvalData.py --Xmin 50000 --Xmax 60000 --Ymin 50000 --Ymax 60000  --f (your file with reconstructed tracks)
   
   Purpose: This script prepares the MC tracking data for EDER-TSU de-segmentation evaluation routines.
   FYI: min and max value arguments have to match those that were used in for previous phase in Step 8.
   The script can take 1-5 minutes depending on the size of the input file.
   Once it finishes it will give the message "The track segment data has been created successfully and written to ....' and exit.

2) python3 E2_GenerateEvalTracks.py --Mode R
   
   Purpose: This script selects and prepares 2-segment track seeds that have a common MC track id.
   The script will send warning, type Y. The program will send HTCondor jobs and exit. The jobs take about an hour.

3) python3 E2_GenerateEvalTracks.py --Mode C
   
   FYI: It will check whether the HTCondor jobs have been completed, if not it will give a warning.
   If the jobs are completed it will remove duplicates from the seeds and generate the following message: "Track segment seed generation is completed".
   
4) python3 E3_DecorateEvalTracks.py --Mode R 
   
   Purpose: This script takes preselected 2-segment track seeds and decorates them with additional information such as DOCA and mutual angle.
   FYI: The script will send warning, type Y. 
   The program will send HTCondor jobs and exit. 
   The jobs take about an hour.
   
5) python3 E3_DecorateEvalTracks.py --Mode C
   
   FYI: It will check whether the HTCondor jobs have been completed, if not it will give a warning.
   The output will generate the file E3_TRUTH_TRACKS.csv that contains all segment track seeds that have a common Segment ID. 
   This file has additional information on the tracks such as opening angle, DOCA etc. 
   This file is used to assess the performance of the EDER-TSU and FEDRA track reconstruction accuracy.
   
6) python3 E4_EvaluateRecData.py 
   
   Purpose: This script compares the output of the previous step with the output of EDER-TSU reconstructed data to calculate gluing performance.
   FYI: The script will return the precision and the recall of the EDER-TSU reconstruction output
   The script can be run with option '--Acceptance'  which takes in account only the seeds with probability above the given value (has to be between 0 and 1).
   If you just want to test CNN, use option --TypeOfAnalysis CNN
   If you want to test FEDRA then use option --TypeOfAnalysis TRACKING
   If you want to test both then use option --TypeOfAnalysis ALL
   
   
   
EDER-TSU Model Training
--
Can only be used if there is a data available with MC vertex truth information.

1) python3 M1_PrepareTrainData.py --Xmin 50000 --Xmax 120000 --Ymin -120000 --Ymax 50000  --f (your file with reconstructed tracks)
   
    Purpose: This script prepares the MC tracking data for EDER-TSU training routines
    FYI: min and max value arguments can be changed or completely removed if all ECC data to be used for training. 
    The X and Y bounds are exclusive (they define the portion of the ECC data that is not used in training).
    The script can take 1-5 minutes depending on the size of the input file. 
    Once it finishes it will give the message "The track segment data has been created successfully and written to ....' and exit.

2) python3 M2_GenerateTrainSeeds.py --Mode R 
   
    Purpose: This script selects and prepares 2-segment track seeds that have either a common MC Track (True label) or do not have a common MC Track (False label). 
    FYI: The script will send warning, type Y. 
    The program will send HTCondor jobs and exit. 
    The jobs take about an hour.

3) M2_GenerateTrainSeeds.py --Mode C 
    
    FYI: It will check whether the HTCondor jobs have been completed, if not it will give a warning.

4) M3_GenerateImages.py --Mode R 
    
    Purpose: This script takes the output from the previous step and decorates the track with its hit information that can be used to render the seed image. This script creates training and validation samples.
    FYI: The script will send warning, type Y. 
    The program will send HTCondor jobs and exit. 
    The jobs take about an hour.
    This script can run with additional option: --PreFit Y.  If the model is present then it has to be located in the model folder and its name has to be specified in the Parameters.py under Pre_CNN_Model_Name.
    This will enable to do a more rigorous selection of Seeds with a help of CNN.

5) M3_GenerateImages.py --Mode C 
    
    FYI: It will check whether the HTCondor jobs have been completed, if not it will give a warning.
    
    Important! In the end the script will ask 'Would you like to delete track seeds data?'. Please type 'N'. We will need the filtered seeds again for later. 
    If you have run on first time with --PreFit Y option than you have to specify it here too.
   
6) M4_RenderImages.py --Mode R 
    
    Purpose: This script takes the seed from the previous step and render their. This script modifies training and validation samples.
    FYI: The script will send warning, type Y. 
    The program will send HTCondor jobs and exit. 
    The jobs take about an hour.

7) M4_RenderImages.py --Mode C 
    
    FYI: It will check whether the HTCondor jobs have been completed, if not it will give a warning.
  
8) python3 M5_TrainModel.py --Mode R
    If the option --ModelNewName has been specified than a new model will be generated by using the parameter ModelArchitecture in the Parameters.py.
    The master script will check whether it is possible to compile this model, if yes it will send an HTCondor job and exit. Otherwise, it will show 'Fail' message and exit.
    The job takes about 4-5 hours.

9) python3 M5_TrainModel.py --Mode C
    
    FYI: It will check whether the HTCondor job has been completed, if not it will give a warning.
    If the job has been completed the script will ask the user whether he wants to continue (N/Y).
    The model training performance (loss and accuracy) will be saved in /EDER-TSU/Models/M5_PERFORMANCE_ModelName.csv file
