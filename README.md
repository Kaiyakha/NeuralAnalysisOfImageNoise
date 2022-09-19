# Neural Analysis of Image Noise
## Build
1. Build Neural Network module with Visual Studio (build) in Release mode.
2. Use _PyInstaller 3.8_ to build an executable:
`py -3.8 -m PyInstaller Handle\CLI.py --paths NeuralNetwork\x64\Release --distpath NCLI -n ncli --onefile`
3. Place config.ini into the NCLI folder.
## Run
Run the app via CMD: `ncli [OPTIONS] <COMMAND> [ARGS]`.
For further instructions, see [reference](https://github.com/Kaiyakha/NeuralAnalysisOfImageNoise-CppBased/blob/master/%D0%92%D0%9A%D0%A0.docx) (_ВКР.docx_, page 56).
