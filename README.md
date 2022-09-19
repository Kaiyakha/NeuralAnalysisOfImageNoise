# Neural Analysis of Image Noise
## Build and Run
1. Build Neural Network module with Visual Studio (build) in Release mode.
2. Use _PyInstaller_ to build an executable:
`py -m PyInstaller Handle\CLI.py --paths NeuralNetwork\x64\Release --distpath NCLI -n ncli --onefile`
3. Place config.ini into the NCLI folder.
4. Run the app via CMD: `ncli <COMMAND> [OPTIONS]`
