# Audio emregency
User hums to create an audio emergency signal    

SWITCHED TO ROS AT THIS POINT   

## Procedure

Run `python emerg_record.py <name>` to record a 3s clip of the user humming. This generates <name>.wav in the output directory and subsequently calls the evaluation script on this wav file   
Optionally run `python emerg_evaluate.py <name>` to process <name>.wav file in output directory and recognize the dominant frequencies.   
Either process generates <name>_freqs.csv in output directory containing dominamt frequencies.   
Run `python emerg_test.py <name>` to start up the live testing process. This requires a <name>_freqs.csv in output directory with correct dominant frequencies. Output is cluttered with ALSA trash messages, ignore it. In another terminal, run `tail -f result.csv` to get a cleaner output.   
If hum is recognised correctly, you see a string "hum" at the end of the output line. If talking is recognised, you see a string "talk".   

### TODO
1) Include everything into ROS   
2) Have a topic publish info and read that instead of tail -f result.csv   

### Investigate
1) Why batch processing gives different results between realtime and wave file execution   


