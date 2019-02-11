# Audio emregency
User hums to create an audio emergency signal

## Procedure

Run `python emerg_record.py` to record a 5s clip of the user humming. This generates output.wav in same directory.   
Run `python emerg_evaluate.py output.wav` to process this sample and recognize the dominant frequencies.   
Edit *emerg_test.py* line 117 and the contents of this array to hold the frequencies found in the above step.   
rin `python emerg_test.py` to start up the live testing process. Output is cluttered with ALSA trash messages, ignore it. In another terminal, run `tail -f result.csv` to get a cleaner output.   
If hum is recognised correctly, you see a string "hum" at the end of the output line. If talking is recognised, you see a string "talk".   

### TODO
Oh my god so much more... hmm lets see..   
General goals:   
1)Test more poeple   
2)Record and evaluate robot noises   
3)Noise suppression

### Investigate
1) Pursue stackoverflow suggestion   
2) Investigate Dynamic Tune Window approach   
3) Follow up with Prof. Tanjy Schultz's work.   


