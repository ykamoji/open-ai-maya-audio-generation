## Open AI TTS implementation for Microsoft Graph API 

Using Open AI model gpt-4o-mini-tts to convert pages from oneNote using GraphAPI


### Dependencies
```
msgraph-core
python-dotenv
openai==2.7.1
pydub==0.25.1
transformers==4.57.1
vllm
snac
```


### Start Up
```
!pip3 install -r requirements.txt
```

### Configuration

#### Initiation Args
```
./start.sh --refreshPages True --graphPgL 1
```

#### Voice Generation Args
```
./start.sh --config Kaggle --pageNums "[15]" --genPgL 1 --step "[0]"
```

