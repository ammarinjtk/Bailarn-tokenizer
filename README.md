# Tokenizer

API contains Tokenizer for Thai Text Platform as a senior project for Computer Engineering, Chulalongkorn University (CP41)

## Short introduction
```
>>> from Tokenizer import *
>>> bailarn_tokenizer = Bailarn_Tokenizer(model_path="./models/0014-0.0443.hdf5")
>>> print(bailarn_tokenizer.predict(sentence="ฉันกินข้าว"))
[['ฉัน', 'กิน', 'ข้าว']]
>>> print(bailarn_tokenizer.predict(sentence="ฉันกินข้าว")[0])
['ฉัน', 'กิน', 'ข้าว']
```

## Authors

* **Ammarin Jetthakun** - *Initial work* - [ammarinjtk](https://github.com/ammarinjtk)

See also the list of [contributors](https://github.com/ammarinjtk/Tokenizer/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License

## Acknowledgement

* The code for this project is inspired from [KenjiroAI, github](https://github.com/KenjiroAI/SynThai)
* The model had trained InterBEST 2009/2010 (size: 5M words) supported by [NECTEC](https://www.nectec.or.th/corpus/index.php?league=pm)
