decompress_original:
	tar xvf data/character.tar.Z -C data/original_data/
	cp data/original_data/murphy/learn.zip data/original_data/
	cp data/original_data/murphy/test.zip data/original_data/
	unzip data/original_data/learn.zip -d data/original_data/learn
	unzip data/original_data/test.zip -d data/original_data/test
	rm -rf data/original_data/murphy
	rm data/original_data/learn.zip
	rm data/original_data/test.zip
