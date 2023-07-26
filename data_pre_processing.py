
def pre_process_training_data(ft):
    data = pd.read_excel('https://github.com/eliastreis/Test/blob/main/training.xlsx?raw=true') 
    data = data[data['Std'] < 2] # Std >2: Daten nicht berücksichtigt (betrifft nur wenige)
    data = data.append(data[data['Std'] <= 1 ]) #std kleiner 1 doppelt
    data = data.append(data[data['Std'] <= 0.5]) #std kleiner 0,5 vierfach
    
    data.sort_values("MOS", inplace=True)
    data["split_sentences"] = data["Sentence"].str.split(" ")
    #data["split_sentences"] = data["Sentence"].apply((lambda x: tf.keras.preprocessing.text.text_to_word_sequence(x, lower=False)))

    # convert every sentence into vector representation using fastText
    data["vector_sentences"] = data["split_sentences"].apply(lambda x: [ft.get_word_vector(e) for e in x])
    data["numeric_features"] = data["Sentence"].apply(extract_numerical_features)

    # 2/3 training, 1/3 validation data sets
    training = data[data.index % 3 != 0]
    validation = data[data.index % 3 == 0]

    # shuffle based on seed
    training = shuffle(training, random_state=784128105)
    validation = shuffle(validation, random_state=2098634387)


    # num of elements (
    # num elements (600, 300, 300)
    training_x1 = np.array([np.array(x) for x in training["vector_sentences"].tolist()])
    training_x2 = np.array(training["numeric_features"].tolist())
    training_y = np.array(training["MOS"].tolist())
    validation_x1 = np.array([np.array(x) for x in validation["vector_sentences"].tolist()])
    validation_x2 = np.array(validation["numeric_features"].tolist())
    validation_y = np.array(validation["MOS"].tolist())

    return (training_x1, training_x2, training_y), (validation_x1, validation_x2, validation_y)

