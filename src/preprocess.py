



# Training function

def model_trainer(X_train, y_train, output_path):
    '''
    Train the model and store it
    :params: X_train, y_train, output_path
    :return: None
    '''
    # initialise the model
    rfor = RandomForestClassifier(max_depth=25, 
                                  min_samples_split=5, 
                                  n_estimators=300,
                                  random_state=8)
       
    # train the model
    rfor.fit(X_train, y_train)
    
    # save the model
    initial_type = [('features_input', FloatTensorType([1, X_train.shape[1]]))]
    onnx = convert_sklearn(rfor, name='rf_champion', initial_types=initial_type)
    with open(output_path, "wb") as f:
        f.write(onnx.SerializeToString())
        f.close()
    return None

#Scoring function
def model_scorer(X, model, index):
    '''
    Score new data with onnx
    :params: X, model, index
    :return: list
    '''
    sess = rt.InferenceSession(model)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    row_to_score = pd.DataFrame([X.iloc[index]])

    score = np.array(row_to_score, dtype=np.float32)
    predictions_onnx = sess.run([label_name], {input_name: score})
    return predictions_onnx[0]




    
